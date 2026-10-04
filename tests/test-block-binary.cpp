#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <map>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
using json=nlohmann::json;
struct observed { int dots=0; std::map<std::string,json> nodes; };
static bool capture(ggml_tensor * tensor,bool ask,void * ptr) {
    if(tensor->op!=GGML_OP_W1A1_MUL_MAT)return false;
    if(!ask) {
        auto & state=*static_cast<observed*>(ptr);++state.dots;
        auto * buffer=tensor->buffer;
        const std::string device=buffer?ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)):"unallocated";
        const std::string packed=tensor->src[0]->name;
        state.nodes[packed]={{"packed",packed},{"activation_bits",tensor->op_params[2]},
            {"output_buffer",device},{"output_shape",{tensor->ne[0],tensor->ne[1]}},
            {"packed_type",ggml_type_name(tensor->src[0]->type)}};
    }
    return true;
}
int main(int argc,char **argv) {
    if(argc!=2 && argc!=4)return 2;
    const int gpu_layers=argc==4?std::stoi(argv[2]):0;
    if(gpu_layers<0 || gpu_layers>999)return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!model)return 1;
    observed operations;
    auto cp=llama_context_default_params();cp.cb_eval=capture;cp.cb_eval_user_data=&operations;cp.n_ctx=128;cp.n_batch=7;cp.n_ubatch=7;cp.n_seq_max=1;
    cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
    if(!ctx)return 1;
    const int h=llama_model_n_embd(model.get()), v=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    auto features=llama_batch_init(3,5*h,1);features.n_tokens=3;
    for(int i=0;i<3;++i) {
        for(int k=0;k<5*h;++k)features.embd[i*5*h+k]=0.01f*float((i+k)%11-5);
        features.pos[i]=i;features.n_seq_id[i]=1;features.seq_id[i][0]=0;features.logits[i]=false;
    }
    int rc=llama_decode(ctx.get(),features);llama_batch_free(features);if(rc)return 1;
    auto noise=llama_batch_init(7,0,1);noise.n_tokens=7;
    for(int i=0;i<7;++i) {noise.token[i]=i?1:2;noise.pos[i]=3+i;noise.n_seq_id[i]=1;noise.seq_id[i][0]=0;noise.logits[i]=true;}
    rc=llama_decode(ctx.get(),noise);llama_batch_free(noise);if(rc)return 1;
    for(int i=0;i<7;++i) {
        auto * logits=llama_get_logits_ith(ctx.get(),i);if(!logits)return 1;
        for(int k=0;k<v;++k)if(!std::isfinite(logits[k]))return 1;
    }
    if(operations.dots<15)return 1;
    json nodes=json::array();for(const auto & node:operations.nodes)nodes.push_back(node.second);
    json hardware=json::array();for(size_t i=0;i<ggml_backend_dev_count();++i)hardware.push_back(ggml_backend_dev_description(ggml_backend_dev_get(i)));
    if(argc==4) {
        std::ifstream existing(argv[3]);if(existing)return 2;
        std::ofstream proof(argv[3]);proof<<json{{"schema","block_native_graph_smoke_v1"},{"passed",true},
            {"gpu_layers",gpu_layers},{"hardware",hardware},{"nodes",nodes},{"instrumented",true},
            {"selected_dense_fallback",false},{"optimizer_updates",0},{"quality_evaluation",false}}.dump(2)<<'\n';
        proof.close();if(!proof)return 1;
    }
    ctx.reset();model.reset();llama_backend_free();std::cout<<"block binary injection/noise graph PASS dots="<<operations.dots<<"\n";return 0;
}
