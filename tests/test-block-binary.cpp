#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <map>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
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
    if(argc!=2 && argc!=4 && argc!=5)return 2;
    const int gpu_layers=argc>=4?std::stoi(argv[2]):0;
    if(gpu_layers<0 || gpu_layers>999)return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!model)return 1;
    observed operations;
    const bool author_layout=std::getenv("DSPARK_REQUIRE_AUTHOR_LAYOUT")!=nullptr;
    auto cp=llama_context_default_params();cp.cb_eval=capture;cp.cb_eval_user_data=&operations;cp.n_ctx=128;cp.n_batch=author_layout?32:7;cp.n_ubatch=cp.n_batch;cp.n_seq_max=1;
    if(author_layout) {cp.n_outputs_max=7;cp.n_outputs_max_per_seq=7;}
    cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> target_model(nullptr,llama_model_free);
    std::unique_ptr<llama_context,decltype(&llama_free)> target_context(nullptr,llama_free);
    json target_binding=nullptr;
    if(argc==5) {
        target_model.reset(llama_model_load_from_file(argv[4],mp));if(!target_model)return 1;
        auto tp=llama_context_default_params();tp.n_ctx=128;tp.n_batch=7;tp.n_ubatch=7;
        tp.type_k=GGML_TYPE_F16;tp.type_v=GGML_TYPE_F16;
        target_context.reset(llama_init_from_model(target_model.get(),tp));if(!target_context)return 1;
        auto * target=target_model.get();
        if(!model->tok_embd || !model->output || !target->tok_embd || !target->output)return 1;
        for(auto * owned : {model->tok_embd,model->output}) for(auto * teacher : {target->tok_embd,target->output}) {
            if(owned==teacher || owned->data==teacher->data)return 1;
        }
        if(model->tok_embd->ne[0]!=target->tok_embd->ne[0] || model->tok_embd->ne[1]!=target->tok_embd->ne[1] ||
            model->output->ne[0]!=target->output->ne[0] || model->output->ne[1]!=target->output->ne[1])return 1;
        if(model->tok_embd->ne[1]==151936 && (model->tok_embd->type!=GGML_TYPE_BF16 || model->output->type!=GGML_TYPE_BF16 ||
            target->tok_embd->type!=GGML_TYPE_F16 || target->output->type!=GGML_TYPE_F16))return 1;
        target_binding={{"private_embedding_and_head_distinct_from_target",true},
            {"draft_embedding_type",ggml_type_name(model->tok_embd->type)},{"draft_head_type",ggml_type_name(model->output->type)},
            {"target_embedding_type",ggml_type_name(target->tok_embd->type)},{"target_head_type",ggml_type_name(target->output->type)},
            {"embedding_shape",{model->tok_embd->ne[0],model->tok_embd->ne[1]}},
            {"head_shape",{model->output->ne[0],model->output->ne[1]}},
            {"target_layer_count",llama_model_n_layer(target)},{"ordered_native_target_taps",model->target_layer_ids}};
        cp.ctx_other=target_context.get();
    }
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
    if(!ctx)return 1;
    if(llama_n_batch(ctx.get())!=cp.n_batch || llama_n_ubatch(ctx.get())!=cp.n_ubatch)return 1;
    const int h=llama_model_n_embd(model.get()), v=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const auto mask=llama_vocab_mask(llama_model_get_vocab(model.get()));
    if(mask<0 || mask>=v) {std::cerr<<"declared model MASK token missing\n";return 1;}
    auto features=llama_batch_init(3,5*h,1);features.n_tokens=3;
    for(int i=0;i<3;++i) {
        for(int k=0;k<5*h;++k)features.embd[i*5*h+k]=0.01f*float((i+k)%11-5);
        features.pos[i]=i;features.n_seq_id[i]=1;features.seq_id[i][0]=0;features.logits[i]=false;
    }
    int rc=llama_decode(ctx.get(),features);llama_batch_free(features);if(rc)return 1;
    auto noise=llama_batch_init(7,0,1);noise.n_tokens=7;
    for(int i=0;i<7;++i) {noise.token[i]=i?mask:2;noise.pos[i]=3+i;noise.n_seq_id[i]=1;noise.seq_id[i][0]=0;noise.logits[i]=true;}
    rc=llama_decode(ctx.get(),noise);llama_batch_free(noise);if(rc)return 1;
    for(int i=0;i<7;++i) {
        auto * logits=llama_get_logits_ith(ctx.get(),i);if(!logits)return 1;
        for(int k=0;k<v;++k)if(!std::isfinite(logits[k]))return 1;
    }
    if(author_layout) {
        auto oversized=llama_batch_init(8,0,1);oversized.n_tokens=8;
        for(int i=0;i<8;++i) {oversized.token[i]=i?mask:2;oversized.pos[i]=10+i;oversized.n_seq_id[i]=1;oversized.seq_id[i][0]=0;oversized.logits[i]=i==7;}
        bool rejected=false;
        try {llama_decode(ctx.get(),oversized);}
        catch(const std::runtime_error & error) {rejected=std::string(error.what())=="DFlash W1Ax Markov block exceeds trained block size";}
        llama_batch_free(oversized);if(!rejected)return 1;
    }
    if(operations.dots<15)return 1;
    json nodes=json::array();for(const auto & node:operations.nodes)nodes.push_back(node.second);
    json hardware=json::array();for(size_t i=0;i<ggml_backend_dev_count();++i)hardware.push_back(ggml_backend_dev_description(ggml_backend_dev_get(i)));
    if(argc>=4) {
        std::ifstream existing(argv[3]);if(existing)return 2;
        std::ofstream proof(argv[3]);proof<<json{{"schema","block_native_graph_smoke_v1"},{"passed",true},
            {"gpu_layers",gpu_layers},{"hardware",hardware},{"nodes",nodes},{"instrumented",true},
            {"author_reference_layout",author_layout},{"draft_n_batch",llama_n_batch(ctx.get())},{"draft_n_ubatch",llama_n_ubatch(ctx.get())},
            {"oversized_noise_guard_checked",author_layout},
            {"paired_target_geometry_checked",argc==5},{"target_binding",target_binding},{"mask_token_id",mask},{"anchor_token_id",2},
            {"noise_input_token_ids",{2,mask,mask,mask,mask,mask,mask}},
            {"noise_input_positions",{3,4,5,6,7,8,9}},
            {"selected_dense_fallback",false},{"optimizer_updates",0},{"quality_evaluation",false}}.dump(2)<<'\n';
        proof.close();if(!proof)return 1;
    }
    ctx.reset();target_context.reset();model.reset();target_model.reset();llama_backend_free();std::cout<<"block binary injection/noise graph PASS dots="<<operations.dots<<"\n";return 0;
}
