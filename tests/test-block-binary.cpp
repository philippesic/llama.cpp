#include "llama.h"
#include "ggml.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
struct observed { int dots=0; };
static bool capture(ggml_tensor * tensor,bool ask,void * ptr) {
    if(tensor->op!=GGML_OP_W1A1_MUL_MAT)return false;
    if(!ask)++static_cast<observed*>(ptr)->dots;
    return true;
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
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
    ctx.reset();model.reset();llama_backend_free();std::cout<<"block binary injection/noise graph PASS dots="<<operations.dots<<"\n";return 0;
}
