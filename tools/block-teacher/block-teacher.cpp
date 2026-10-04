// Persistent, bounded target-only TRAIN teacher. Each request resets KV and
// teacher-forces exact caller token IDs. No drafter, sampling or optimizer.
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
using json = nlohmann::json;
namespace fs = std::filesystem;
struct placement {
    std::set<std::string> result_buffers;
    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        if (std::string(tensor->name) != "result_output") return false;
        if (!ask && tensor->buffer) {
            auto & state = *static_cast<placement *>(data);
            state.result_buffers.insert(ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer)));
        }
        return true;
    }
};
int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: llama-block-teacher TARGET.gguf OUTPUT_ROOT MAX_TOKENS GPU_LAYERS (JSONL stdin)");
        const int max_tokens = std::stoi(argv[3]), gpu_layers = std::stoi(argv[4]);
        if (max_tokens <= 0 || max_tokens > 32768 || gpu_layers < 0 || gpu_layers > 999) throw std::runtime_error("invalid bounded geometry");
        const fs::path root = fs::absolute(argv[2]);
        fs::create_directories(root);
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = gpu_layers;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        if (!model) throw std::runtime_error("target load failed");
        placement execution;
        auto cp = llama_context_default_params();
        cp.cb_eval = placement::callback;cp.cb_eval_user_data = &execution;
        cp.n_ctx = max_tokens; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = 1;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if (!ctx) throw std::runtime_error("target context failed");
        const int hidden = llama_model_n_embd(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        json target_storage = json::object();
        for (const auto & item : model->tensors_by_name) {
            if (!item.second->buffer) throw std::runtime_error("unallocated target model tensor");
            const auto name = ggml_backend_buft_name(ggml_backend_buffer_get_type(item.second->buffer));
            const size_t count = target_storage.contains(name) ? target_storage[name].get<size_t>() : 0;
            target_storage[name] = count + 1;
        }
        json hardware=json::array();
        for (size_t i=0;i<ggml_backend_dev_count();++i) hardware.push_back(ggml_backend_dev_description(ggml_backend_dev_get(i)));
        std::string line;
        while (std::getline(std::cin,line)) {
            if (line.size() > 2*1024*1024) throw std::runtime_error("request exceeds JSON byte cap");
            const auto request=json::parse(line);
            if (!request.is_object() || request.size()!=4 || !request.contains("id") || !request.contains("tokens") || !request.contains("tap_ids") || !request.contains("logits_mode")) throw std::runtime_error("invalid teacher request fields");
            const auto id=request.at("id").get<std::string>();
            if (id.empty() || id.size()>128 || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos) throw std::runtime_error("invalid output id");
            auto tokens=request.at("tokens").get<std::vector<llama_token>>();
            const auto ids=request.at("tap_ids").get<std::vector<uint32_t>>();
            const auto mode=request.at("logits_mode").get<std::string>();
            if (tokens.empty() || tokens.size()>size_t(max_tokens) || ids.size()!=5 || std::set<uint32_t>(ids.begin(),ids.end()).size()!=5 || (mode!="all" && mode!="last")) throw std::runtime_error("invalid token/tap/logit bounds");
            for (auto token:tokens) if (token<0 || token>=vocab) throw std::runtime_error("token outside target vocabulary");
            for (auto tap:ids) {
                if (tap>=uint32_t(llama_model_n_layer(model.get()))) throw std::runtime_error("tap outside native layer-input range");
                llama_set_embeddings_layer_inp(ctx.get(),tap,true);
            }
            const auto directory=root/id;
            if (!fs::create_directory(directory)) throw std::runtime_error("output id exists; refusing overwrite");
            std::ofstream features(directory/"features.f32",std::ios::binary), logits(directory/"logits.f32",std::ios::binary);
            if (!features || !logits) throw std::runtime_error("cannot open teacher outputs");
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            execution.result_buffers.clear();
            for (size_t off=0;off<tokens.size();off+=256) {
                const int n=int(std::min(size_t(256),tokens.size()-off));
                auto batch=llama_batch_init(n,0,1);
                batch.n_tokens=n;
                for(int i=0;i<n;++i) {
                    batch.token[i]=tokens[off+i];batch.pos[i]=off+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=true;
                }
                const int rc=llama_decode(ctx.get(),batch);
                llama_batch_free(batch);
                if(rc) throw std::runtime_error("native target decode failed code="+std::to_string(rc));
                std::vector<const float*> taps;
                for(auto tap:ids) {
                    const auto * data=llama_get_embeddings_layer_inp(ctx.get(),tap);
                    if(!data) throw std::runtime_error("native layer-input extraction missing");
                    taps.push_back(data);
                }
                for(int i=0;i<n;++i) {
                    for(const auto * data:taps) features.write(reinterpret_cast<const char*>(data+size_t(i)*hidden),size_t(hidden)*sizeof(float));
                    if(mode=="all" || off+i+1==tokens.size()) {
                        const float * row=llama_get_logits_ith(ctx.get(),i);
                        if(!row) throw std::runtime_error("full-vocabulary native logits missing");
                        logits.write(reinterpret_cast<const char*>(row),size_t(vocab)*sizeof(float));
                    }
                }
            }
            features.close();logits.close();
            if(!features || !logits) throw std::runtime_error("teacher output write failed");
            for(auto tap:ids) llama_set_embeddings_layer_inp(ctx.get(),tap,false);
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            const json receipt={{"schema","block_native_teacher_request_v1"},{"id",id},{"tokens",tokens},{"tap_ids",ids},
                {"features_shape",{tokens.size(),5,hidden}},{"logits_shape",{mode=="all"?tokens.size():size_t(1),size_t(vocab)}},
                {"logits_mode",mode},{"hardware",hardware},{"target_storage_buffers",target_storage},
                {"executed_result_buffers",execution.result_buffers},{"gpu_layers",gpu_layers},{"kv_type","F16"},
                {"prefix_contract","teacher_forced_exact_caller_token_ids"},{"optimizer_updates",0},{"complete",true}};
            std::ofstream meta(directory/"native-receipt.json");meta<<receipt.dump(2)<<'\n';meta.close();
            if(!meta) throw std::runtime_error("receipt write failed");
            std::cout<<receipt.dump()<<std::endl;
        }
        ctx.reset();model.reset();llama_backend_free();
    } catch(const std::exception & e) {std::cerr<<e.what()<<std::endl;return 1;}
    return 0;
}
