// Persistent target-only TRAIN producer. Exact-prefix replay and native prompt
// tokenization/greedy continuation share the same frozen target and clean F16 KV.
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cmath>
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
        const auto * vocabulary = llama_model_get_vocab(model.get());
        const int hidden = llama_model_n_embd(model.get());
        const int vocab = llama_vocab_n_tokens(vocabulary);
        json target_storage = json::object(), hardware = json::array();
        for (const auto & item : model->tensors_by_name) {
            if (!item.second->buffer) throw std::runtime_error("unallocated target model tensor");
            const auto name = ggml_backend_buft_name(ggml_backend_buffer_get_type(item.second->buffer));
            const size_t count = target_storage.contains(name) ? target_storage[name].get<size_t>() : 0;
            target_storage[name] = count + 1;
        }
        for (size_t i=0;i<ggml_backend_dev_count();++i) hardware.push_back(ggml_backend_dev_description(ggml_backend_dev_get(i)));
        std::string line;
        while (std::getline(std::cin,line)) {
            if (line.size() > 2*1024*1024) throw std::runtime_error("request exceeds JSON byte cap");
            const auto request=json::parse(line);
            const bool generate=request.contains("prompt");
            if (!request.is_object() || request.size()!=(request.contains("decode_history")?5:4) || (generate && request.contains("decode_history")) || !request.contains("id") ||
                    !request.contains("tap_ids") || !request.contains("logits_mode") ||
                    (generate == request.contains("tokens"))) throw std::runtime_error("invalid teacher request fields");
            const auto id=request.at("id").get<std::string>();
            if (id.empty() || id.size()>128 || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos) throw std::runtime_error("invalid output id");
            const auto ids=request.at("tap_ids").get<std::vector<uint32_t>>();
            const auto mode=request.at("logits_mode").get<std::string>();
            if ((ids.size()!=3 && ids.size()!=5) || std::set<uint32_t>(ids.begin(),ids.end()).size()!=ids.size() ||
                    (mode!="all" && mode!="last" && mode!="none")) throw std::runtime_error("invalid tap/logit bounds");
            std::vector<llama_token> tokens;
            int max_new=0;
            std::string rendered, chat_template, template_mode;
            if (generate) {
                const auto & prompt=request.at("prompt");
                if (!prompt.is_object() || (prompt.size()!=3 && prompt.size()!=4) || !prompt.contains("template_mode") || !prompt.contains("max_new_tokens")) throw std::runtime_error("invalid native prompt fields");
                template_mode=prompt.at("template_mode").get<std::string>();
                if (!prompt.at("max_new_tokens").is_number_integer()) throw std::runtime_error("generation bound must be integer");
                max_new=prompt.at("max_new_tokens").get<int>();
                if (max_new<0 || max_new>max_tokens) throw std::runtime_error("invalid generation bound");
                int max_prompt=max_tokens-max_new;
                if (prompt.contains("max_prompt_tokens")) {
                    if (!prompt.at("max_prompt_tokens").is_number_integer()) throw std::runtime_error("prompt cap must be integer");
                    max_prompt=prompt.at("max_prompt_tokens").get<int>();
                }
                if (max_prompt<=0 || max_prompt>max_tokens || max_new>max_tokens-max_prompt) throw std::runtime_error("invalid native prompt token cap");
                if (template_mode=="raw_text" && prompt.contains("text")) {
                    rendered=prompt.at("text").get<std::string>();
                } else if (template_mode=="native_chat" && prompt.contains("messages")) {
                    const auto & messages=prompt.at("messages");
                    if (!messages.is_array() || messages.empty() || messages.size()>1024) throw std::runtime_error("invalid native chat messages");
                    const auto * model_template=llama_model_chat_template(model.get(),nullptr);
                    if (!model_template) throw std::runtime_error("native model chat template is missing");
                    chat_template=model_template;
                    std::vector<std::string> roles, contents;
                    for (const auto & message:messages) {
                        if (!message.is_object() || message.size()!=2 || !message.contains("role") || !message.contains("content")) throw std::runtime_error("native chat requires role/content fields");
                        roles.push_back(message.at("role").get<std::string>());
                        contents.push_back(message.at("content").get<std::string>());
                    }
                    std::vector<llama_chat_message> chat;
                    for (size_t i=0;i<roles.size();++i) chat.push_back({roles[i].c_str(),contents[i].c_str()});
                    int n=llama_chat_apply_template(chat_template.c_str(),chat.data(),chat.size(),true,nullptr,0);
                    if (n<=0 || n>2*1024*1024) throw std::runtime_error("native chat template unsupported or exceeds cap");
                    std::vector<char> text(n+1);
                    const int observed=llama_chat_apply_template(chat_template.c_str(),chat.data(),chat.size(),true,text.data(),text.size());
                    if (observed!=n) throw std::runtime_error("native template size changed");
                    rendered.assign(text.data(),n);
                } else throw std::runtime_error("unsupported native prompt/template mode");
                if (rendered.empty() || rendered.size()>2*1024*1024) throw std::runtime_error("empty or excessive native prompt");
                int n=llama_tokenize(vocabulary,rendered.data(),rendered.size(),nullptr,0,true,true);
                if (n<0) n=-n;
                if (n<=0 || n>max_prompt || n+max_new>max_tokens) throw std::runtime_error("native tokenized prompt or continuation exceeds token cap before decode");
                tokens.resize(n);
                const int observed=llama_tokenize(vocabulary,rendered.data(),rendered.size(),tokens.data(),tokens.size(),true,true);
                if (observed!=n) throw std::runtime_error("native tokenization size changed");
            } else {
                for (const auto & token:request.at("tokens")) {
                    if (!token.is_number_integer()) throw std::runtime_error("token prefix must contain integers");
                    tokens.push_back(token.get<llama_token>());
                }
            }
            if (tokens.empty() || tokens.size()>size_t(max_tokens)) throw std::runtime_error("invalid token prefix bounds");
            for (auto token:tokens) if (token<0 || token>=vocab) throw std::runtime_error("token outside target vocabulary");
            for (auto tap:ids) {
                if (tap>=uint32_t(llama_model_n_layer(model.get()))) throw std::runtime_error("tap outside native layer-input range");
                llama_set_embeddings_layer_inp(ctx.get(),tap,true);
            }
            const size_t prompt_length=tokens.size();
            const auto directory=root/id;
            if (!fs::create_directory(directory)) throw std::runtime_error("output id exists; refusing overwrite");
            std::ofstream features(directory/"features.f32",std::ios::binary), logits(directory/"logits.f32",std::ios::binary);
            if (!features || !logits) throw std::runtime_error("cannot open teacher outputs");
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            execution.result_buffers.clear();
            json decode_history=json::array();
            std::vector<float> last_logits(vocab);
            auto decode=[&](size_t off,int n,const char * phase) {
                auto batch=llama_batch_init(n,0,1);batch.n_tokens=n;
                for(int i=0;i<n;++i) {
                    batch.token[i]=tokens[off+i];batch.pos[i]=off+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=true;
                }
                const int rc=llama_decode(ctx.get(),batch);llama_batch_free(batch);
                if(rc) throw std::runtime_error("native target decode failed code="+std::to_string(rc));
                std::vector<const float*> taps;
                for(auto tap:ids) {
                    const auto * data=llama_get_embeddings_layer_inp(ctx.get(),tap);
                    if(!data) throw std::runtime_error("native layer-input extraction missing");
                    taps.push_back(data);
                }
                for(int i=0;i<n;++i) {
                    for(const auto * data:taps) features.write(reinterpret_cast<const char*>(data+size_t(i)*hidden),size_t(hidden)*sizeof(float));
                    const float * row=llama_get_logits_ith(ctx.get(),i);
                    if(!row) throw std::runtime_error("full-vocabulary native logits missing");
                    for(int j=0;j<vocab;++j) if(!std::isfinite(row[j])) throw std::runtime_error("nonfinite native teacher logits");
                    if(mode=="all") logits.write(reinterpret_cast<const char*>(row),size_t(vocab)*sizeof(float));
                    if(i+1==n) std::copy(row,row+vocab,last_logits.begin());
                }
                decode_history.push_back({{"offset",off},{"count",n},{"phase",phase},{"kv_reused_from_same_chain",off!=0}});
            };
            if (request.contains("decode_history")) {
                size_t next=0;
                const auto & history=request.at("decode_history");
                if (!history.is_array() || history.empty()) throw std::runtime_error("empty source decode history");
                for (const auto & chunk:history) {
                    if (!chunk.is_object() || chunk.size()!=4 || !chunk.at("offset").is_number_integer() || !chunk.at("count").is_number_integer()) throw std::runtime_error("invalid source decode history fields");
                    const size_t off=chunk.at("offset").get<size_t>();
                    const int n=chunk.at("count").get<int>();
                    const auto phase=chunk.at("phase").get<std::string>();
                    if (off!=next || n<=0 || n>256 || off+size_t(n)>tokens.size() ||
                            (phase!="prefill" && phase!="target_only_greedy") ||
                            (phase=="target_only_greedy" && n!=1) ||
                            chunk.at("kv_reused_from_same_chain").get<bool>()!=(off!=0)) throw std::runtime_error("invalid source decode partition");
                    decode(off,n,phase.c_str());next+=n;
                }
                if (next!=tokens.size()) throw std::runtime_error("source decode history does not cover exact prefix");
            } else {
                for (size_t off=0;off<prompt_length;off+=256) decode(off,int(std::min(size_t(256),prompt_length-off)),"prefill");
            }
            std::string termination="max_new_tokens";
            for (int generated=0;generated<max_new;++generated) {
                llama_token selected=0;
                for(int j=1;j<vocab;++j) if(last_logits[j]>last_logits[selected]) selected=j;
                tokens.push_back(selected);decode(tokens.size()-1,1,"target_only_greedy");
                if(llama_vocab_is_eog(vocabulary,selected)) {termination="eog";break;}
            }
            if(mode=="last") logits.write(reinterpret_cast<const char*>(last_logits.data()),size_t(vocab)*sizeof(float));
            features.close();logits.close();
            if(!features || !logits) throw std::runtime_error("teacher output write failed");
            for(auto tap:ids) llama_set_embeddings_layer_inp(ctx.get(),tap,false);
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            json receipt={{"schema","block_native_teacher_request_v1"},{"id",id},{"tokens",tokens},{"tap_ids",ids},
                {"features_shape",{tokens.size(),ids.size(),size_t(hidden)}},{"logits_shape",{mode=="all"?tokens.size():mode=="last"?size_t(1):size_t(0),size_t(vocab)}},
                {"logits_mode",mode},{"hardware",hardware},{"target_storage_buffers",target_storage},
                {"executed_result_buffers",execution.result_buffers},{"gpu_layers",gpu_layers},{"kv_type","F16"},
                {"prefix_contract",generate?"native_tokenized_prompt_then_target_only_greedy":"teacher_forced_exact_caller_token_ids"},
                {"decode_history",decode_history},{"optimizer_updates",0},{"complete",true}};
            if(generate) {
                receipt["prompt"]=request.at("prompt");receipt["prompt_length"]=prompt_length;
                receipt["rendered_prompt"]=rendered;receipt["chat_template"]=chat_template;
                receipt["tokenizer"]={{"add_special",true},{"parse_special",true},{"implementation","llama_tokenize"}};
                receipt["generation"]={{"mode","native_target_greedy"},{"max_new_tokens",max_new},
                    {"generated_tokens",tokens.size()-prompt_length},{"termination",termination},{"stop_eog",true}};
            }
            std::ofstream meta(directory/"native-receipt.json");meta<<receipt.dump(2)<<'\n';meta.close();
            if(!meta) throw std::runtime_error("receipt write failed");
            std::cout<<receipt.dump()<<std::endl;
        }
        ctx.reset();model.reset();llama_backend_free();
    } catch(const std::exception & e) {std::cerr<<e.what()<<std::endl;return 1;}
    return 0;
}
