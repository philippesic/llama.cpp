// Tiny CPU loader/encoder fixtures for the versioned learned scalar contract.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"
#include "gguf.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

static gguf_context * fixture(int bits, float delta, float clip, const std::string & invalid = "") {
    auto * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "eagle3");
    gguf_set_val_str(g, "tokenizer.ggml.model", "none");
    for (auto kv : std::vector<std::pair<const char *, uint32_t>>{
            {"vocab_size",16}, {"context_length",32}, {"embedding_length",64},
            {"block_count",1}, {"feed_forward_length",128}, {"attention.head_count",2},
            {"attention.head_count_kv",1}, {"rope.dimension_count",32}, {"target_hidden_size",64}})
        gguf_set_val_u32(g, (std::string("eagle3.")+kv.first).c_str(), kv.second);
    gguf_set_val_f32(g, "eagle3.attention.layer_norm_rms_epsilon", 1e-5f);
    uint32_t layers[] = {0,1,2};
    gguf_set_arr_data(g, "eagle3.target_layers", GGUF_TYPE_UINT32, layers, 3);
    const char * groups[] = {"fusion", "attention", "ffn", "head"};
    gguf_set_arr_str(g, "eagle3.w1a1.groups", groups, 4);
    const char * bases[] = {"fc", "blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v", "blk.0.attn_output", "blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down", "output"};
    const int64_t widths[] = {192,128,128,128,64,64,64,128,64};
    const int64_t rows[] = {64,64,32,32,64,128,128,64,16};
    std::vector<std::string> names; std::vector<const char *> ptrs;
    for (auto base : bases) names.push_back(std::string(base)+".weight");
    for (auto & name : names) ptrs.push_back(name.c_str());
    gguf_set_arr_str(g, "eagle3.w1a1.tensors", ptrs.data(), ptrs.size());
    gguf_set_val_u32(g, "eagle3.w1a1.version", 2);
    gguf_set_val_u32(g, "eagle3.w1a1.scale_group_size", 0);
    gguf_set_val_u32(g, "eagle3.w1a1.activation_bits", bits);
    gguf_set_val_str(g, "eagle3.w1a1.bit_order", "little");
    gguf_set_val_str(g, "eagle3.w1a1.sign_rule", "nonnegative_is_one");
    gguf_set_val_str(g, "eagle3.w1a1.scale_rule", "f32_learned_nonnegative");
    gguf_set_val_str(g, "eagle3.w1a1.arithmetic", "f32");
    ggml_init_params ip = {1024*1024, nullptr, false}; auto * ctx = ggml_init(ip);
    auto tensor = [&](const std::string & name, ggml_type type, int64_t k, int64_t m = 1) {
        auto * t = ggml_new_tensor_2d(ctx, type, k, m); ggml_set_name(t, name.c_str());
        if (type == GGML_TYPE_I32) std::fill_n((int32_t *) t->data, k*m, -1);
        else if (type == GGML_TYPE_F16) std::fill_n((ggml_fp16_t *) t->data, k*m, ggml_fp32_to_fp16(invalid == "correction_zero" ? 0.f : invalid == "correction_nonfinite" ? std::numeric_limits<float>::infinity() : .25f));
        else std::fill_n((float *) t->data, k*m, name.find(".w1ax_midpoint") != std::string::npos ? (invalid == "affine_nonfinite" ? std::numeric_limits<float>::infinity() : invalid == "affine_zero" ? 0.f : .25f) : name == "fc.correction_bias" ? (invalid == "correction_bias_bound" ? .5f : .125f) : name == "fc.w1a1_scale" && invalid == "affine_zeroalpha" ? 0.f : name.find("scale") != std::string::npos ? .125f : 1.f);
        gguf_add_tensor(g, t);
    };
    for (int i = 0; i < 9; ++i) {
        std::string key = names[i]; std::replace(key.begin(), key.end(), '.', '_');
        key = "eagle3.w1a1.tensor."+key;
        const std::string packed = std::string(bases[i])+".w1a1_packed", scale = std::string(bases[i])+".w1a1_scale";
        gguf_set_val_u32(g, (key+".logical_k").c_str(), widths[i]);
        gguf_set_val_str(g, (key+".packed").c_str(), packed.c_str()); gguf_set_val_str(g, (key+".scale").c_str(), scale.c_str());
        tensor(packed, GGML_TYPE_I32, (widths[i]+31)/32, rows[i]); tensor(scale, GGML_TYPE_F32, rows[i]);
    }
    tensor("token_embd.weight", GGML_TYPE_F32, 64, 16);
    tensor("output_norm.weight", GGML_TYPE_F32, 64);
    tensor("blk.0.attn_norm.weight", GGML_TYPE_F32, 64);
    tensor("blk.0.attn_norm_2.weight", GGML_TYPE_F32, 64);
    tensor("blk.0.ffn_norm.weight", GGML_TYPE_F32, 64);
    if (invalid != "legacy" && invalid != "correction_fixed" && invalid.find("affine_fixed") != 0 && bits != 16) {
        const char * boundaries[] = {"fc", "qkv", "attn_output", "gate_up", "down", "head"};
        gguf_set_val_u32(g, "eagle3.w1a1.activation_quantizer.version", invalid == "version" ? 2 : 1);
        gguf_set_arr_str(g, "eagle3.w1a1.activation_quantizer.boundaries", boundaries, 6);
        for (auto boundary : boundaries) {
            const std::string prefix = std::string("eagle3.w1a1.activation_quantizer.")+boundary+".";
            gguf_set_val_f32(g, (prefix+"threshold_delta").c_str(), invalid == "nonfinite" ? std::numeric_limits<float>::infinity() : delta);
            if (!(invalid == "missing" && std::string(boundary) == "head")) gguf_set_val_f32(g, (prefix+"clip_ratio").c_str(), clip);
        }
        if (invalid == "extra") gguf_set_val_f32(g, "eagle3.w1a1.activation_quantizer.extra", 1);
    }
    if (invalid.find("correction_") == 0 || invalid == "affine_correction") {
        const uint32_t rank = invalid == "correction_rank4" ? 4 : invalid == "correction_rank_bad" ? 2 : 1;
        const bool bias = invalid == "correction_bias" || invalid == "correction_bias_bound";
        const std::string prefix = "eagle3.fusion_correction.";
        gguf_set_val_u32(g, (prefix+"version").c_str(), invalid == "correction_version" ? 2 : 1);
        gguf_set_val_u32(g, (prefix+"rank").c_str(), rank);
        gguf_set_val_str(g, (prefix+"u_name").c_str(), "fc.correction_u.weight");
        gguf_set_val_str(g, (prefix+"v_name").c_str(), "fc.correction_v.weight");
        gguf_set_val_str(g, (prefix+"bias_name").c_str(), bias ? "fc.correction_bias" : "");
        gguf_set_val_f32(g, (prefix+"bias_bound").c_str(), bias ? .25f : 0);
        gguf_set_val_str(g, (prefix+"arithmetic").c_str(), "raw_f32_v_f16_dot_f32_u_f16_dot_f32_add_base_f32_bias_f32");
        if (invalid == "correction_extra") gguf_set_val_u32(g, (prefix+"extra").c_str(), 1);
        if (invalid != "correction_missing") tensor("fc.correction_u.weight", GGML_TYPE_F16, rank, invalid == "correction_shape" ? 63 : 64);
        tensor("fc.correction_v.weight", GGML_TYPE_F16, 192, rank);
        if (bias) tensor("fc.correction_bias", GGML_TYPE_F32, 64);
    }
    if (invalid.find("affine_") == 0) {
        const bool all = invalid == "affine_all" || invalid == "affine_fixed_all";
        const std::string prefix = "eagle3.affine_weights.";
        gguf_set_val_u32(g,(prefix+"version").c_str(),invalid == "affine_version" ? 2 : 1);
        gguf_set_val_str(g,(prefix+"coverage").c_str(),all ? "all" : "fusion");
        gguf_set_val_str(g,(prefix+"arithmetic").c_str(),"integer_dot_alpha_beta_plus_integer_sum_midpoint_beta_before_bias_f32");
        std::vector<std::string> midpoint_names; std::vector<const char *> midpoint_ptrs;
        for (int i = 0; i < (all ? 9 : 1); ++i) midpoint_names.push_back(std::string(bases[i])+".w1ax_midpoint");
        for (const auto & name : midpoint_names) midpoint_ptrs.push_back(name.c_str());
        gguf_set_arr_str(g,(prefix+"bases").c_str(),bases,all ? 9 : 1);
        gguf_set_arr_str(g,(prefix+"midpoint_tensors").c_str(),midpoint_ptrs.data(),midpoint_ptrs.size());
        for (int i = 0; i < (all ? 9 : 1); ++i) if (invalid != "affine_missing") tensor(midpoint_names[i],GGML_TYPE_F32,invalid == "affine_shape" ? rows[i]-1 : rows[i]);
        if (invalid == "affine_extra") tensor("extra.w1ax_midpoint",GGML_TYPE_F32,64);
    }
    const std::string path = "/tmp/eagle3-learned-"+std::to_string(getpid())+".gguf";
    if (!gguf_write_to_file(g, path.c_str(), false)) return nullptr;
    ggml_free(ctx);
    return g;
}

static bool check_packs() {
    auto * backend = ggml_backend_cpu_init();
    for (int bits : {1,4,8,16}) for (bool affine : {false,true}) for (int variant : {0,1,2}) {
        if (bits == 16 && !affine) continue;
        const float delta = bits == 1 ? (variant == 0 ? 0.f : variant == 1 ? .75f : -1.f) : 0.f;
        const float clip = bits == 1 || bits == 16 || variant == 0 ? 1.f : .625f;
        ggml_init_params ip = {1024*1024, nullptr, true}; auto * ctx = ggml_init(ip);
        const int k = 33, n = 3;
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        auto * pack = affine ? ggml_w1ax_pack_affine(ctx,a,bits,delta,clip,true) : ggml_w1ax_pack_learned(ctx, a, bits, delta, clip);
        auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, pack);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> acts(k*n);
        for (int t = 0; t < n; ++t) for (int i = 0; i < k; ++i) {
            acts[t*k+i] = t == 0 ? (i%3 == 0 ? -2.f : i%3 == 1 ? .25f : 1.f) :
                t == 1 ? (i%2 ? -0.f : 0.f) : (i%2 ? -std::numeric_limits<float>::denorm_min() : std::numeric_limits<float>::denorm_min());
        }
        ggml_backend_tensor_set(a, acts.data(), 0, acts.size()*sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        const auto layout = affine ? ggml_w1ax_affine_layout(k,n,bits) : ggml_w1ax_pack_layout(k,n,bits);
        std::vector<uint32_t> actual(layout.total_words); ggml_backend_tensor_get(pack, actual.data(), 0, ggml_nbytes(pack));
        for (int t = 0; t < n; ++t) {
            double sum = 0; float maximum = 0;
            for (int i = 0; i < k; ++i) { sum += std::abs(double(acts[t*k+i])); maximum = std::max(maximum,std::abs(acts[t*k+i])); }
            const int qmax = bits == 8 ? 127 : 7; const float limit = maximum*clip;
            const float beta = bits == 1 ? float(sum/k) : bits == 16 ? 1.f : limit/qmax;
            float expected_sum = 0;
            float actual_beta; memcpy(&actual_beta, actual.data()+layout.scale_offset+t, sizeof(float));
            if (memcmp(&beta, &actual_beta, sizeof(float))) { fprintf(stderr,"pack beta mismatch\n"); return false; }
            for (int i = 0; i < k; ++i) {
                if (bits == 16) {
                    expected_sum += ggml_fp16_to_fp32(ggml_fp32_to_fp16(acts[t*k+i]));
                } else if (bits == 1) {
                    volatile float threshold = delta*beta;
                    volatile float shifted = delta == 0 ? acts[t*k+i] : acts[t*k+i]-threshold;
                    const bool expected = shifted >= 0;
                    expected_sum += expected ? 1 : -1;
                    const bool observed = (actual[t*((k+31)/32)+i/32]>>(i%32))&1u;
                    if (observed != expected) { fprintf(stderr,"packed sign mismatch t=%d i=%d\n",t,i); return false; }
                } else {
                    const float inv = limit == 0 ? 0 : float(qmax)/limit;
                    const float normalized = limit == 0 ? 0 : std::isfinite(inv) ? acts[t*k+i]*inv : float(double(acts[t*k+i])/double(limit)*qmax);
                    const int expected = std::max(-qmax,std::min(qmax,int(std::nearbyint(normalized))));
                    expected_sum += expected;
                    if (((int8_t *) actual.data())[t*k+i] != expected) { fprintf(stderr,"packed code mismatch t=%d i=%d\n",t,i); return false; }
                    if (bits == 4) for (int bit = 0; bit < 4; ++bit) {
                        const uint32_t plane = actual[layout.codes_words+(t*((k+31)/32)+i/32)*4+bit];
                        if (((plane>>(i%32))&1u) != (((uint8_t) expected>>bit)&1u)) { fprintf(stderr,"packed plane mismatch\n"); return false; }
                    }
                }
            }
            if (affine) {
                float actual_sum; memcpy(&actual_sum,actual.data()+layout.scale_offset+n+t,sizeof(float));
                if (memcmp(&actual_sum,&expected_sum,sizeof(float))) { fprintf(stderr,"affine shared sum mismatch bits=%d token=%d\n",bits,t); return false; }
            }
            if (bits == 1 && (actual[t*((k+31)/32)+(k+31)/32-1] & ~1u)) { fprintf(stderr,"packed tail mismatch\n"); return false; }
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend); return true;
}

int main() {
    llama_backend_init();
    if (!check_packs()) return 1;
    llama_log_set([](ggml_log_level level, const char * text, void *){ if (level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }, nullptr);
    for (int bits : {1,4,8,16}) {
        const std::string value = std::to_string(bits); setenv("GGML_W1AX_ACT_BITS", value.c_str(), 1);
        for (const std::string mode : {"legacy", "default", "learned", "version", "nonfinite", "missing", "extra", "incompatible", "correction_fixed", "correction_zero", "correction_rank1", "correction_rank4", "correction_bias", "correction_version", "correction_rank_bad", "correction_missing", "correction_shape", "correction_extra", "correction_nonfinite", "correction_bias_bound", "affine_fixed_fusion", "affine_fixed_all", "affine_fusion", "affine_all", "affine_correction", "affine_zero", "affine_zeroalpha", "affine_version", "affine_shape", "affine_missing", "affine_extra", "affine_nonfinite"}) {
            if (bits == 16 && mode.find("affine_") != 0) continue;
            const float delta = mode == "incompatible" ? .5f : bits == 1 && (mode == "learned" || mode == "affine_all" || mode == "affine_correction") ? .75f : 0;
            const float clip = (bits == 4 || bits == 8) && (mode == "learned" || mode == "affine_all" || mode == "affine_correction") ? .625f : 1;
            auto * g = fixture(bits, delta, clip, mode);
            llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
            const std::string path = "/tmp/eagle3-learned-"+std::to_string(getpid())+".gguf";
            auto * model = llama_model_load_from_file(path.c_str(), mp); gguf_free(g); std::remove(path.c_str());
            const bool valid = mode == "affine_fixed_fusion" || mode == "affine_fixed_all" || mode == "affine_fusion" || mode == "affine_all" || mode == "affine_correction" || mode == "affine_zero" || mode == "affine_zeroalpha" || mode == "correction_fixed" || mode == "correction_zero" || mode == "correction_rank1" || mode == "correction_rank4" || mode == "correction_bias" || mode == "legacy" || mode == "default" || mode == "learned" || (mode == "incompatible" && bits == 1);
            if ((model != nullptr) != valid) { fprintf(stderr, "loader mismatch bits=%d mode=%s\n", bits, mode.c_str()); return 1; }
            if (!model) continue;
            llama_context_params cp = llama_context_default_params(); cp.n_ctx = 32; cp.n_threads = 2; cp.n_threads_batch = 2; cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
            auto * ctx = llama_init_from_model(model, cp);
            if (!ctx) { fprintf(stderr, "context failure\n"); return 1; }
            auto batch = llama_batch_init(1, 192, 1); batch.n_tokens = 1; batch.pos[0] = 0; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
            double abs_sum = 0; float absmax = 0;
            for (int i = 0; i < 192; ++i) { batch.embd[i] = i%3 == 0 ? -2.f : i%3 == 1 ? .25012345f : 1.0002345f; abs_sum += std::abs(batch.embd[i]); absmax = std::max(absmax, std::abs(batch.embd[i])); }
            const float beta = float(abs_sum/192), limit = absmax*clip, qmax = bits == 8 ? 127 : 7; int dot = 0;
            for (int i = 0; i < 192; ++i) dot += bits == 1 ? (batch.embd[i] - delta*beta >= 0 ? 1 : -1) : std::max(-int(qmax), std::min(int(qmax), int(std::nearbyint(batch.embd[i]*(qmax/limit)))));
            float half_sum = 0;
            for (int i = 0; i < 192; ++i) half_sum += ggml_fp16_to_fp32(ggml_fp32_to_fp16(batch.embd[i]));
            const float code_sum = bits == 16 ? half_sum : float(dot);
            const float scale = bits == 1 ? beta : bits == 16 ? 1.f : limit/qmax;
            float ref = code_sum*(mode == "affine_zeroalpha" ? 0.f : .125f)*scale;
            if (mode.find("affine_") == 0) {
                volatile float midpoint_term = code_sum*(mode == "affine_zero" ? 0.f : .25f);
                volatile float scaled = midpoint_term*scale;
                ref += scaled;
            }
            if ((mode.find("correction_") == 0 || mode == "affine_correction") && mode != "correction_zero") {
                float latent = 0; for (int i = 0; i < 192; ++i) latent += batch.embd[i]*.25f;
                const int rank = mode == "correction_rank4" ? 4 : 1;
                float correction = 0; for (int i = 0; i < rank; ++i) correction += latent*.25f;
                ref += correction; if (mode == "correction_bias") ref += .125f;
            }
            if (llama_encode(ctx, batch)) { fprintf(stderr, "encode failure\n"); return 1; }
            auto * output = llama_get_embeddings(ctx);
            if (!output) { fprintf(stderr, "missing output\n"); return 1; }
            for (int i = 0; i < 64; ++i) if (std::abs(output[i]-ref) > 1e-4) { fprintf(stderr, "numeric mismatch %g vs %g\n", output[i], ref); return 1; }
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        }
    }
    puts("EAGLE3 learned/correction/affine loader/encoder fixtures passed"); llama_backend_free();
}
