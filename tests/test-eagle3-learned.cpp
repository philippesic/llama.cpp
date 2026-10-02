// Tiny selectable CPU/CUDA loader/encoder fixtures for the versioned learned scalar contract.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"
#include "gguf.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-context.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include "../vendor/nlohmann/json.hpp"
#include "build-info.h"

using json = nlohmann::ordered_json;
static json measurements = {{"schema_version",1},{"input_scope","synthetic_operator"},{"status","failed"},{"failure_reason","run did not complete"},{"pack_cases",json::array()},{"loader_cases",json::array()},{"encoder_cases",json::array()},{"projection_cases",json::array()}};
static FILE * report_stream = nullptr;
static void checkpoint_report() {
    if (!report_stream) return;
    const std::string data = measurements.dump(2)+"\n";
    rewind(report_stream);
    if (fwrite(data.data(),1,data.size(),report_stream) != data.size() || fflush(report_stream) || ftruncate(fileno(report_stream),data.size())) {
        fprintf(stderr,"failed to save JSON measurement report\n"); std::exit(3);
    }
}
struct report_guard { ~report_guard() { checkpoint_report(); if (report_stream) fclose(report_stream); } };
static std::string hex_bytes(const void * data,size_t size) {
    const auto * bytes = (const unsigned char *) data; const char * digits = "0123456789abcdef";
    std::string result(size*2,'0');
    for (size_t i = 0; i < size; ++i) { result[2*i] = digits[bytes[i]>>4]; result[2*i+1] = digits[bytes[i]&15]; }
    return result;
}
static std::string tensor_hex(ggml_tensor * t) {
    std::vector<unsigned char> bytes(ggml_nbytes(t)); ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    return hex_bytes(bytes.data(),bytes.size());
}
static double relative_rms(const std::vector<float> & expected,const std::vector<float> & actual) {
    double squared = 0, reference = 0;
    for (size_t i = 0; i < expected.size(); ++i) { squared += std::pow(double(actual[i])-expected[i],2); reference += double(expected[i])*expected[i]; }
    return std::sqrt(squared/std::max(reference,1e-30));
}

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
        else if (type == GGML_TYPE_F16) std::fill_n((ggml_fp16_t *) t->data, k*m, ggml_fp32_to_fp16(((invalid == "correction_zero" || invalid == "correction_zero_rank4") && name == "fc.correction_u.weight") ? 0.f : invalid == "correction_nonfinite" ? std::numeric_limits<float>::infinity() : .25f));
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
        const uint32_t rank = (invalid == "correction_rank4" || invalid == "correction_rank4_bias" || invalid == "correction_zero_rank4") ? 4 : invalid == "correction_rank_bad" ? 2 : 1;
        const bool bias = invalid == "correction_bias" || invalid == "correction_rank4_bias" || invalid == "correction_bias_bound";
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

struct fixture_backend {
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    const char * requested = "CPU";
};

struct execution_audit {
    llama_context * ctx = nullptr;
    ggml_backend_dev_t device = nullptr;
    int arithmetic_nodes = 0;
    bool correct = true;
    json nodes = json::array();
};

static bool audit_node(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & audit = *(execution_audit *) user_data;
    const bool arithmetic = tensor->op == GGML_OP_W1AX_PACK || tensor->op == GGML_OP_W1A1_MUL_MAT ||
        tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_ADD;
    if (ask) return arithmetic;
    auto * backend = ggml_backend_sched_get_tensor_backend(audit.ctx->get_sched(), tensor);
    if (!backend || ggml_backend_get_device(backend) != audit.device) {
        fprintf(stderr,"requested device fallback at %s (%s)\n",tensor->name,ggml_op_name(tensor->op));
        audit.correct = false;
        return false;
    }
    json node = {{"name",tensor->name},{"op",ggml_op_name(tensor->op)},{"device",ggml_backend_dev_name(ggml_backend_get_device(backend))},{"output_f32_hex",tensor->type == GGML_TYPE_F32 ? tensor_hex(tensor) : ""},{"sources",json::array()}};
    for (int i = 0; i < GGML_MAX_SRC; ++i) if (tensor->src[i]) {
        auto * source = tensor->src[i];
        node["sources"].push_back({{"index",i},{"name",source->name},{"type",ggml_type_name(source->type)},{"shape",{source->ne[0],source->ne[1],source->ne[2],source->ne[3]}},{"raw_hex",tensor_hex(source)}});
    }
    audit.nodes.push_back(std::move(node));
    ++audit.arithmetic_nodes;
    return true;
}

static bool check_packs(const fixture_backend & selected, int & cases) {
    auto * backend = selected.backend;
    for (int bits : {1,4,8,16}) for (bool affine : {false,true}) for (bool learned : {false,true}) for (int variant : {0,1,2}) {
        if (!affine && !learned) continue;
        if (bits == 16 && !affine) continue;
        const float delta = bits == 1 ? (variant == 0 ? 0.f : variant == 1 ? .75f : -1.f) : 0.f;
        const float clip = bits == 1 || bits == 16 || variant == 0 ? 1.f : .625f;
        ggml_init_params ip = {1024*1024, nullptr, true}; auto * ctx = ggml_init(ip);
        const int k = 33, n = 4;
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        auto * pack = affine ? ggml_w1ax_pack_affine(ctx,a,bits,delta,clip,learned) : ggml_w1ax_pack_learned(ctx, a, bits, delta, clip);
        auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, pack);
        if (!ggml_backend_supports_op(backend,pack)) { fprintf(stderr,"requested backend does not support pack\n"); return false; }
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer || (std::string(selected.requested) == "CUDA" && ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) != selected.device)) { fprintf(stderr,"pack buffer fallback\n"); return false; }
        std::vector<float> acts(k*n);
        for (int t = 0; t < n; ++t) for (int i = 0; i < k; ++i) {
            acts[t*k+i] = t == 0 ? (i%3 == 0 ? -2.f : i%3 == 1 ? .25f : 1.f) :
                t == 1 ? (i%2 ? -0.f : 0.f) : t == 2 ? (i%2 ? -std::numeric_limits<float>::denorm_min() : std::numeric_limits<float>::denorm_min()) : (i%3 == 0 ? 0.f : i%3 == 1 ? -std::numeric_limits<float>::denorm_min() : std::numeric_limits<float>::denorm_min());
        }
        ggml_backend_tensor_set(a, acts.data(), 0, acts.size()*sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        const auto layout = affine ? ggml_w1ax_affine_layout(k,n,bits) : ggml_w1ax_pack_layout(k,n,bits);
        std::vector<uint32_t> actual(layout.total_words); ggml_backend_tensor_get(pack, actual.data(), 0, ggml_nbytes(pack));
        std::vector<uint32_t> expected_packed(layout.total_words,0);
        for (int t = 0; t < n; ++t) {
            double sum = 0; float maximum = 0;
            for (int i = 0; i < k; ++i) { sum += std::abs(double(acts[t*k+i])); maximum = std::max(maximum,std::abs(acts[t*k+i])); }
            const int qmax = bits == 8 ? 127 : 7; const float limit = maximum*clip;
            const float beta = bits == 1 ? float(sum/k) : bits == 16 ? 1.f : limit/qmax;
            float expected_sum = 0;
            memcpy(expected_packed.data()+layout.scale_offset+t,&beta,sizeof(float));
            float actual_beta; memcpy(&actual_beta, actual.data()+layout.scale_offset+t, sizeof(float));
            if (memcmp(&beta, &actual_beta, sizeof(float))) {
                uint32_t expected_bits,actual_bits;
                memcpy(&expected_bits,&beta,sizeof(expected_bits)); memcpy(&actual_bits,&actual_beta,sizeof(actual_bits));
                fprintf(stderr,"pack beta mismatch bits=%d affine=%d learned=%d variant=%d delta=%a clip=%a k=%d n=%d token=%d expected_beta=%a actual_beta=%a expected_bits=0x%08x actual_bits=0x%08x sum_f64=%a absmax=%a limit=%a\n",
                        bits,int(affine),int(learned),variant,double(delta),double(clip),k,n,t,double(beta),double(actual_beta),unsigned(expected_bits),unsigned(actual_bits),sum,double(maximum),double(limit));
                measurements["failure_reason"] = "pack_beta_mismatch";
                measurements["failed_pack_case"] = {{"bits",bits},{"affine",affine},{"learned",learned},{"variant",variant},{"threshold_delta",delta},{"clip_ratio",clip},{"k",k},{"n",n},{"token",t},
                    {"expected_beta_f32_hex",hex_bytes(&beta,sizeof(float))},{"native_beta_f32_hex",hex_bytes(&actual_beta,sizeof(float))},{"expected_beta_u32",expected_bits},{"native_beta_u32",actual_bits},{"reference_sum_f64",sum},{"reference_absmax_f32",maximum},{"reference_limit_f32",limit},
                    {"input_f32_hex",hex_bytes(acts.data(),acts.size()*sizeof(float))},{"native_packed_hex",hex_bytes(actual.data(),actual.size()*sizeof(uint32_t))},
                    {"layout",{{"codes_words",layout.codes_words},{"planes_words",layout.planes_words},{"scale_offset",layout.scale_offset},{"sum_offset",affine ? json(layout.scale_offset+n) : json(nullptr)},{"total_words",layout.total_words}}}};
                checkpoint_report();
                return false;
            }
            for (int i = 0; i < k; ++i) {
                if (bits == 16) {
                    expected_sum += ggml_fp16_to_fp32(ggml_fp32_to_fp16(acts[t*k+i]));
                } else if (bits == 1) {
                    volatile float threshold = delta*beta;
                    volatile float shifted = delta == 0 ? acts[t*k+i] : acts[t*k+i]-threshold;
                    const bool expected = shifted >= 0;
                    expected_sum += expected ? 1 : -1;
                    if (expected) expected_packed[t*((k+31)/32)+i/32] |= uint32_t(1)<<(i%32);
                    const bool observed = (actual[t*((k+31)/32)+i/32]>>(i%32))&1u;
                    if (observed != expected) { fprintf(stderr,"packed sign mismatch t=%d i=%d\n",t,i); return false; }
                } else {
                    const float inv = limit == 0 ? 0 : float(qmax)/limit;
                    const float normalized = limit == 0 ? 0 : std::isfinite(inv) ? acts[t*k+i]*inv : float(double(acts[t*k+i])/double(limit)*qmax);
                    const int expected = std::max(-qmax,std::min(qmax,int(std::nearbyint(normalized))));
                    expected_sum += expected;
                    ((int8_t *) expected_packed.data())[t*k+i] = (int8_t) expected;
                    if (bits == 4) for (int bit = 0; bit < 4; ++bit) expected_packed[layout.codes_words+(t*((k+31)/32)+i/32)*4+bit] |= (((uint8_t) expected>>bit)&1u)<<(i%32);
                    if (((int8_t *) actual.data())[t*k+i] != expected) { fprintf(stderr,"packed code mismatch t=%d i=%d\n",t,i); return false; }
                    if (bits == 4) for (int bit = 0; bit < 4; ++bit) {
                        const uint32_t plane = actual[layout.codes_words+(t*((k+31)/32)+i/32)*4+bit];
                        if (((plane>>(i%32))&1u) != (((uint8_t) expected>>bit)&1u)) { fprintf(stderr,"packed plane mismatch\n"); return false; }
                    }
                }
            }
            if (affine) {
                memcpy(expected_packed.data()+layout.scale_offset+n+t,&expected_sum,sizeof(float));
                float actual_sum; memcpy(&actual_sum,actual.data()+layout.scale_offset+n+t,sizeof(float));
                if (memcmp(&actual_sum,&expected_sum,sizeof(float))) { fprintf(stderr,"affine shared sum mismatch bits=%d token=%d\n",bits,t); return false; }
            }
            if (bits == 1 && (actual[t*((k+31)/32)+(k+31)/32-1] & ~1u)) { fprintf(stderr,"packed tail mismatch\n"); return false; }
        }
        measurements["pack_cases"].push_back({{"kind","scalar_pack"},{"bits",bits},{"affine",affine},{"learned",learned},{"threshold_delta",delta},{"clip_ratio",clip},{"k",k},{"n",n},{"input_f32_hex",hex_bytes(acts.data(),acts.size()*sizeof(float))},{"expected_packed_hex",hex_bytes(expected_packed.data(),expected_packed.size()*4)},{"native_packed_hex",hex_bytes(actual.data(),actual.size()*4)},{"layout",{{"codes_words",layout.codes_words},{"planes_words",layout.planes_words},{"scale_offset",layout.scale_offset},{"sum_offset",affine ? json(layout.scale_offset+n) : json(nullptr)},{"total_words",layout.total_words}}}});
        checkpoint_report();
        if (expected_packed != actual) { fprintf(stderr,"whole packed bytes mismatch\n"); return false; }
        ++cases;
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    return true;
}

static bool check_projections(const fixture_backend & selected,int & pack_cases) {
    const char * boundaries[] = {"fc","qkv","attn_output","gate_up","down","head"};
    const int widths[] = {193,131,65,65,131,65}; // Odd synthetic widths exercise tails.
    const char * bases[] = {"fc","blk.0.attn_q","blk.0.attn_k","blk.0.attn_v","blk.0.attn_output","blk.0.ffn_gate","blk.0.ffn_up","blk.0.ffn_down","output"};
    const int boundary_for[] = {0,1,1,1,2,3,3,4,5};
    const float alpha[] = {0,.125f,.25f}, mu[] = {.25f,0,-.125f}, zero_mu[] = {0,0,0};
    for (int bits : {1,4,8,16}) {
        ggml_init_params ip = {2*1024*1024,nullptr,true}; auto * ctx = ggml_init(ip);
        auto * graph = ggml_new_graph(ctx);
        ggml_tensor * inputs[6], * packs[6], * cast_codes[6] = {};
        ggml_tensor * outputs[9], * zero_outputs[9], * baseline_outputs[9], * weights[9], * scales[9], * means[9], * zeros[9];
        std::vector<float> acts[6]; std::vector<uint32_t> packed_weights[9];
        const bool learned = bits != 16; const float delta = bits == 1 ? .75f : 0, clip = bits == 4 || bits == 8 ? .625f : 1;
        for (int b = 0; b < 6; ++b) {
            const int k = widths[b]; acts[b].resize(k);
            for (int i = 0; i < k; ++i) acts[b][i] = i%3 == 0 ? -2.f : i%3 == 1 ? .25012345f : 1.0002345f;
            inputs[b] = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,1);
            packs[b] = ggml_w1ax_pack_affine(ctx,inputs[b],bits,delta,clip,learned);
            if (bits == 16) { cast_codes[b] = ggml_cast(ctx,ggml_cast(ctx,inputs[b],GGML_TYPE_F16),GGML_TYPE_F32); ggml_build_forward_expand(graph,cast_codes[b]); }
        }
        for (int p = 0; p < 9; ++p) {
            const int b = boundary_for[p], k = widths[b], words = (k+31)/32;
            weights[p] = ggml_new_tensor_2d(ctx,GGML_TYPE_I32,words,3); scales[p] = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3);
            means[p] = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3); zeros[p] = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3);
            packed_weights[p].resize(words*3,0);
            for (int row = 0; row < 3; ++row) for (int i = 0; i < k; ++i) if (row == 0 || (row == 1 && i%2 == 0)) packed_weights[p][row*words+i/32] |= uint32_t(1)<<(i%32);
            outputs[p] = ggml_w1ax_mul_mat_affine(ctx,weights[p],scales[p],means[p],packs[b],k,bits);
            zero_outputs[p] = ggml_w1ax_mul_mat_affine(ctx,weights[p],scales[p],zeros[p],packs[b],k,bits);
            baseline_outputs[p] = ggml_w1ax_mul_mat_shared(ctx,weights[p],scales[p],packs[b],k,bits);
            ggml_build_forward_expand(graph,outputs[p]); ggml_build_forward_expand(graph,zero_outputs[p]); ggml_build_forward_expand(graph,baseline_outputs[p]);
        }
        for (int i = 0; i < graph->n_nodes; ++i) if (!ggml_backend_supports_op(selected.backend,graph->nodes[i])) { fprintf(stderr,"projection backend unsupported operation\n"); return false; }
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,selected.backend);
        if (!buffer) return false;
        for (int b = 0; b < 6; ++b) ggml_backend_tensor_set(inputs[b],acts[b].data(),0,acts[b].size()*4);
        for (int p = 0; p < 9; ++p) {
            ggml_backend_tensor_set(weights[p],packed_weights[p].data(),0,packed_weights[p].size()*4); ggml_backend_tensor_set(scales[p],alpha,0,sizeof(alpha));
            ggml_backend_tensor_set(means[p],mu,0,sizeof(mu)); ggml_backend_tensor_set(zeros[p],zero_mu,0,sizeof(zero_mu));
        }
        if (ggml_backend_graph_compute(selected.backend,graph) != GGML_STATUS_SUCCESS) return false;
        std::vector<float> expected_codes[6],native_codes[6]; float beta[6],sum[6]; size_t pack_index[6];
        for (int b = 0; b < 6; ++b) {
            const int k = widths[b],qmax = bits == 8 ? 127 : 7;
            double meanabs = 0; float maximum = 0;
            for (float x : acts[b]) { meanabs += std::abs(double(x)); maximum = std::max(maximum,std::abs(x)); }
            const float limit = maximum*clip; beta[b] = bits == 1 ? float(meanabs/k) : bits == 16 ? 1.f : limit/qmax; sum[b] = 0;
            const auto layout = ggml_w1ax_affine_layout(k,1,bits); std::vector<uint32_t> expected(layout.total_words,0),actual(layout.total_words);
            ggml_backend_tensor_get(packs[b],actual.data(),0,actual.size()*4); expected_codes[b].resize(k); native_codes[b].resize(k);
            if (bits == 16) ggml_backend_tensor_get(cast_codes[b],native_codes[b].data(),0,k*4);
            for (int i = 0; i < k; ++i) {
                volatile float threshold = delta*beta[b]; volatile float shifted = delta == 0 ? acts[b][i] : acts[b][i]-threshold;
                const float code = bits == 16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(acts[b][i])) : bits == 1 ? (shifted >= 0 ? 1.f : -1.f) : std::max(-float(qmax),std::min(float(qmax),std::nearbyint(acts[b][i]*(float(qmax)/limit))));
                expected_codes[b][i] = code; sum[b] += code;
                if (bits == 1) { if (code > 0) expected[i/32] |= uint32_t(1)<<(i%32); native_codes[b][i] = (actual[i/32]>>(i%32))&1u ? 1.f : -1.f; }
                else if (bits != 16) {
                    ((int8_t *) expected.data())[i] = (int8_t) code; native_codes[b][i] = ((int8_t *) actual.data())[i];
                    if (bits == 4) for (int bit = 0; bit < 4; ++bit) expected[layout.codes_words+(i/32)*4+bit] |= (((uint8_t) (int) code>>bit)&1u)<<(i%32);
                }
            }
            memcpy(expected.data()+layout.scale_offset,&beta[b],4); memcpy(expected.data()+layout.scale_offset+1,&sum[b],4);
            pack_index[b] = measurements["pack_cases"].size();
            measurements["pack_cases"].push_back({{"kind","projection_pack"},{"boundary",boundaries[b]},{"bits",bits},{"affine",true},{"learned",learned},{"threshold_delta",delta},{"clip_ratio",clip},{"k",k},{"n",1},{"input_f32_hex",hex_bytes(acts[b].data(),k*4)},{"expected_packed_hex",hex_bytes(expected.data(),expected.size()*4)},{"native_packed_hex",hex_bytes(actual.data(),actual.size()*4)},{"layout",{{"codes_words",layout.codes_words},{"planes_words",layout.planes_words},{"scale_offset",layout.scale_offset},{"sum_offset",layout.scale_offset+1},{"total_words",layout.total_words}}}});
            checkpoint_report();
            if (expected != actual || expected_codes[b] != native_codes[b]) { fprintf(stderr,"projection packed bytes/codes mismatch\n"); return false; }
            ++pack_cases;
        }
        for (int p = 0; p < 9; ++p) {
            const int b = boundary_for[p],k = widths[b],words = (k+31)/32;
            std::vector<float> expected(3),expected_baseline(3),actual(3),actual_zero(3),actual_baseline(3);
            for (int row = 0; row < 3; ++row) {
                float dot = 0; for (int i = 0; i < k; ++i) dot += ((packed_weights[p][row*words+i/32]>>(i%32))&1u ? 1 : -1)*expected_codes[b][i];
                volatile float weighted = dot*alpha[row]; volatile float base = weighted*beta[b]; volatile float midpoint = sum[b]*mu[row]; volatile float correction = midpoint*beta[b];
                expected_baseline[row] = base; expected[row] = base+correction;
            }
            ggml_backend_tensor_get(outputs[p],actual.data(),0,12); ggml_backend_tensor_get(zero_outputs[p],actual_zero.data(),0,12); ggml_backend_tensor_get(baseline_outputs[p],actual_baseline.data(),0,12);
            const auto layout = ggml_w1ax_affine_layout(k,1,bits); std::vector<uint32_t> data(layout.total_words); ggml_backend_tensor_get(packs[b],data.data(),0,data.size()*4);
            measurements["projection_cases"].push_back({{"base",bases[p]},{"boundary",boundaries[b]},{"bits",bits},{"learned",learned},{"threshold_delta",delta},{"clip_ratio",clip},{"k",k},{"n",1},{"pack_case_index",pack_index[b]},{"native_codes_source",bits == 16 ? "explicit_backend_f16_boundary_cast" : "packed_sign_or_integer_codes"},{"arithmetic_exception",bits == 16 ? "a16_f16_values_f32_sum_beta_one" : "none"},{"input_f32_hex",hex_bytes(acts[b].data(),k*4)},{"alpha_f32_hex",hex_bytes(alpha,sizeof(alpha))},{"midpoint_f32_hex",hex_bytes(mu,sizeof(mu))},{"weights_i32_hex",hex_bytes(packed_weights[p].data(),packed_weights[p].size()*4)},{"expected_codes_f32_hex",hex_bytes(expected_codes[b].data(),k*4)},{"native_codes_f32_hex",hex_bytes(native_codes[b].data(),k*4)},{"expected_scale_f32_hex",hex_bytes(&beta[b],4)},{"native_scale_f32_hex",hex_bytes(data.data()+layout.scale_offset,4)},{"expected_sum_f32_hex",hex_bytes(&sum[b],4)},{"native_sum_f32_hex",hex_bytes(data.data()+layout.scale_offset+1,4)},{"expected_output_f32_hex",hex_bytes(expected.data(),12)},{"native_output_f32_hex",hex_bytes(actual.data(),12)},{"expected_baseline_output_f32_hex",hex_bytes(expected_baseline.data(),12)},{"native_baseline_output_f32_hex",hex_bytes(actual_baseline.data(),12)},{"native_zero_midpoint_output_f32_hex",hex_bytes(actual_zero.data(),12)},{"relative_rms",relative_rms(expected,actual)}});
            checkpoint_report();
            if (memcmp(actual_zero.data(),actual_baseline.data(),12)) { fprintf(stderr,"zero midpoint identity mismatch\n"); return false; }
            for (int row = 0; row < 3; ++row) if (!std::isfinite(actual[row]) || std::abs(actual[row]-expected[row]) > 1e-4) { fprintf(stderr,"projection output mismatch\n"); return false; }
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    return true;
}

int main(int argc, char ** argv) {
    fixture_backend selected;
    const char * report_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--backend" && i+1 < argc && (std::string(argv[i+1]) == "CPU" || std::string(argv[i+1]) == "CUDA")) selected.requested = argv[++i];
        else if (std::string(argv[i]) == "--json-report" && i+1 < argc && !report_path) report_path = argv[++i];
        else { fprintf(stderr,"Usage: %s [--backend CPU|CUDA] [--json-report <newpath>]\n",argv[0]); return 2; }
    }
    if (report_path) {
        const int fd = open(report_path,O_CREAT|O_EXCL|O_WRONLY,0600);
        if (fd < 0) { fprintf(stderr,"JSON report must be a new writable path\n"); return 2; }
        report_stream = fdopen(fd,"w");
        if (!report_stream) { close(fd); return 2; }
    }
    report_guard guard;
    const uint16_t endian_probe = 1;
    if (*(const unsigned char *) &endian_probe != 1) { fprintf(stderr,"JSON fixture requires little-endian host\n"); return 2; }
    measurements["byte_order"] = "little";
    measurements["requested_backend"] = selected.requested;
    measurements["command"] = json::array(); for (int i = 0; i < argc; ++i) measurements["command"].push_back(argv[i]);
    measurements["runtime"] = {{"version",llama_version()},{"build_commit",llama_commit()},{"compiler",llama_compiler()},{"build_target",llama_build_target()}};
    checkpoint_report();
    const bool cuda = std::string(selected.requested) == "CUDA";
#ifndef GGML_USE_CUDA
    if (cuda) { fprintf(stderr,"CUDA requested but this fixture was built without CUDA support; refusing fallback\n"); return 2; }
#endif
    llama_backend_init();
    if (cuda) {
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) { fprintf(stderr,"CUDA requested but no CUDA device is available; refusing fallback\n"); return 2; }
        selected.device = ggml_backend_reg_dev_get(reg,0);
        if (!selected.device || ggml_backend_dev_type(selected.device) != GGML_BACKEND_DEVICE_TYPE_GPU || std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(selected.device))) != "CUDA") return 2;
        selected.backend = ggml_backend_dev_init(selected.device,nullptr);
    } else {
        selected.backend = ggml_backend_cpu_init();
        selected.device = ggml_backend_get_device(selected.backend);
    }
    if (!selected.backend) { fprintf(stderr,"failed to initialize requested backend\n"); return 2; }
    measurements["backend"] = {{"registration",ggml_backend_reg_name(ggml_backend_dev_backend_reg(selected.device))},{"device",ggml_backend_dev_name(selected.device)},{"hardware",ggml_backend_dev_description(selected.device)}};
    checkpoint_report();
    printf("fixture backend=%s device=%s hardware=%s\n",selected.requested,ggml_backend_dev_name(selected.device),ggml_backend_dev_description(selected.device));
    int pack_cases = 0, loader_cases = 0, graph_cases = 0, graph_nodes = 0;
    if (!check_packs(selected,pack_cases) || !check_projections(selected,pack_cases)) return 1;
    llama_log_set([](ggml_log_level level, const char * text, void *){ if (level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }, nullptr);
    for (int bits : {1,4,8,16}) {
        const std::string value = std::to_string(bits); setenv("GGML_W1AX_ACT_BITS", value.c_str(), 1);
        for (const std::string mode : {"legacy", "default", "learned", "version", "nonfinite", "missing", "extra", "incompatible", "correction_fixed", "correction_zero", "correction_rank1", "correction_rank4", "correction_rank4_bias", "correction_zero_rank4", "correction_bias", "correction_version", "correction_rank_bad", "correction_missing", "correction_shape", "correction_extra", "correction_nonfinite", "correction_bias_bound", "affine_fixed_fusion", "affine_fixed_all", "affine_fusion", "affine_all", "affine_correction", "affine_zero", "affine_zeroalpha", "affine_version", "affine_shape", "affine_missing", "affine_extra", "affine_nonfinite"}) {
            if (bits == 16 && mode.find("affine_") != 0) continue;
            const float delta = mode == "incompatible" ? .5f : bits == 1 && (mode == "learned" || mode == "affine_all" || mode == "affine_correction") ? .75f : 0;
            const float clip = (bits == 4 || bits == 8) && (mode == "learned" || mode == "affine_all" || mode == "affine_correction") ? .625f : 1;
            auto * g = fixture(bits, delta, clip, mode);
            const int rank_key = gguf_find_key(g,"eagle3.fusion_correction.rank");
            const int correction_rank = rank_key >= 0 ? gguf_get_val_u32(g,rank_key) : 0;
            const int bias_key = gguf_find_key(g,"eagle3.fusion_correction.bias_name");
            const bool correction_bias = bias_key >= 0 && strlen(gguf_get_val_str(g,bias_key)) > 0;
            const bool valid = mode == "affine_fixed_fusion" || mode == "affine_fixed_all" || mode == "affine_fusion" || mode == "affine_all" || mode == "affine_correction" || mode == "affine_zero" || mode == "affine_zeroalpha" || mode == "correction_fixed" || mode == "correction_zero" || mode == "correction_rank1" || mode == "correction_rank4" || mode == "correction_rank4_bias" || mode == "correction_zero_rank4" || mode == "correction_bias" || mode == "legacy" || mode == "default" || mode == "learned" || (mode == "incompatible" && bits == 1);
            llama_model_params mp = llama_model_default_params();
            ggml_backend_dev_t devices[] = {selected.device,nullptr};
            mp.n_gpu_layers = cuda && valid ? 1000 : 0;
            // Invalid payloads remain CPU loader tests; valid graph devices are pinned.
            ggml_backend_dev_t cpu_only[] = {nullptr};
            mp.devices = cuda && valid ? devices : cpu_only;
            const std::string path = "/tmp/eagle3-learned-"+std::to_string(getpid())+".gguf";
            auto * model = llama_model_load_from_file(path.c_str(), mp); gguf_free(g); std::remove(path.c_str());

            measurements["loader_cases"].push_back({{"bits",bits},{"mode",mode},{"expected_valid",valid},{"native_loaded",model != nullptr},{"execution_backend",cuda && valid ? "CUDA" : "CPU"}});
            checkpoint_report();
            ++loader_cases;
            if ((model != nullptr) != valid) { fprintf(stderr, "loader mismatch bits=%d mode=%s\n", bits, mode.c_str()); return 1; }
            if (!model) continue;
            llama_context_params cp = llama_context_default_params(); cp.n_ctx = 32; cp.n_threads = 2; cp.n_threads_batch = 2; cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
            execution_audit audit; audit.device = selected.device;
            cp.cb_eval = audit_node; cp.cb_eval_user_data = &audit;
            auto * ctx = llama_init_from_model(model, cp);
            audit.ctx = ctx;
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
            const float base_expected = ref;
            float correction_expected = 0, bias_expected = 0;
            if ((mode.find("correction_") == 0 || mode == "affine_correction") && mode != "correction_zero" && mode != "correction_zero_rank4") {
                float latent = 0; for (int i = 0; i < 192; ++i) latent += batch.embd[i]*.25f;
                const int rank = correction_rank;
                float correction = 0; for (int i = 0; i < rank; ++i) correction += latent*.25f;
                correction_expected = correction; bias_expected = correction_bias ? .125f : 0;
                ref += correction; if (correction_bias) ref += .125f;
            }
            if (llama_encode(ctx, batch)) { fprintf(stderr, "encode failure\n"); return 1; }
            if (!audit.correct || audit.arithmetic_nodes == 0) { fprintf(stderr,"encoder device audit failed\n"); return 1; }
            ++graph_cases; graph_nodes += audit.arithmetic_nodes;
            auto * output = llama_get_embeddings(ctx);
            if (!output) { fprintf(stderr, "missing output\n"); return 1; }
            std::vector<float> expected_output(64,ref), actual_output(output,output+64);
            measurements["encoder_cases"].push_back({{"bits",bits},{"mode",mode},{"correction_rank",correction_rank},{"correction_bias",correction_bias},{"raw_input_f32_hex",hex_bytes(batch.embd,192*sizeof(float))},{"expected_output_f32_hex",hex_bytes(expected_output.data(),64*sizeof(float))},{"native_output_f32_hex",hex_bytes(actual_output.data(),64*sizeof(float))},{"base_expected_f32",base_expected},{"correction_expected_f32",correction_expected},{"bias_expected_f32",bias_expected},{"relative_rms",relative_rms(expected_output,actual_output)},{"nodes",audit.nodes}});
            checkpoint_report();
            for (int i = 0; i < 64; ++i) if (!std::isfinite(output[i]) || std::abs(output[i]-ref) > 1e-4) { fprintf(stderr, "numeric mismatch %g vs %g\n", output[i], ref); return 1; }
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        }
    }
    printf("EAGLE3 fixtures passed backend=%s exact_pack_cases=%d loader_cases=%d actual_graph_cases=%d arithmetic_nodes=%d\n",selected.requested,pack_cases,loader_cases,graph_cases,graph_nodes);
    measurements["counters"] = {{"pack_cases",pack_cases},{"loader_cases",loader_cases},{"graph_cases",graph_cases},{"arithmetic_nodes",graph_nodes},{"projection_cases",measurements["projection_cases"].size()}};
    ggml_backend_free(selected.backend); llama_backend_free();
    measurements["status"] = "passed"; measurements.erase("failure_reason"); checkpoint_report();
}
