#include "models.h"

#include <algorithm>
#include <set>

void llama_model_eagle3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("EAGLE3 model requires 'extract_layers' in GGUF metadata");
    }
    if (target_layer_ids.size() != 3) {
        throw std::runtime_error("EAGLE3 requires exactly 3 entries in 'extract_layers'");
    }
    LLAMA_LOG_INFO("%s: EAGLE3 extract_layers = [%d, %d, %d]\n", __func__,
            target_layer_ids[0],
            target_layer_ids[1],
            target_layer_ids[2]);

    uint32_t n_embd_tgt = 0;

    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt);
    LLAMA_LOG_INFO("%s: EAGLE3 n_embd_tgt = %u (draft n_embd = %u)\n", __func__, n_embd_tgt, hparams.n_embd);

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    // eagle3 norm_before_residual (optional, default false)
    // compatible with Readhat eagle3 speculator model
    ml.get_key(LLM_KV_NORM_BEFORE_RESIDUAL, hparams.norm_before_residual, false);
    if (hparams.norm_before_residual) {
        LLAMA_LOG_INFO("%s: EAGLE3gnorm_before_residual = true\n", __func__);
    }

    // eagle3 norm_before_fc (optional, default false)
    // compatible with eagle3.1 (e.g. nvidia/gpt-oss-120b-Eagle3-v3)
    ml.get_key(LLM_KV_NORM_BEFORE_FC, hparams.norm_before_fc, false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_eagle3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    const int64_t n_embd_attn_input = 2 * n_embd;

    // Get vocab size from the d2t tensor in the GGUF file (optional - only needed if eagle3 has different vocab_size than target)
    // d2t: draft to target vocabulary mapping
    int64_t n_draft_vocab = n_vocab;  // Default: same as target vocab
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_draft_vocab = d2t_meta->ne[0]; // update draft vocab size
        d2t = create_tensor(tn(LLM_TENSOR_D2T), {n_draft_vocab}, 0);
        LLAMA_LOG_INFO("%s: EAGLE3 using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long)n_draft_vocab);
    } else {
        d2t = nullptr; // no d2t, use default vocab size
        LLAMA_LOG_INFO("%s: EAGLE3 without d2t - sharing same vocab_size with target (vocab_size = %lld)\n", __func__, (long long)n_draft_vocab);
    }

    // RMSNorm on the fused target features (input to fc), only when norm_before_fc is set.
    if (hparams.norm_before_fc) {
        output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd_inp}, 0);
    }

    // Output layer (uses draft vocab size)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    uint32_t full_w1a1_version = 0;
    const bool has_full_w1a1 = ml->get_key("eagle3.w1a1.version", full_w1a1_version, false);
    std::vector<std::string> w1a1_groups, w1a1_tensors;
    if (has_full_w1a1) {
        std::string bit_order, sign_rule, scale_rule, arithmetic;
        ml->get_arr("eagle3.w1a1.groups", w1a1_groups);
        ml->get_arr("eagle3.w1a1.tensors", w1a1_tensors);
        ml->get_key("eagle3.w1a1.bit_order", bit_order);
        ml->get_key("eagle3.w1a1.sign_rule", sign_rule);
        ml->get_key("eagle3.w1a1.scale_rule", scale_rule);
        ml->get_key("eagle3.w1a1.arithmetic", arithmetic);
        const std::map<std::string, std::vector<std::string>> group_tensors = {
            {"fusion", {"fc.weight"}},
            {"attention", {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight"}},
            {"ffn", {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_up.weight"}},
            {"head", {"output.weight"}},
        };
        std::set<std::string> expected;
        for (const auto & group : w1a1_groups) {
            const auto it = group_tensors.find(group);
            if (it == group_tensors.end()) {
                throw std::runtime_error("EAGLE3 W1A1 metadata contains an unknown group");
            }
            expected.insert(it->second.begin(), it->second.end());
        }
        const std::set<std::string> declared(w1a1_tensors.begin(), w1a1_tensors.end());
        const std::set<std::string> declared_groups(w1a1_groups.begin(), w1a1_groups.end());
        if (full_w1a1_version != 1 || w1a1_groups.empty() || declared_groups.size() != w1a1_groups.size() ||
                declared.size() != w1a1_tensors.size() ||
                declared != expected || bit_order != "little" || sign_rule != "nonnegative_is_one" ||
                scale_rule != "f32_mean_abs" || arithmetic != "f32") {
            throw std::runtime_error("EAGLE3 W1A1 full-drafter metadata has an unsupported or incomplete audit record");
        }
        std::string coverage;
        for (const auto & group : w1a1_groups) coverage += (coverage.empty() ? "" : ",") + group;
        LLAMA_LOG_INFO("%s: EAGLE3 W1A1 active groups: %s (%zu tensors)\n", __func__, coverage.c_str(), declared.size());
    }

    uint32_t w8a8_version = 0;
    const bool has_w8a8 = ml->get_key("eagle3.w8a8.version", w8a8_version, false);
    const std::set<std::string> w8a8_expected = {
        "fc.weight", "output.weight",
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_up.weight",
    };
    if (has_w8a8) {
        uint32_t legacy_w1a1_version = 0;
        if (has_full_w1a1 || ml->get_key("eagle3.w1a1_head.version", legacy_w1a1_version, false)) {
            throw std::runtime_error("EAGLE3 W8A8 cannot be combined with W1A1 metadata");
        }
        std::vector<std::string> groups, names;
        std::string weight_dtype, activation_dtype, weight_scale, activation_scale;
        std::string rounding, zero_scale, accumulator, output_rule;
        ml->get_arr("eagle3.w8a8.groups", groups);
        ml->get_arr("eagle3.w8a8.tensors", names);
        ml->get_key("eagle3.w8a8.weight_dtype", weight_dtype);
        ml->get_key("eagle3.w8a8.activation_dtype", activation_dtype);
        ml->get_key("eagle3.w8a8.weight_scale", weight_scale);
        ml->get_key("eagle3.w8a8.activation_scale", activation_scale);
        ml->get_key("eagle3.w8a8.rounding", rounding);
        ml->get_key("eagle3.w8a8.zero_scale", zero_scale);
        ml->get_key("eagle3.w8a8.accumulator", accumulator);
        ml->get_key("eagle3.w8a8.output", output_rule);
        const std::set<std::string> declared(names.begin(), names.end());
        const std::set<std::string> declared_groups(groups.begin(), groups.end());
        if (w8a8_version != 1 || groups.size() != 4 || declared_groups != std::set<std::string>{"fusion", "attention", "ffn", "head"} ||
                names.size() != 9 || declared != w8a8_expected || weight_dtype != "i8_signed" || activation_dtype != "i8_signed" ||
                weight_scale != "f32_absmax_per_row" || activation_scale != "f32_absmax_per_token" ||
                rounding != "nearest_even" || zero_scale != "zero_codes" || accumulator != "i32" ||
                output_rule != "f32_dot_weight_activation_then_bias") {
            throw std::runtime_error("EAGLE3 W8A8 metadata has unsupported or incomplete all-linear coverage");
        }
        LLAMA_LOG_INFO("%s: EAGLE3 W8A8 active for all nine draft linears\n", __func__);
    }

    for (const auto & name : w8a8_expected) {
        const std::string base = name.substr(0, name.size() - std::string(".weight").size());
        const bool codes_present = ml->get_tensor_meta((base + ".w8a8_codes").c_str()) != nullptr;
        const bool scale_present = ml->get_tensor_meta((base + ".w8a8_scale").c_str()) != nullptr;
        if (codes_present != has_w8a8 || scale_present != has_w8a8) {
            throw std::runtime_error("EAGLE3 W8A8 metadata and all nine code/scale tensor pairs must agree");
        }
    }

    auto load_w8a8_linear = [&](llm_tensor tensor, int bid, const std::string & base_name,
            int64_t logical_k, int64_t rows, ggml_tensor *& dense,
            ggml_tensor *& codes, ggml_tensor *& scales) {
        const std::string base = base_name.substr(0, base_name.size() - std::string(".weight").size());
        const std::string codes_name = base + ".w8a8_codes";
        const std::string scale_name = base + ".w8a8_scale";
        const auto * codes_meta = ml->get_tensor_meta(codes_name.c_str());
        const auto * scale_meta = ml->get_tensor_meta(scale_name.c_str());
        if (!codes_meta || !scale_meta || ml->get_tensor_meta(base_name.c_str()) != nullptr ||
                ml->get_tensor_meta((base + ".w1a1_packed").c_str()) != nullptr ||
                ml->get_tensor_meta((base + ".w1a1_scale").c_str()) != nullptr ||
                codes_meta->type != GGML_TYPE_I8 || scale_meta->type != GGML_TYPE_F32 ||
                codes_meta->ne[0] != logical_k || codes_meta->ne[1] != rows ||
                codes_meta->ne[2] != 1 || codes_meta->ne[3] != 1 ||
                scale_meta->ne[0] != rows || scale_meta->ne[1] != 1 ||
                scale_meta->ne[2] != 1 || scale_meta->ne[3] != 1) {
            throw std::runtime_error("EAGLE3 W8A8 requires I8 codes and F32 row scales with no dense/W1A1 shadow");
        }
        std::string suffix = base_name;
        std::replace(suffix.begin(), suffix.end(), '.', '_');
        const std::string audit = "eagle3.w8a8.tensor." + suffix;
        uint32_t audited_k = 0;
        std::string audited_codes, audited_scale;
        ml->get_key((audit + ".logical_k").c_str(), audited_k);
        ml->get_key((audit + ".codes").c_str(), audited_codes);
        ml->get_key((audit + ".scale").c_str(), audited_scale);
        if (audited_k != logical_k || audited_codes != codes_name || audited_scale != scale_name) {
            throw std::runtime_error("EAGLE3 W8A8 tensor audit metadata disagrees with its shape or name");
        }
        codes = create_tensor(LLM_TN_IMPL(LLM_ARCH_EAGLE3, tensor, "w8a8_codes", bid, -1), {logical_k, rows}, 0);
        scales = create_tensor(LLM_TN_IMPL(LLM_ARCH_EAGLE3, tensor, "w8a8_scale", bid, -1), {rows}, 0);
        dense = nullptr;
    };

    auto load_w1a1_linear = [&](llm_tensor tensor, int bid, const std::string & base_name,
            int64_t logical_k, int64_t rows, ggml_tensor *& dense,
            ggml_tensor *& packed, ggml_tensor *& scales, int flags) {
        const std::string packed_name = base_name.substr(0, base_name.size() - std::string(".weight").size()) + ".w1a1_packed";
        const std::string scale_name = base_name.substr(0, base_name.size() - std::string(".weight").size()) + ".w1a1_scale";
        const bool declared = std::find(w1a1_tensors.begin(), w1a1_tensors.end(), base_name) != w1a1_tensors.end();
        const auto * packed_meta = ml->get_tensor_meta(packed_name.c_str());
        const auto * scale_meta = ml->get_tensor_meta(scale_name.c_str());
        if (declared != (packed_meta != nullptr && scale_meta != nullptr) || (packed_meta != nullptr) != (scale_meta != nullptr)) {
            throw std::runtime_error("EAGLE3 W1A1 declared tensor requires both packed data and scales");
        }
        if (!declared) {
            dense = create_tensor(tn(tensor, "weight", bid), {logical_k, rows}, flags);
            return;
        }
        if (ml->get_tensor_meta(base_name.c_str()) != nullptr || packed_meta->type != GGML_TYPE_I32 || scale_meta->type != GGML_TYPE_F32) {
            throw std::runtime_error("EAGLE3 W1A1 tensor must omit its dense shadow and use I32 signs/F32 scales");
        }
        std::string suffix = base_name;
        std::replace(suffix.begin(), suffix.end(), '.', '_');
        uint32_t audited_k = 0;
        std::string audited_packed, audited_scale;
        const std::string audit = "eagle3.w1a1.tensor." + suffix;
        ml->get_key((audit + ".logical_k").c_str(), audited_k);
        ml->get_key((audit + ".packed").c_str(), audited_packed);
        ml->get_key((audit + ".scale").c_str(), audited_scale);
        if (audited_k != logical_k || audited_packed != packed_name || audited_scale != scale_name ||
                packed_meta->ne[0] != (logical_k + 31) / 32 || packed_meta->ne[1] != rows ||
                scale_meta->ne[0] != rows) {
            throw std::runtime_error("EAGLE3 W1A1 tensor shape or audit metadata mismatch");
        }
        packed = create_tensor(LLM_TN_IMPL(LLM_ARCH_EAGLE3, tensor, "w1a1_packed", bid, -1), {(logical_k + 31) / 32, rows}, 0);
        scales = create_tensor(LLM_TN_IMPL(LLM_ARCH_EAGLE3, tensor, "w1a1_scale", bid, -1), {rows}, 0);
        dense = nullptr;
        LLAMA_LOG_INFO("%s: EAGLE3 W1A1 loaded %s (K=%lld, rows=%lld)\n", __func__, base_name.c_str(),
                (long long) logical_k, (long long) rows);
    };

    if (has_full_w1a1 && ml->get_key("eagle3.w1a1_head.version", full_w1a1_version, false)) {
        throw std::runtime_error("EAGLE3 full W1A1 and legacy head metadata cannot be combined");
    }

    // Feature fusion layer: projects 3 target layers to draft hidden size.
    if (has_full_w1a1) {
        load_w1a1_linear(LLM_TENSOR_FC, -1, tn(LLM_TENSOR_FC, "weight").str(), n_embd_inp, n_embd,
                fc, fc_w1a1_packed, fc_w1a1_scale, 0);
    } else if (has_w8a8) {
        load_w8a8_linear(LLM_TENSOR_FC, -1, tn(LLM_TENSOR_FC, "weight").str(), n_embd_inp, n_embd,
                fc, fc_w8a8_codes, fc_w8a8_scale);
    } else {
        fc = create_tensor(tn(LLM_TENSOR_FC, "weight"), {n_embd_inp, n_embd}, 0);
    }

    // The packed head is opt-in through a versioned GGUF contract. Do not
    // retain a dense shadow copy and never substitute the target head for a
    // malformed packed draft head.
    uint32_t w1a1_version = 0;
    const bool has_w1a1 = ml->get_key("eagle3.w1a1_head.version", w1a1_version, false);
    const bool has_packed = ml->get_tensor_meta(tn(LLM_TENSOR_OUTPUT_W1A1_PACKED).str().c_str()) != nullptr;
    const bool has_scale  = ml->get_tensor_meta(tn(LLM_TENSOR_OUTPUT_W1A1_SCALE ).str().c_str()) != nullptr;
    if (!has_full_w1a1 && (has_w1a1 != (has_packed && has_scale) || has_packed != has_scale)) {
        throw std::runtime_error("EAGLE3 packed head requires metadata and both companion tensors");
    }
    if (has_full_w1a1) {
        load_w1a1_linear(LLM_TENSOR_OUTPUT, -1, tn(LLM_TENSOR_OUTPUT, "weight").str(), n_embd, n_draft_vocab,
                output, output_w1a1_packed, output_w1a1_scale, TENSOR_NOT_REQUIRED);
    } else if (has_w8a8) {
        load_w8a8_linear(LLM_TENSOR_OUTPUT, -1, tn(LLM_TENSOR_OUTPUT, "weight").str(), n_embd, n_draft_vocab,
                output, output_w8a8_codes, output_w8a8_scale);
    } else if (has_w1a1) {
        uint32_t logical_k = 0;
        std::string packed_name, scale_name, bit_order, sign_rule, scale_rule, arithmetic;
        ml->get_key("eagle3.w1a1_head.logical_k", logical_k);
        ml->get_key("eagle3.w1a1_head.packed_tensor", packed_name);
        ml->get_key("eagle3.w1a1_head.scale_tensor", scale_name);
        ml->get_key("eagle3.w1a1_head.bit_order", bit_order);
        ml->get_key("eagle3.w1a1_head.sign_rule", sign_rule);
        ml->get_key("eagle3.w1a1_head.scale_rule", scale_rule);
        ml->get_key("eagle3.w1a1_head.arithmetic", arithmetic);
        if (w1a1_version != 1 || logical_k != n_embd ||
                packed_name != tn(LLM_TENSOR_OUTPUT_W1A1_PACKED).str() ||
                scale_name  != tn(LLM_TENSOR_OUTPUT_W1A1_SCALE ).str() ||
                bit_order != "little" || sign_rule != "nonnegative_is_one" ||
                scale_rule != "f32_mean_abs" || arithmetic != "f32" ||
                ml->get_tensor_meta(tn(LLM_TENSOR_OUTPUT, "weight").str().c_str()) != nullptr) {
            throw std::runtime_error("EAGLE3 packed head has unsupported metadata or a dense shadow head");
        }

        const auto * packed_meta = ml->get_tensor_meta(packed_name.c_str());
        const auto * scale_meta  = ml->get_tensor_meta(scale_name.c_str());
        if (packed_meta->type != GGML_TYPE_I32 || scale_meta->type != GGML_TYPE_F32) {
            throw std::runtime_error("EAGLE3 packed head requires I32 signs and F32 scales");
        }
        output_w1a1_packed = create_tensor(tn(LLM_TENSOR_OUTPUT_W1A1_PACKED), {(n_embd + 31)/32, n_draft_vocab}, 0);
        output_w1a1_scale  = create_tensor(tn(LLM_TENSOR_OUTPUT_W1A1_SCALE ), {n_draft_vocab}, 0);
        output = nullptr;
        LLAMA_LOG_INFO("%s: EAGLE3 using packed W1A1 draft head (K = %lld, rows = %lld)\n",
                __func__, (long long) n_embd, (long long) n_draft_vocab);
    } else {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_draft_vocab}, TENSOR_NOT_REQUIRED);
    }

    // Token embeddings (optional - Llama 3.3 70B EAGLE3 has its own)
    const struct ggml_tensor * tok_embd_meta = ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    if (tok_embd_meta) {
        const int64_t n_target_vocab = tok_embd_meta->ne[1];
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_target_vocab}, 0);
        LLAMA_LOG_INFO("%s: EAGLE3 using its own token_embd (vocab = %lld)\n", __func__, (long long)n_target_vocab);
    }

    // Single decoder layer
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // input_layernorm: applied to token embeddings
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // eagle3 specific: hidden_norm applied to fused target features
        layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);

        // Attention takes input_embeds_normed + fused_target_normed as input
        if (has_full_w1a1) {
            load_w1a1_linear(LLM_TENSOR_ATTN_Q, i, tn(LLM_TENSOR_ATTN_Q, "weight", i).str(), n_embd_attn_input, n_embd_head_k * n_head,
                    layer.wq, layer.wq_w1a1_packed, layer.wq_w1a1_scale, 0);
            load_w1a1_linear(LLM_TENSOR_ATTN_K, i, tn(LLM_TENSOR_ATTN_K, "weight", i).str(), n_embd_attn_input, n_embd_k_gqa,
                    layer.wk, layer.wk_w1a1_packed, layer.wk_w1a1_scale, 0);
            load_w1a1_linear(LLM_TENSOR_ATTN_V, i, tn(LLM_TENSOR_ATTN_V, "weight", i).str(), n_embd_attn_input, n_embd_v_gqa,
                    layer.wv, layer.wv_w1a1_packed, layer.wv_w1a1_scale, 0);
            load_w1a1_linear(LLM_TENSOR_ATTN_OUT, i, tn(LLM_TENSOR_ATTN_OUT, "weight", i).str(), n_embd_head_k * n_head, n_embd,
                    layer.wo, layer.wo_w1a1_packed, layer.wo_w1a1_scale, 0);
        } else if (has_w8a8) {
            load_w8a8_linear(LLM_TENSOR_ATTN_Q, i, tn(LLM_TENSOR_ATTN_Q, "weight", i).str(), n_embd_attn_input, n_embd_head_k * n_head,
                    layer.wq, layer.wq_w8a8_codes, layer.wq_w8a8_scale);
            load_w8a8_linear(LLM_TENSOR_ATTN_K, i, tn(LLM_TENSOR_ATTN_K, "weight", i).str(), n_embd_attn_input, n_embd_k_gqa,
                    layer.wk, layer.wk_w8a8_codes, layer.wk_w8a8_scale);
            load_w8a8_linear(LLM_TENSOR_ATTN_V, i, tn(LLM_TENSOR_ATTN_V, "weight", i).str(), n_embd_attn_input, n_embd_v_gqa,
                    layer.wv, layer.wv_w8a8_codes, layer.wv_w8a8_scale);
            load_w8a8_linear(LLM_TENSOR_ATTN_OUT, i, tn(LLM_TENSOR_ATTN_OUT, "weight", i).str(), n_embd_head_k * n_head, n_embd,
                    layer.wo, layer.wo_w8a8_codes, layer.wo_w8a8_scale);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd_attn_input, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd_attn_input, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd_attn_input, n_embd_v_gqa}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (has_full_w1a1) {
            load_w1a1_linear(LLM_TENSOR_FFN_GATE, i, tn(LLM_TENSOR_FFN_GATE, "weight", i).str(), n_embd, n_ff,
                    layer.ffn_gate, layer.ffn_gate_w1a1_packed, layer.ffn_gate_w1a1_scale, 0);
            load_w1a1_linear(LLM_TENSOR_FFN_DOWN, i, tn(LLM_TENSOR_FFN_DOWN, "weight", i).str(), n_ff, n_embd,
                    layer.ffn_down, layer.ffn_down_w1a1_packed, layer.ffn_down_w1a1_scale, 0);
            load_w1a1_linear(LLM_TENSOR_FFN_UP, i, tn(LLM_TENSOR_FFN_UP, "weight", i).str(), n_embd, n_ff,
                    layer.ffn_up, layer.ffn_up_w1a1_packed, layer.ffn_up_w1a1_scale, 0);
        } else if (has_w8a8) {
            load_w8a8_linear(LLM_TENSOR_FFN_GATE, i, tn(LLM_TENSOR_FFN_GATE, "weight", i).str(), n_embd, n_ff,
                    layer.ffn_gate, layer.ffn_gate_w8a8_codes, layer.ffn_gate_w8a8_scale);
            load_w8a8_linear(LLM_TENSOR_FFN_DOWN, i, tn(LLM_TENSOR_FFN_DOWN, "weight", i).str(), n_ff, n_embd,
                    layer.ffn_down, layer.ffn_down_w8a8_codes, layer.ffn_down_w8a8_scale);
            load_w8a8_linear(LLM_TENSOR_FFN_UP, i, tn(LLM_TENSOR_FFN_UP, "weight", i).str(), n_embd, n_ff,
                    layer.ffn_up, layer.ffn_up_w8a8_codes, layer.ffn_up_w8a8_scale);
        } else {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        }

        // rope_freqs for llama3 rope scaling (optional - only if eagle3 config has rope_scaling)
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED);
    }
}

template <>
ggml_tensor * llama_model_eagle3::graph<true>::build_inp_embd_enc() const {
    ggml_tensor * cur = nullptr;

    // Input: Target model features (3 layers concatenated: low, mid, high)
    // Data will be provided via ubatch->embd in encode_eagle3_features()
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// eagle3 Encoder: processes target model features through feature fusion layer
// Input: target_features e.g. [12288, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [4096, n_tokens] stored in context
template <>
llama_model_eagle3::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;
    auto eagle_linear = [&](ggml_tensor * dense, ggml_tensor * packed, ggml_tensor * scales,
            ggml_tensor * codes8, ggml_tensor * scales8, ggml_tensor * input, int64_t logical_k) {
        if (codes8) {
            if (!loras->empty()) throw std::runtime_error("EAGLE3 W8A8 projections do not support draft LoRA adapters");
            if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx0, input, GGML_TYPE_F32);
            return ggml_w8a8_mul_mat(ctx0, codes8, scales8, input);
        }
        if (!packed) return build_lora_mm(dense, input);
        if (!loras->empty()) throw std::runtime_error("EAGLE3 packed W1A1 projections do not support draft LoRA adapters");
        if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx0, input, GGML_TYPE_F32);
        return ggml_w1a1_mul_mat(ctx0, packed, scales, input, logical_k);
    };

    cur = build_inp_embd_enc();

    // RMSNorm on the fused target features before fc
    if (hparams.norm_before_fc) {
        cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
        cb(cur, "enc_input_norm", -1);
    }

    // Feature fusion layer
    cur = eagle_linear(model.fc, model.fc_w1a1_packed, model.fc_w1a1_scale,
            model.fc_w8a8_codes, model.fc_w8a8_scale,
            cur, hparams.n_embd_inp_enc());
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [4096, n_tokens]
    // store in t_h_nextn (same as MTP) so can be read via llama_get_embeddings_nextn(ctx_dft)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// eagle3 Decoder: processes draft tokens using g_embeddings from encoder
// Input: draft tokens + g_embeddings from encoder
// Output: draft logits
template <>
llama_model_eagle3::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_layer == 1);  // eagle3 has only one decoder layer

    ggml_tensor * cur;
    ggml_tensor * inpL;
    auto eagle_linear = [&](ggml_tensor * dense, ggml_tensor * packed, ggml_tensor * scales,
            ggml_tensor * codes8, ggml_tensor * scales8, ggml_tensor * input, int64_t logical_k) {
        if (codes8) {
            if (!loras->empty()) throw std::runtime_error("EAGLE3 W8A8 projections do not support draft LoRA adapters");
            if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx0, input, GGML_TYPE_F32);
            return ggml_w8a8_mul_mat(ctx0, codes8, scales8, input);
        }
        if (!packed) return build_lora_mm(dense, input);
        if (!loras->empty()) throw std::runtime_error("EAGLE3 packed W1A1 projections do not support draft LoRA adapters");
        if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx0, input, GGML_TYPE_F32);
        return ggml_w1a1_mul_mat(ctx0, packed, scales, input, logical_k);
    };

    // eagle3 Decoder receives:
    // 1. Token embeddings (e.g.from eagle3's own tok_embd for Llama 3.3 70B, or target model for Llama 3.1 8B)
    // 2. g_embeddings from encoder
    auto * tok_embd = model.tok_embd;
    if (model.tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "EAGLE3 decoder requires token embeddings (own or from target model)");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * inp_embd = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inp_embd, "inp_embd", -1);

    ggml_tensor * inp_g = inp->embd;
    cb(inp_g, "inp_g_embeddings", -1);

    res->add_input(std::move(inp));

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        // Apply input_layernorm to the token embeddings
        ggml_tensor * embd_norm = build_norm(inp_embd,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // Apply hidden_norm to inp_g
        ggml_tensor * g_norm = build_norm(inp_g,
                model.layers[il].attn_norm_2, NULL,
                LLM_NORM_RMS, -1);
        cb(g_norm, "g_norm", il);

        // norm_before_residual: determines what goes into the residual connection (compatible with Readhat eagle3 speculator model)
        // - false (default): use raw inp_g for residual
        // - true: use normalized g_norm for residual
        // inpL is the concatenated input (normalized inp_embd + normalized inp_g)
        ggml_tensor * inpSA = hparams.norm_before_residual ? g_norm : inpL;

        // Concatenate normalized inp_embd and normalized inp_g
        cur = ggml_concat(ctx0, embd_norm, g_norm, il);
        cb(cur, "concat_embd", il);

        // Self-attention with concatenated input
        ggml_tensor * Qcur = eagle_linear(model.layers[il].wq,
                model.layers[il].wq_w1a1_packed, model.layers[il].wq_w1a1_scale,
                model.layers[il].wq_w8a8_codes, model.layers[il].wq_w8a8_scale, cur, 2 * n_embd);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = eagle_linear(model.layers[il].wk,
                model.layers[il].wk_w1a1_packed, model.layers[il].wk_w1a1_scale,
                model.layers[il].wk_w8a8_codes, model.layers[il].wk_w8a8_scale, cur, 2 * n_embd);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = eagle_linear(model.layers[il].wv,
                model.layers[il].wv_w1a1_packed, model.layers[il].wv_w1a1_scale,
                model.layers[il].wv_w8a8_codes, model.layers[il].wv_w8a8_scale, cur, 2 * n_embd);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        cur = build_attn(inp_attn,
                (model.layers[il].wo_w1a1_packed || model.layers[il].wo_w8a8_codes) ? nullptr : model.layers[il].wo, NULL, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        if (model.layers[il].wo_w1a1_packed || model.layers[il].wo_w8a8_codes) {
            cur = eagle_linear(nullptr, model.layers[il].wo_w1a1_packed,
                    model.layers[il].wo_w1a1_scale,
                    model.layers[il].wo_w8a8_codes, model.layers[il].wo_w8a8_scale, cur, n_embd_head * n_head);
        }

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        const auto & layer = model.layers[il];
        if (layer.ffn_up_w1a1_packed || layer.ffn_gate_w1a1_packed || layer.ffn_down_w1a1_packed ||
                layer.ffn_up_w8a8_codes || layer.ffn_gate_w8a8_codes || layer.ffn_down_w8a8_codes) {
            ggml_tensor * up = eagle_linear(layer.ffn_up, layer.ffn_up_w1a1_packed,
                    layer.ffn_up_w1a1_scale, layer.ffn_up_w8a8_codes, layer.ffn_up_w8a8_scale, cur, n_embd);
            ggml_tensor * gate = eagle_linear(layer.ffn_gate, layer.ffn_gate_w1a1_packed,
                    layer.ffn_gate_w1a1_scale, layer.ffn_gate_w8a8_codes, layer.ffn_gate_w8a8_scale, cur, n_embd);
            gate = ggml_silu(ctx0, gate);
            cur = ggml_mul(ctx0, up, gate);
            cur = eagle_linear(layer.ffn_down, layer.ffn_down_w1a1_packed,
                    layer.ffn_down_w1a1_scale, layer.ffn_down_w8a8_codes, layer.ffn_down_w8a8_scale, cur, hparams.n_ff(il));
        } else {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }

    cur = inpL;

    // Output prenorm state (for next token's g_embeddings in autoregressive generation)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head - projects to draft vocabulary
    // if the draft has no own output projection, inherit the target model's lm_head
    if (model.output_w1a1_packed != nullptr || model.output_w8a8_codes != nullptr) {
        if (!loras->empty()) {
            throw std::runtime_error("EAGLE3 low-bit head does not support draft LoRA adapters");
        }
        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        }
        if (model.output_w8a8_codes) {
            cur = ggml_w8a8_mul_mat(ctx0, model.output_w8a8_codes, model.output_w8a8_scale, cur);
        } else {
            cur = ggml_w1a1_mul_mat(ctx0, model.output_w1a1_packed, model.output_w1a1_scale, cur, hparams.n_embd);
        }
    } else {
        auto * output = model.output;
        if (output == nullptr) {
            GGML_ASSERT(cparams.ctx_other != nullptr);
            const auto * model_other = llama_get_model(cparams.ctx_other);

            GGML_ASSERT(model_other->output != nullptr && "EAGLE3 decoder requires an output projection (own or from target model)");
            output = model_other->output;
        }
        cur = build_lora_mm(output, cur);
    }

    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_eagle3::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
