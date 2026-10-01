#include "w1a1.cuh"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// Each lane owns a complete little-bit-order sign word. Accumulate absolute
// values in double, as the CPU implementation does, before rounding the row
// mean to F32. The unused tail bits remain zero.
static __global__ void w1a1_pack_activations(
        const float * activations, int64_t k, int64_t words, int64_t n,
        uint32_t * packed, float * scales, float delta) {
    __shared__ double sums[256];
    for (int64_t token = blockIdx.x; token < n; token += gridDim.x) {
        double sum = 0.0;
        const float * row = activations + token*k;
        for (int64_t word = threadIdx.x; word < words; word += blockDim.x) {
            for (int bit = 0; bit < 32 && word*32 + bit < k; ++bit) sum += (double) __uint_as_float(__float_as_uint(row[word*32 + bit]) & 0x7fffffffu);
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) scales[token] = __double2float_rn(sums[0] / (double) k);
        __syncthreads();
        const float threshold = __fmul_rn(delta, scales[token]);
        for (int64_t word = threadIdx.x; word < words; word += blockDim.x) {
            uint32_t bits = 0;
            for (int bit = 0; bit < 32 && word*32 + bit < k; ++bit) {
                const float value = (__float_as_uint(delta) & 0x7fffffffu) == 0 ? row[word*32 + bit] : __fsub_rn(row[word*32 + bit], threshold);
                const uint32_t raw = __float_as_uint(value);
                const bool positive = (raw & 0x80000000u) == 0 || (raw & 0x7fffffffu) == 0;
                bits |= (uint32_t) positive << bit;
            }
            packed[token*words + word] = bits;
        }
        __syncthreads();
    }
}

// Four rows per block, one group of 32 lanes per row. Shared-memory reduction
// keeps the portable POPC path usable by all variants of ggml-cuda.
static __global__ void w1a1_xor_popc(
        const uint32_t * weights, const float * weight_scales,
        const uint32_t * activations, const float * activation_scales,
        int64_t m, int64_t n, int64_t k, int64_t words, float * output, bool warp_reduce, const float * midpoints, const float * code_sums) {
    __shared__ int counts[4][32];
    const int64_t row = (int64_t) blockIdx.x*4 + threadIdx.y;
    const bool valid_row = row < m;
    const int lane = threadIdx.x;
    const uint32_t tail_mask = (k & 31) == 0 ? UINT32_MAX : (uint32_t(1) << (k & 31)) - 1;
    for (int64_t token = blockIdx.y; token < n; token += gridDim.y) {
        int mismatches = 0;
        if (valid_row) {
            for (int64_t word = lane; word < words; word += 32) {
                const uint32_t mask = word == words - 1 ? tail_mask : UINT32_MAX;
                mismatches += __popc((weights[row*words + word] ^ activations[token*words + word]) & mask);
            }
        }
        int total = mismatches;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (warp_reduce) {
            for (int stride = 16; stride > 0; stride /= 2) total += __shfl_down_sync(0xffffffffu, total, stride, 32);
        } else
#endif
        {
            counts[threadIdx.y][lane] = mismatches;
            __syncthreads();
            for (int stride = 16; stride > 0; stride /= 2) {
                if (lane < stride) counts[threadIdx.y][lane] += counts[threadIdx.y][lane + stride];
                __syncthreads();
            }
            total = counts[threadIdx.y][0];
        }
        if (valid_row && lane == 0) {
            const float dot = (float) (k - 2*(int64_t) total);
            const float weighted = dot*weight_scales[row];
            output[token*m + row] = midpoints ? __fadd_rn(__fmul_rn(__fmul_rn(dot, weight_scales[row]), activation_scales[token]), __fmul_rn(__fmul_rn(code_sums[token], midpoints[row]), activation_scales[token])) : weighted*activation_scales[token];
        }
        if (!warp_reduce) __syncthreads();
    }
}

// One block per token. Quantization is per complete logical token, with F32
// absmax/scale and round-to-nearest-even codes. Zero vectors produce zero codes.
// A4 planes contain two's-complement bits; unused tail bits are always clear.
static __global__ void w1ax_quantize(
        const float * activations, int64_t k, int64_t words, int bits,
        int8_t * codes, uint32_t * planes, float * scales, float clip, bool learned) {
    __shared__ float maxima[256];
    __shared__ float token_scale;
    const int64_t token = blockIdx.x;
    const float * row = activations + token*k;
    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        // ggml-cuda uses -use_fast_math: float min/max may flush subnormals.
        // Nonnegative finite IEEE bits have the same order as their magnitudes.
        local_max = learned ? __uint_as_float(max(__float_as_uint(local_max), __float_as_uint(row[i]) & 0x7fffffffu)) : fmaxf(local_max, fabsf(row[i]));
    }
    maxima[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] = learned ? __uint_as_float(max(__float_as_uint(maxima[threadIdx.x]), __float_as_uint(maxima[threadIdx.x + stride]))) : fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maxima[0] = __fmul_rn(maxima[0], clip);
        token_scale = __fdiv_rn(maxima[0], bits == 8 ? 127.0f : 7.0f);
        scales[token] = token_scale;
    }
    __syncthreads();
    const int qmax = bits == 8 ? 127 : 7;
    const bool zero_limit = learned ? (__float_as_uint(maxima[0]) & 0x7fffffffu) == 0 : maxima[0] == 0.0f;
    const float inv = zero_limit ? 0.0f : __fdiv_rn((float) qmax, maxima[0]);
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        const bool finite_inv = (__float_as_uint(inv) & 0x7f800000u) != 0x7f800000u;
        const float normalized = zero_limit ? 0.0f : (!learned || finite_inv) ? __fmul_rn(row[i], inv) : __double2float_rn((double) row[i]/(double) maxima[0]*qmax);
        const int q = __float2int_rn(learned ? fmaxf(-qmax, fminf(qmax, normalized)) : normalized);
        codes[token*k + i] = (int8_t) max(-qmax, min(qmax, q));
    }
    __syncthreads();
    if (bits == 4) {
        for (int64_t word = threadIdx.x; word < words; word += blockDim.x) {
            uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            for (int bit = 0; bit < 32 && word*32 + bit < k; ++bit) {
                const uint32_t q = (uint8_t) codes[token*k + word*32 + bit] & 15u;
                p0 |= ((q >> 0) & 1u) << bit;
                p1 |= ((q >> 1) & 1u) << bit;
                p2 |= ((q >> 2) & 1u) << bit;
                p3 |= ((q >> 3) & 1u) << bit;
            }
            const int64_t offset = (token*words + word)*4;
            planes[offset + 0] = p0;
            planes[offset + 1] = p1;
            planes[offset + 2] = p2;
            planes[offset + 3] = p3;
        }
    }
}

static __global__ void w1ax_integer_dot(
        const uint32_t * weights, const float * weight_scales,
        const int8_t * codes, const uint32_t * planes, const float * act_scales,
        int64_t m, int64_t k, int64_t words, int bits, bool bitserial, float * output, int32_t * raw_dots, bool warp_reduce, const float * midpoints, const float * code_sums) {
    __shared__ int sums[4][32];
    const int64_t row = (int64_t) blockIdx.x*4 + threadIdx.y;
    const int64_t token = blockIdx.y;
    const int lane = threadIdx.x;
    int sum = 0;
    if (row < m) {
        if (bits == 4 && bitserial) {
            for (int64_t word = lane; word < words; word += 32) {
                const uint32_t w = weights[row*words + word];
                const uint32_t * p = planes + (token*words + word)*4;
                sum += 2*__popc(w & p[0]) - __popc(p[0]);
                sum += 2*(2*__popc(w & p[1]) - __popc(p[1]));
                sum += 4*(2*__popc(w & p[2]) - __popc(p[2]));
                sum -= 8*(2*__popc(w & p[3]) - __popc(p[3]));
            }
        } else {
            for (int64_t i = lane; i < k; i += 32) {
                const int sign = (weights[row*words + i/32] & (1u << (i%32))) ? 1 : -1;
                sum += sign*(int) codes[token*k + i];
            }
        }
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (warp_reduce) {
        for (int stride = 16; stride > 0; stride /= 2) sum += __shfl_down_sync(0xffffffffu, sum, stride, 32);
    } else
#endif
    {
        sums[threadIdx.y][lane] = sum;
        __syncthreads();
        for (int stride = 16; stride > 0; stride /= 2) {
            if (lane < stride) sums[threadIdx.y][lane] += sums[threadIdx.y][lane + stride];
            __syncthreads();
        }
        sum = sums[threadIdx.y][0];
    }
    if (row < m && lane == 0) {
        if (raw_dots) raw_dots[token*m + row] = sum;
        const float weighted = (float) sum * weight_scales[row];
        output[token*m + row] = midpoints ? __fadd_rn(__fmul_rn(__fmul_rn((float) sum, weight_scales[row]), act_scales[token]), __fmul_rn(__fmul_rn(code_sums[token], midpoints[row]), act_scales[token])) : weighted * act_scales[token];
    }
}

// Independent scalar integer reference. Enable only for correctness runs;
// the production path never materializes raw dot values.
static __global__ void w1ax_validate_integer_dots(
        const uint32_t * weights, const int8_t * codes, const int32_t * raw_dots,
        int64_t m, int64_t n, int64_t k, int64_t words) {
    const int64_t index = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= m*n) return;
    const int64_t token = index / m;
    const int64_t row = index % m;
    int32_t reference = 0;
    for (int64_t i = 0; i < k; ++i) {
        const int sign = (weights[row*words + i/32] & (1u << (i%32))) ? 1 : -1;
        reference += sign*(int) codes[token*k + i];
    }
    if (reference != raw_dots[index]) asm("trap;");
}

// Explicit F32->FP16 cast happens at the operator boundary. Add signed FP16
// values in F32 reduction order, then apply the F32 binary-weight row scale.
static __global__ void w1a16_signadd(
        const uint32_t * weights, const float * weight_scales, const float * activations,
        int64_t m, int64_t k, int64_t words, float * output, const float * midpoints, const float * code_sums) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t token = blockIdx.y;
    if (row >= m) return;
    float sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        const float value = __half2float(__float2half_rn(activations[token*k + i]));
        const float signed_value = (weights[row*words + i/32] & (1u << (i%32))) ? value : -value;
        sum = midpoints ? __fadd_rn(sum, signed_value) : sum + signed_value;
    }
    output[token*m + row] = midpoints ? __fadd_rn(__fmul_rn(sum, weight_scales[row]), __fmul_rn(code_sums[token], midpoints[row])) : sum * weight_scales[row];
}

// Quality reference: separate F32 group products and sums, never FMA.
static __global__ void w1a16_group128_signadd(
        const uint32_t * weights, const float * scales, const float * acts,
        int64_t m, int64_t k, int64_t words, float * output) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t token = blockIdx.y;
    if (row >= m) return;
    const int64_t groups = (k + 127)/128;
    float total = 0.0f;
    for (int64_t group = 0; group < groups; ++group) {
        float dot = 0.0f;
        const int64_t end = k < (group + 1)*128 ? k : (group + 1)*128;
        for (int64_t i = group*128; i < end; ++i) {
            const float x = __half2float(__float2half_rn(acts[token*k + i]));
            dot = __fadd_rn(dot, (weights[row*words + i/32] & (1u << (i%32))) ? x : -x);
        }
        total = __fadd_rn(total, __fmul_rn(dot, scales[row*groups + group]));
    }
    output[token*m + row] = total;
}

// Diagnostic capture for identical-input replay. Enable only for a short run:
// GGML_W1AX_CAPTURE_DIR must name an existing directory. Files contain an
// eight-byte magic, five little-endian integer fields, a fixed 128-byte
// weight tensor name, then contiguous [N,K] F32 activation values. Capture
// synchronizes the stream and is intentionally excluded from timing trials.
// captures.jsonl adds per-process ggml_time_us timestamps for round attribution;
// its timestamp is taken before the activation download and synchronization.
static void w1ax_capture_activations(
        cudaStream_t stream, const ggml_tensor * weights, const ggml_tensor * acts,
        int64_t k, int64_t m, int64_t n, int bits) {
    static const char * directory = getenv("GGML_W1AX_CAPTURE_DIR");
    if (!directory || !*directory) return;
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) GGML_LOG_WARN("W1Ax activation capture requires CUDA graph capture disabled\n");
        return;
    }
    const int64_t timestamp_us = ggml_time_us();
    std::vector<float> host((size_t) k*n);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), acts->data, host.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (float value : host) GGML_ASSERT(std::isfinite(value));
    static std::atomic<unsigned long long> sequence{0};
    const unsigned long long id = sequence.fetch_add(1);
    char path[1024];
    const int len = snprintf(path, sizeof(path), "%s/op-%012llu.bin", directory, id);
    GGML_ASSERT(len > 0 && (size_t) len < sizeof(path));
    FILE * file = fopen(path, "wb");
    GGML_ASSERT(file != nullptr);
    const char magic[8] = {'W', '1', 'A', 'X', 'A', 'C', 'T', '1'};
    const uint64_t header[4] = {(uint64_t) id, (uint64_t) k, (uint64_t) m, (uint64_t) n};
    const uint32_t precision = (uint32_t) bits;
    char name[128] = {};
    strncpy(name, weights->name, sizeof(name) - 1);
    GGML_ASSERT(fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    GGML_ASSERT(fwrite(header, sizeof(uint64_t), 4, file) == 4);
    GGML_ASSERT(fwrite(&precision, sizeof(precision), 1, file) == 1);
    GGML_ASSERT(fwrite(name, 1, sizeof(name), file) == sizeof(name));
    GGML_ASSERT(fwrite(host.data(), sizeof(float), host.size(), file) == host.size());
    GGML_ASSERT(fclose(file) == 0);
    const int64_t capture_end_us = ggml_time_us();

    // Tensor names usually contain only dots and identifiers, but serialize
    // arbitrary valid names without allowing quotes/control bytes to break JSONL.
    std::string escaped_name;
    for (const unsigned char * p = (const unsigned char *) name; *p; ++p) {
        if (*p == '"' || *p == '\\') escaped_name += '\\';
        if (*p < 0x20) {
            char escape[7];
            snprintf(escape, sizeof(escape), "\\u%04x", (unsigned) *p);
            escaped_name += escape;
        } else {
            escaped_name += (char) *p;
        }
    }
    static std::mutex sidecar_mutex;
    const std::lock_guard<std::mutex> lock(sidecar_mutex);
    const int sidecar_len = snprintf(path, sizeof(path), "%s/captures.jsonl", directory);
    GGML_ASSERT(sidecar_len > 0 && (size_t) sidecar_len < sizeof(path));
    FILE * sidecar = fopen(path, "a");
    GGML_ASSERT(sidecar != nullptr);
    GGML_ASSERT(fprintf(sidecar,
            "{\"schema_version\":1,\"sequence\":%llu,\"file\":\"op-%012llu.bin\","
            "\"weight_tensor\":\"%s\",\"k\":%lld,\"m\":%lld,\"n\":%lld,"
            "\"bits\":%d,\"timestamp_us\":%lld,\"capture_start_us\":%lld,\"capture_end_us\":%lld}\n",
            id, id, escaped_name.c_str(), (long long) k, (long long) m,
            (long long) n, bits, (long long) timestamp_us,
            (long long) timestamp_us, (long long) capture_end_us) > 0);
    GGML_ASSERT(fclose(sidecar) == 0);
}

// One shared S per token, derived from the exact emitted codes. A16 uses the
// same F16 boundary values as signadd, accumulated in sequential F32 order.
static __global__ void w1ax_code_sum(const float * acts, const uint32_t * packed,
        int64_t k, int64_t n, int bits, float * sums, float * scales) {
    const int64_t token = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (token >= n) return;
    if (bits == 16) {
        float sum = 0;
        for (int64_t i = 0; i < k; ++i) sum = __fadd_rn(sum, __half2float(__float2half_rn(acts[token*k+i])));
        sums[token] = sum; scales[token] = 1;
    } else {
        int32_t sum = 0;
        const int64_t words = (k+31)/32;
        for (int64_t i = 0; i < k; ++i) sum += bits == 1 ? ((packed[token*words+i/32] >> (i%32)) & 1u ? 1 : -1) : ((const int8_t *) packed)[token*k+i];
        sums[token] = (float) sum;
    }
}

void ggml_cuda_w1ax_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_eagle_scope eagle_pack_scope(ctx, "activation_pack_scales", dst);
    const ggml_tensor * acts = dst->src[0];
    const int bits = ggml_get_op_params_i32(dst, 0);
    const float delta = ggml_get_op_params_f32(dst, 1), clip = ggml_get_op_params_f32(dst, 2);
    const bool learned = ggml_get_op_params_i32(dst, 3);
    const int64_t k = acts->ne[0], n = acts->ne[1], words = (k + 31)/32;
    const bool affine = ggml_get_op_params_i32(dst, 4);
    const struct ggml_w1ax_pack_layout layout = affine ? ggml_w1ax_affine_layout(k, n, bits) : ggml_w1ax_pack_layout(k, n, bits);
    uint32_t * data = (uint32_t *) dst->data;
    float * scales = (float *) (data + layout.scale_offset);
    GGML_ASSERT(bits == 1 || bits == 4 || bits == 8 || (bits == 16 && affine));
    GGML_ASSERT(n <= INT_MAX);
    cudaStream_t stream = ctx.stream();
    if (bits == 1) {
        w1a1_pack_activations<<<dim3((unsigned) n), 256, 0, stream>>>(
                (const float *) acts->data, k, words, n, data, scales, delta);
    } else if (bits != 16) {
        if (layout.codes_words*4 > k*n) CUDA_CHECK(cudaMemsetAsync((int8_t *) data + k*n, 0, layout.codes_words*4 - k*n, stream));
        w1ax_quantize<<<dim3((unsigned) n), 256, 0, stream>>>(
                (const float *) acts->data, k, words, bits, (int8_t *) data, data + layout.codes_words, scales, clip, learned);
    }
    if (affine) w1ax_code_sum<<<dim3((unsigned) ((n+127)/128)),128,0,stream>>>((const float *) acts->data, data, k,n,bits,(float *) (data+layout.scale_offset+n),scales);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_w1a1_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * scales  = dst->src[1];
    const ggml_tensor * acts    = dst->src[2];
    int64_t k;
    memcpy(&k, dst->op_params, sizeof(k));
    const int bits = ggml_get_op_params_i32(dst, 2);
    const float delta = ggml_get_op_params_f32(dst, 3), clip = ggml_get_op_params_f32(dst, 4);
    const bool learned = ggml_get_op_params_i32(dst, 5);
    const int64_t words = (k - 1)/32 + 1;
    const int64_t m = weights->ne[1];
    const int64_t n = acts->ne[1];
    GGML_ASSERT(k > 0 && m > 0 && n > 0);
    GGML_ASSERT(weights->type == GGML_TYPE_I32 && scales->type == GGML_TYPE_F32);
    GGML_ASSERT(acts->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const bool grouped = scales->ne[0] != m || scales->ne[1] != 1;
    GGML_ASSERT(weights->ne[0] == words && scales->ne[0] == (grouped ? (k + 127)/128 : m) && acts->ne[0] == k);
    GGML_ASSERT(!grouped || bits == 16);
    GGML_ASSERT(bits == 1 || bits == 4 || bits == 8 || bits == 16);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(scales));
    GGML_ASSERT(ggml_is_contiguous(acts) && ggml_is_contiguous(dst));
    GGML_ASSERT((m - 1)/4 + 1 <= INT_MAX);

    static std::atomic<unsigned> logged{0};
    const char * a4_kernel = getenv("GGML_W1AX_A4_KERNEL");
    GGML_ASSERT(bits != 4 || !a4_kernel || !*a4_kernel ||
            strcmp(a4_kernel, "bitserial") == 0 || strcmp(a4_kernel, "conventional") == 0);
    const bool bitserial = bits == 4 && (!a4_kernel || strcmp(a4_kernel, "conventional") != 0);
    const char * warp_value = getenv("GGML_W1AX_WARP_REDUCE");
    bool warp_reduce = warp_value && strcmp(warp_value, "1") == 0;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    warp_reduce = false;
#endif
    if (warp_reduce && bits != 16) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true)) GGML_LOG_INFO("CUDA W1Ax opt-in warp32 integer reduction dispatch\n");
    }
    const unsigned flag = bits == 1 ? 1u : bits == 4 ? (bitserial ? 2u : 4u) : bits == 8 ? 8u : 16u;
    if (!(logged.fetch_or(flag) & flag)) {
        const char * marker = bits == 1 ? "CUDA packed W1A1 XOR/POPCOUNT dispatch" :
            bits == 4 ? (bitserial ? "CUDA packed W1A4 BITSERIAL dispatch" : "CUDA packed W1A4 CONVENTIONAL dispatch") :
            bits == 8 ? "CUDA packed W1A8 INT8 dispatch" : "CUDA packed W1A16 FP16 SIGNADD dispatch";
        GGML_LOG_INFO("%s: %s (K=%lld, rows=%lld, tokens=%lld)\n", __func__, marker,
                (long long) k, (long long) m, (long long) n);
    }

    const cudaStream_t stream = ctx.stream();
    const unsigned token_blocks = (unsigned) (n < 65535 ? n : 65535);
    GGML_ASSERT(n <= 65535);
    w1ax_capture_activations(stream, weights, acts, k, m, n, bits);
    const ggml_tensor * shared = dst->src[3];
    const float * midpoints = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    const struct ggml_w1ax_pack_layout layout = midpoints ? ggml_w1ax_affine_layout(k,n,bits) : ggml_w1ax_pack_layout(k,n,bits);
    const float * code_sums = midpoints ? (const float *) ((const uint32_t *) shared->data + layout.scale_offset+n) : nullptr;
    if (bits == 16 && grouped) {
        ggml_cuda_matmul_audit(ctx, weights, acts, "custom_A16_group128", "F16", "F32_signadd_and_scales", "inside_kernel_F16_round_no_separate_pack");
        ggml_cuda_eagle_scope eagle_dot_scope(ctx, "dot_output_A16_inside_cast", dst);
        w1a16_group128_signadd<<<dim3((unsigned) ((m + 127)/128), token_blocks), 128, 0, stream>>>(
                (const uint32_t *) weights->data, (const float *) scales->data,
                (const float *) acts->data, m, k, words, (float *) dst->data);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (bits == 16) {
        ggml_cuda_matmul_audit(ctx, weights, acts, "custom_A16_row", "F16", "F32_signadd_and_scales", "inside_kernel_F16_round_no_separate_pack");
        ggml_cuda_eagle_scope eagle_dot_scope(ctx, "dot_output_A16_inside_cast", dst);
        w1a16_signadd<<<dim3((unsigned) ((m + 127)/128), token_blocks), 128, 0, stream>>>(
                (const uint32_t *) weights->data, (const float *) scales->data,
                (const float *) acts->data, m, k, words, (float *) dst->data, midpoints, code_sums);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    ggml_cuda_pool_alloc<float> act_scales(ctx.pool());
    const float * scale_ptr = shared ? (const float *) ((const uint32_t *) shared->data + layout.scale_offset) : act_scales.alloc(n);
    if (bits == 4 || bits == 8) {
        static const bool check_integer_dots = getenv("GGML_W1AX_ASSERT_INT_DOT") &&
            strcmp(getenv("GGML_W1AX_ASSERT_INT_DOT"), "0") != 0;
        ggml_cuda_pool_alloc<int8_t> codes(ctx.pool());
        ggml_cuda_pool_alloc<uint32_t> planes(ctx.pool());
        const int8_t * code_ptr = shared ? (const int8_t *) shared->data : codes.alloc((size_t) n*k);
        const uint32_t * plane_ptr = shared ? (const uint32_t *) shared->data + layout.codes_words : planes.alloc(bits == 4 ? (size_t) n*words*4 : 1);
        ggml_cuda_pool_alloc<int32_t> raw_dots(ctx.pool(), check_integer_dots ? (size_t) n*m : 1);
        if (!shared) {
            ggml_cuda_eagle_scope eagle_pack_scope(ctx, "activation_pack_scales", dst);
            w1ax_quantize<<<dim3(token_blocks), 256, 0, stream>>>(
                    (const float *) acts->data, k, words, bits, codes.ptr, planes.ptr, act_scales.ptr, clip, learned);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            ggml_cuda_eagle_scope eagle_dot_scope(ctx, "dot_output", dst);
            w1ax_integer_dot<<<dim3((unsigned) ((m - 1)/4 + 1), token_blocks), dim3(32, 4), 0, stream>>>(
                    (const uint32_t *) weights->data, (const float *) scales->data,
                    code_ptr, plane_ptr, scale_ptr, m, k, words, bits, bitserial, (float *) dst->data,
                    check_integer_dots ? raw_dots.ptr : nullptr, warp_reduce, midpoints, code_sums);
            CUDA_CHECK(cudaGetLastError());
        }
        if (check_integer_dots) {
            w1ax_validate_integer_dots<<<dim3((unsigned) ((m*n + 127)/128)), 128, 0, stream>>>(
                    (const uint32_t *) weights->data, code_ptr, raw_dots.ptr, m, n, k, words);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        return;
    }
    GGML_ASSERT(bits == 1);
    ggml_cuda_pool_alloc<uint32_t> packed(ctx.pool());
    const uint32_t * packed_ptr = shared ? (const uint32_t *) shared->data : packed.alloc((size_t) n*words);
    if (!shared) {
        ggml_cuda_eagle_scope eagle_pack_scope(ctx, "activation_pack_scales", dst);
        w1a1_pack_activations<<<dim3(token_blocks), 256, 0, stream>>>(
                (const float *) acts->data, k, words, n, packed.ptr, act_scales.ptr, delta);
        CUDA_CHECK(cudaGetLastError());
    }
    ggml_cuda_eagle_scope eagle_dot_scope(ctx, "dot_output", dst);
    w1a1_xor_popc<<<dim3((unsigned) ((m - 1)/4 + 1), token_blocks), dim3(32, 4), 0, stream>>>(
            (const uint32_t *) weights->data, (const float *) scales->data,
            packed_ptr, scale_ptr, m, n, k, words, (float *) dst->data, warp_reduce, midpoints, code_sums);
    CUDA_CHECK(cudaGetLastError());
}

// Only EAGLE graph builders assign this marker; target linears remain untouched.
void ggml_cuda_eagle_capture_dense(ggml_backend_cuda_context & ctx,
        const ggml_tensor * weights, const ggml_tensor * acts) {
    if (strncmp(acts->name, "eagle_capture_", 14) != 0) return;
    GGML_ASSERT(acts->type == GGML_TYPE_F32 && ggml_is_contiguous(acts));
    GGML_ASSERT(acts->ne[2] == 1 && acts->ne[3] == 1);
    w1ax_capture_activations(ctx.stream(), weights, acts, acts->ne[0], weights->ne[1], acts->ne[1],
            strncmp(acts->name, "eagle_capture_a16_", 18) == 0 ? 16 : 32);
}
