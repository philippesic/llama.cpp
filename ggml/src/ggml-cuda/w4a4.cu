#include "w4a4.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// GGML CUDA compiles with -use_fast_math. Keep the versioned quantizer's F32
// division and output scale order explicit rather than allowing reciprocal
// approximation, FTZ, or multiply reassociation.
static __device__ __forceinline__ float w4a4_div_rn(float a, float b) {
    float result;
    asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __device__ __forceinline__ float w4a4_mul_rn(float a, float b) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __device__ __forceinline__ float w4a4_abs(float value) {
    float result;
    asm volatile("abs.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static __device__ __forceinline__ float w4a4_max(float a, float b) {
    float result;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

// One block per input token. The first pass computes its F32 absmax; the
// second packs two signed [-7, 7] codes into each byte. An odd tail gets a
// zero high nibble. __float2int_rn implements nearest-even rounding.
static __global__ void w4a4_pack_activations(
        const float * activations, int64_t token_stride_bytes, int64_t k,
        int64_t packed_k, int64_t n, uint8_t * packed, float * scales) {
    __shared__ float maxima[256];
    __shared__ float token_scale;
    for (int64_t token = blockIdx.x; token < n; token += gridDim.x) {
        const float * row = (const float *) ((const char *) activations + token * token_stride_bytes);
        float absmax = 0.0f;
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            const float value = row[i];
            if ((__float_as_uint(value) & 0x7f800000u) == 0x7f800000u) asm volatile("trap;");
            absmax = w4a4_max(absmax, w4a4_abs(value));
        }
        maxima[threadIdx.x] = absmax;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) maxima[threadIdx.x] = w4a4_max(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            token_scale = w4a4_div_rn(maxima[0], 7.0f);
            scales[token] = token_scale;
        }
        __syncthreads();
        const float scale = token_scale;
        for (int64_t byte = threadIdx.x; byte < packed_k; byte += blockDim.x) {
            uint8_t lo = 0;
            uint8_t hi = 0;
            if (scale != 0.0f) {
                const int q0 = __float2int_rn(w4a4_div_rn(row[2*byte], scale));
                lo = (uint8_t) (max(-7, min(7, q0)) & 0x0f);
                if (2*byte + 1 < k) {
                    const int q1 = __float2int_rn(w4a4_div_rn(row[2*byte + 1], scale));
                    hi = (uint8_t) (max(-7, min(7, q1)) & 0x0f);
                }
            }
            packed[token*packed_k + byte] = lo | (hi << 4);
        }
        __syncthreads();
    }
}

static __device__ __forceinline__ int w4a4_signed_nibble(uint8_t nibble) {
    return nibble < 8 ? (int) nibble : (int) nibble - 16;
}

// Portable signed-nibble vector dot. Every lane multiplies values decoded
// from packed I4 bytes; this kernel does not issue INT4 Tensor Core MMA.
static __global__ void w4a4_vector_dot(
        const uint8_t * weights, const float * weight_scales,
        const uint8_t * activations, const float * activation_scales,
        int64_t m, int64_t n, int64_t k, int64_t packed_k, float * output) {
    __shared__ int32_t partial[4][32];
    const int64_t row = (int64_t) blockIdx.x * 4 + threadIdx.y;
    const bool valid_row = row < m;
    const int lane = threadIdx.x;
    for (int64_t token = blockIdx.y; token < n; token += gridDim.y) {
        int32_t dot = 0;
        if (valid_row) {
            const uint8_t * weight = weights + row * packed_k;
            const uint8_t * act = activations + token * packed_k;
            for (int64_t byte = lane; byte < packed_k; byte += 32) {
                const uint8_t w = weight[byte];
                const uint8_t a = act[byte];
                dot += w4a4_signed_nibble(w & 0x0f) * w4a4_signed_nibble(a & 0x0f);
                if (2*byte + 1 < k) {
                    dot += w4a4_signed_nibble(w >> 4) * w4a4_signed_nibble(a >> 4);
                }
            }
        }
        partial[threadIdx.y][lane] = dot;
        __syncthreads();
        for (int stride = 16; stride > 0; stride /= 2) {
            if (lane < stride) partial[threadIdx.y][lane] += partial[threadIdx.y][lane + stride];
            __syncthreads();
        }
        if (valid_row && lane == 0) {
            const float weight_scale = weight_scales[row];
            if ((__float_as_uint(weight_scale) & 0x7f800000u) == 0x7f800000u) asm volatile("trap;");
            const float weighted = w4a4_mul_rn((float) partial[threadIdx.y][0], weight_scale);
            output[token*m + row] = w4a4_mul_rn(weighted, activation_scales[token]);
        }
        __syncthreads();
    }
}

static __device__ __forceinline__ uint32_t w4a4_packed_nibble(
        const uint8_t * packed, int64_t row, int64_t k_index,
        int64_t rows, int64_t k, int64_t packed_k) {
    if (row >= rows || k_index >= k) return 0;
    const uint8_t byte = packed[row * packed_k + k_index/2];
    return (byte >> (4 * (k_index & 1))) & 0x0f;
}

// SM75 signed-I4 Tensor Core candidate. The per-lane A/B and D fragments are
// exactly those checked by test-w4a4-sm75-mma: group=lane>>2, t=lane&3;
// A=(row=group, K=8*t+i), B=(token=group, K=8*t+i), D=(row=group,
// token=2*t+j). All lanes execute each MMA, including masked M/N/K tails.
// Byte assembly is safe for odd K and unaligned packed row starts.
static __global__ void w4a4_sm75_mma_dot(
        const uint8_t * weights, const float * weight_scales,
        const uint8_t * activations, const float * activation_scales,
        int64_t m, int64_t n, int64_t k, int64_t packed_k, float * output) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int64_t row = (int64_t) blockIdx.x * 8 + group;
    for (int64_t token_base = (int64_t) blockIdx.y * 8; token_base < n; token_base += (int64_t) gridDim.y * 8) {
        const int64_t token_for_b = token_base + group;
        int d0 = 0;
        int d1 = 0;
        for (int64_t k_base = 0; k_base < k; k_base += 32) {
            uint32_t a = 0;
            uint32_t b = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t ki = k_base + thread_in_group * 8 + i;
                a |= w4a4_packed_nibble(weights, row, ki, m, k, packed_k) << (4 * i);
                b |= w4a4_packed_nibble(activations, token_for_b, ki, n, k, packed_k) << (4 * i);
            }
#if __CUDA_ARCH__ >= 750
            const int c0 = d0;
            const int c1 = d1;
            asm volatile(
                "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                : "=r"(d0), "=r"(d1)
                : "r"(a), "r"(b), "r"(c0), "r"(c1));
#else
            // The host selector never launches this kernel below SM75.
            asm volatile("trap;");
            GGML_UNUSED(a);
            GGML_UNUSED(b);
#endif
        }
        const int64_t token0 = token_base + 2 * thread_in_group;
        if (row < m && token0 < n) {
            const float weight_scale = weight_scales[row];
            if ((__float_as_uint(weight_scale) & 0x7f800000u) == 0x7f800000u) asm volatile("trap;");
            const float weighted = w4a4_mul_rn((float) d0, weight_scale);
            output[token0 * m + row] = w4a4_mul_rn(weighted, activation_scales[token0]);
        }
        if (row < m && token0 + 1 < n) {
            const float weight_scale = weight_scales[row];
            if ((__float_as_uint(weight_scale) & 0x7f800000u) == 0x7f800000u) asm volatile("trap;");
            const float weighted = w4a4_mul_rn((float) d1, weight_scale);
            output[(token0 + 1) * m + row] = w4a4_mul_rn(weighted, activation_scales[token0 + 1]);
        }
    }
}

void ggml_cuda_w4a4_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * scales  = dst->src[1];
    const ggml_tensor * acts    = dst->src[2];
    const int64_t k = acts->ne[0];
    const int64_t packed_k = (k + 1)/2;
    const int64_t m = weights->ne[1];
    const int64_t n = acts->ne[1];
    GGML_ASSERT(k > 0 && k <= INT32_MAX/49 && m > 0 && n > 0);
    GGML_ASSERT(weights->type == GGML_TYPE_I8 && scales->type == GGML_TYPE_F32);
    GGML_ASSERT(acts->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->ne[0] == packed_k && scales->ne[0] == m);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(scales));
    GGML_ASSERT(acts->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
    GGML_ASSERT((m + 3)/4 <= INT_MAX);

    static const bool use_mma = [] {
        const char * value = std::getenv("GGML_CUDA_W4A4_MMA");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (use_mma) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (cc != GGML_CUDA_CC_TURING) {
            GGML_ABORT("GGML_CUDA_W4A4_MMA=1 requires an SM75 NVIDIA device");
        }
    }

    static std::atomic<unsigned> logged_vector_shapes{0};
    static std::atomic<unsigned> logged_mma_shapes{0};
    const unsigned shape_bit = n == 1 ? 1u : 2u;
    if (use_mma) {
        if ((logged_mma_shapes.fetch_or(shape_bit) & shape_bit) == 0) {
            GGML_LOG_INFO("%s: CUDA W4A4 SM75 signed-I4 Tensor Core MMA m8n8k32 candidate (K=%lld, rows=%lld, tokens=%lld)\n",
                    __func__, (long long) k, (long long) m, (long long) n);
        }
    } else if ((logged_vector_shapes.fetch_or(shape_bit) & shape_bit) == 0) {
        GGML_LOG_INFO("%s: CUDA W4A4 signed-nibble vector dot, scalar integer MUL/ADD; no INT4 Tensor Core MMA (K=%lld, rows=%lld, tokens=%lld)\n",
                __func__, (long long) k, (long long) m, (long long) n);
    }

    ggml_cuda_pool_alloc<uint8_t> packed(ctx.pool(), (size_t) n * packed_k);
    ggml_cuda_pool_alloc<float> act_scales(ctx.pool(), n);
    const cudaStream_t stream = ctx.stream();
    const unsigned token_blocks = (unsigned) (n < 65535 ? n : 65535);
    w4a4_pack_activations<<<dim3(token_blocks), 256, 0, stream>>>(
            (const float *) acts->data, acts->nb[1], k, packed_k, n, packed.ptr, act_scales.ptr);
    CUDA_CHECK(cudaGetLastError());
    if (use_mma) {
        const unsigned mma_token_blocks = (unsigned) ((n + 7)/8 < 65535 ? (n + 7)/8 : 65535);
        w4a4_sm75_mma_dot<<<dim3((unsigned) ((m + 7)/8), mma_token_blocks), 32, 0, stream>>>(
                (const uint8_t *) weights->data, (const float *) scales->data,
                packed.ptr, act_scales.ptr, m, n, k, packed_k, (float *) dst->data);
    } else {
        w4a4_vector_dot<<<dim3((unsigned) ((m + 3)/4), token_blocks), dim3(32, 4), 0, stream>>>(
                (const uint8_t *) weights->data, (const float *) scales->data,
                packed.ptr, act_scales.ptr, m, n, k, packed_k, (float *) dst->data);
    }
    CUDA_CHECK(cudaGetLastError());
}

#else

void ggml_cuda_w4a4_mul_mat(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("W4A4 CUDA operator requires an NVIDIA backend");
}

#endif
