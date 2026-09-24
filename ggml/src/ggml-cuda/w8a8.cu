#include "w8a8.cuh"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// GGML CUDA uses -use_fast_math. Explicit RN operations preserve the versioned
// W8A8 contract: F32 division before nearest-even conversion and two ordered
// F32 scale multiplications.
static __device__ __forceinline__ float w8a8_div_rn(float a, float b) {
    float result;
    asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __device__ __forceinline__ float w8a8_mul_rn(float a, float b) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __device__ __forceinline__ float w8a8_abs(float value) {
    float result;
    asm volatile("abs.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static __device__ __forceinline__ float w8a8_max(float a, float b) {
    float result;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __global__ void w8a8_pack_activations(
        const char * activations, size_t row_stride, int64_t k, int64_t n,
        int8_t * codes, float * scales) {
    __shared__ float maxima[256];
    for (int64_t token = blockIdx.x; token < n; token += gridDim.x) {
        const float * input = (const float *) (activations + token * row_stride);
        float max_abs = 0.0f;
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            const float value = input[i];
            if ((__float_as_uint(value) & 0x7f800000u) == 0x7f800000u) {
                asm volatile("trap;");
            }
            max_abs = w8a8_max(max_abs, w8a8_abs(value));
        }
        maxima[threadIdx.x] = max_abs;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) maxima[threadIdx.x] = w8a8_max(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            __syncthreads();
        }
        const float scale = w8a8_div_rn(maxima[0], 127.0f);
        if (threadIdx.x == 0) scales[token] = scale;
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            int code = 0;
            if (scale != 0.0f) {
                const float scaled = w8a8_div_rn(input[i], scale);
                code = __float2int_rn(scaled);
                code = max(-127, min(127, code));
            }
            codes[token * k + i] = (int8_t) code;
        }
        __syncthreads();
    }
}

static __device__ __forceinline__ int w8a8_pack_4(const int8_t * values, int64_t first, int64_t k) {
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t byte = first + i < k ? (uint8_t) values[first + i] : 0;
        packed |= byte << (8 * i);
    }
    return (int) packed;
}

// Four warps per block; each warp owns one output row and performs an exact
// signed-byte dot. This is a DP4A/SIMT path, not an INT8 Tensor Core claim.
static __global__ void w8a8_signed_dot(
        const int8_t * weights, const float * weight_scales,
        const int8_t * activations, const float * activation_scales,
        int64_t m, int64_t n, int64_t k, float * output) {
    const int64_t row = (int64_t) blockIdx.x * 4 + threadIdx.y;
    const int lane = threadIdx.x;
    for (int64_t token = blockIdx.y; token < n; token += gridDim.y) {
        int dot = 0;
        if (row < m) {
            const int8_t * weight = weights + row * k;
            const int8_t * act = activations + token * k;
            for (int64_t first = lane * 4; first < k; first += 128) {
                dot = ggml_cuda_dp4a(w8a8_pack_4(weight, first, k), w8a8_pack_4(act, first, k), dot);
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (row < m && lane == 0) {
            const float weight_scale = weight_scales[row];
            if ((__float_as_uint(weight_scale) & 0x7f800000u) == 0x7f800000u) {
                asm volatile("trap;");
            }
            const float weighted = w8a8_mul_rn((float) dot, weight_scale);
            output[token * m + row] = w8a8_mul_rn(weighted, activation_scales[token]);
        }
    }
}

void ggml_cuda_w8a8_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * scales  = dst->src[1];
    const ggml_tensor * acts    = dst->src[2];
    const int64_t k = weights->ne[0];
    const int64_t m = weights->ne[1];
    const int64_t n = acts->ne[1];
    GGML_ASSERT(k > 0 && m > 0 && n > 0 && k <= INT_MAX / (128 * 127));
    GGML_ASSERT(weights->type == GGML_TYPE_I8 && scales->type == GGML_TYPE_F32);
    GGML_ASSERT(acts->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scales->ne[0] == m && acts->ne[0] == k && acts->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(scales) && ggml_is_contiguous(dst));
    GGML_ASSERT((m - 1) / 4 + 1 <= INT_MAX);

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        GGML_LOG_INFO("%s: CUDA W8A8 signed INT8 dot/I32 accumulation dispatch (K=%lld, rows=%lld, tokens=%lld)\n",
                __func__, (long long) k, (long long) m, (long long) n);
    }

    ggml_cuda_pool_alloc<int8_t> quantized(ctx.pool(), (size_t) n * k);
    ggml_cuda_pool_alloc<float> act_scales(ctx.pool(), n);
    const cudaStream_t stream = ctx.stream();
    const unsigned token_blocks = (unsigned) (n < 65535 ? n : 65535);
    w8a8_pack_activations<<<dim3(token_blocks), 256, 0, stream>>>(
            (const char *) acts->data, acts->nb[1], k, n, quantized.ptr, act_scales.ptr);
    CUDA_CHECK(cudaGetLastError());
    w8a8_signed_dot<<<dim3((unsigned) ((m - 1) / 4 + 1), token_blocks), dim3(32, 4), 0, stream>>>(
            (const int8_t *) weights->data, (const float *) scales->data,
            quantized.ptr, act_scales.ptr, m, n, k, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}

#else

void ggml_cuda_w8a8_mul_mat(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("W8A8 CUDA operator requires an NVIDIA backend");
}

#endif
