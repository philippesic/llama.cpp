#include "w4a4.cuh"

#include <atomic>
#include <climits>

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
            if (!isfinite(value)) asm volatile("trap;");
            absmax = fmaxf(absmax, fabsf(value));
        }
        maxima[threadIdx.x] = absmax;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            token_scale = maxima[0] / 7.0f;
            scales[token] = token_scale;
        }
        __syncthreads();
        const float scale = token_scale;
        for (int64_t byte = threadIdx.x; byte < packed_k; byte += blockDim.x) {
            uint8_t lo = 0;
            uint8_t hi = 0;
            if (scale != 0.0f) {
                const int q0 = __float2int_rn(__fdiv_rn(row[2*byte], scale));
                lo = (uint8_t) (max(-7, min(7, q0)) & 0x0f);
                if (2*byte + 1 < k) {
                    const int q1 = __float2int_rn(__fdiv_rn(row[2*byte + 1], scale));
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
            const float weighted = (float) partial[threadIdx.y][0] * weight_scales[row];
            output[token*m + row] = weighted * activation_scales[token];
        }
        __syncthreads();
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

    static std::atomic<unsigned> logged_shapes{0};
    const unsigned shape_bit = n == 1 ? 1u : 2u;
    if ((logged_shapes.fetch_or(shape_bit) & shape_bit) == 0) {
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
    w4a4_vector_dot<<<dim3((unsigned) ((m + 3)/4), token_blocks), dim3(32, 4), 0, stream>>>(
            (const uint8_t *) weights->data, (const float *) scales->data,
            packed.ptr, act_scales.ptr, m, n, k, packed_k, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}
