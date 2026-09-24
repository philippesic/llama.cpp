#include "w1a1.cuh"

#include <climits>
#include <cstring>

// Each lane owns a complete little-bit-order sign word. Accumulate absolute
// values in double, as the CPU implementation does, before rounding the row
// mean to F32. The unused tail bits remain zero.
static __global__ void w1a1_pack_activations(
        const float * activations, int64_t k, int64_t words, int64_t n,
        uint32_t * packed, float * scales) {
    __shared__ double sums[256];
    for (int64_t token = blockIdx.x; token < n; token += gridDim.x) {
        double sum = 0.0;
        const float * row = activations + token*k;
        for (int64_t word = threadIdx.x; word < words; word += blockDim.x) {
            uint32_t bits = 0;
            const int64_t first = word*32;
            const int count = (int) (k - first < 32 ? k - first : 32);
            for (int bit = 0; bit < count; ++bit) {
                const float value = row[first + bit];
                const uint32_t raw = __float_as_uint(value);
                // sign(0) is +1, including negative zero. Read the sign bit
                // directly so fast-math cannot flush negative subnormals.
                const bool positive = (raw & 0x80000000u) == 0 || (raw & 0x7fffffffu) == 0;
                bits |= (uint32_t) positive << bit;
                sum += (double) fabsf(value);
            }
            packed[token*words + word] = bits;
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) scales[token] = (float) (sums[0] / (double) k);
        __syncthreads();
    }
}

// Four rows per block, one group of 32 lanes per row. Shared-memory reduction
// keeps the portable POPC path usable by all variants of ggml-cuda.
static __global__ void w1a1_xor_popc(
        const uint32_t * weights, const float * weight_scales,
        const uint32_t * activations, const float * activation_scales,
        int64_t m, int64_t n, int64_t k, int64_t words, float * output) {
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
        counts[threadIdx.y][lane] = mismatches;
        __syncthreads();
        for (int stride = 16; stride > 0; stride /= 2) {
            if (lane < stride) counts[threadIdx.y][lane] += counts[threadIdx.y][lane + stride];
            __syncthreads();
        }
        if (valid_row && lane == 0) {
            const float dot = (float) (k - 2*(int64_t) counts[threadIdx.y][0]);
            const float weighted = dot*weight_scales[row];
            output[token*m + row] = weighted*activation_scales[token];
        }
        __syncthreads();
    }
}

void ggml_cuda_w1a1_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * scales  = dst->src[1];
    const ggml_tensor * acts    = dst->src[2];
    int64_t k;
    memcpy(&k, dst->op_params, sizeof(k));
    const int64_t words = (k - 1)/32 + 1;
    const int64_t m = weights->ne[1];
    const int64_t n = acts->ne[1];
    GGML_ASSERT(k > 0 && m > 0 && n > 0);
    GGML_ASSERT(weights->type == GGML_TYPE_I32 && scales->type == GGML_TYPE_F32);
    GGML_ASSERT(acts->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->ne[0] == words && scales->ne[0] == m && acts->ne[0] == k);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(scales));
    GGML_ASSERT(ggml_is_contiguous(acts) && ggml_is_contiguous(dst));
    GGML_ASSERT((m - 1)/4 + 1 <= INT_MAX);

    ggml_cuda_pool_alloc<uint32_t> packed(ctx.pool(), (size_t) n*words);
    ggml_cuda_pool_alloc<float> act_scales(ctx.pool(), n);
    const cudaStream_t stream = ctx.stream();
    const unsigned token_blocks = (unsigned) (n < 65535 ? n : 65535);
    w1a1_pack_activations<<<dim3(token_blocks), 256, 0, stream>>>(
            (const float *) acts->data, k, words, n, packed.ptr, act_scales.ptr);
    CUDA_CHECK(cudaGetLastError());
    w1a1_xor_popc<<<dim3((unsigned) ((m - 1)/4 + 1), token_blocks), dim3(32, 4), 0, stream>>>(
            (const uint32_t *) weights->data, (const float *) scales->data,
            packed.ptr, act_scales.ptr, m, n, k, words, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}
