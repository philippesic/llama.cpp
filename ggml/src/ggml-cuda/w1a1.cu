#include "w1a1.cuh"

#include <atomic>
#include <climits>
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

// One block per token. Quantization is per complete logical token, with F32
// absmax/scale and round-to-nearest-even codes. Zero vectors produce zero codes.
// A4 planes contain two's-complement bits; unused tail bits are always clear.
static __global__ void w1ax_quantize(
        const float * activations, int64_t k, int64_t words, int bits,
        int8_t * codes, uint32_t * planes, float * scales) {
    __shared__ float maxima[256];
    __shared__ float token_scale;
    const int64_t token = blockIdx.x;
    const float * row = activations + token*k;
    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) local_max = fmaxf(local_max, fabsf(row[i]));
    maxima[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        token_scale = __fdiv_rn(maxima[0], bits == 8 ? 127.0f : 7.0f);
        scales[token] = token_scale;
    }
    __syncthreads();
    const int qmax = bits == 8 ? 127 : 7;
    const float inv = maxima[0] == 0.0f ? 0.0f : __fdiv_rn((float) qmax, maxima[0]);
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        const int q = __float2int_rn(__fmul_rn(row[i], inv));
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
        int64_t m, int64_t k, int64_t words, int bits, bool bitserial, float * output, int32_t * raw_dots) {
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
    sums[threadIdx.y][lane] = sum;
    __syncthreads();
    for (int stride = 16; stride > 0; stride /= 2) {
        if (lane < stride) sums[threadIdx.y][lane] += sums[threadIdx.y][lane + stride];
        __syncthreads();
    }
    if (row < m && lane == 0) {
        if (raw_dots) raw_dots[token*m + row] = sums[threadIdx.y][0];
        const float weighted = (float) sums[threadIdx.y][0] * weight_scales[row];
        output[token*m + row] = weighted * act_scales[token];
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
        int64_t m, int64_t k, int64_t words, float * output) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t token = blockIdx.y;
    if (row >= m) return;
    float sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        const float value = __half2float(__float2half_rn(activations[token*k + i]));
        sum += (weights[row*words + i/32] & (1u << (i%32))) ? value : -value;
    }
    output[token*m + row] = sum * weight_scales[row];
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

void ggml_cuda_w1a1_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * scales  = dst->src[1];
    const ggml_tensor * acts    = dst->src[2];
    int64_t k;
    memcpy(&k, dst->op_params, sizeof(k));
    const int bits = ggml_get_op_params_i32(dst, 2);
    const int64_t words = (k - 1)/32 + 1;
    const int64_t m = weights->ne[1];
    const int64_t n = acts->ne[1];
    GGML_ASSERT(k > 0 && m > 0 && n > 0);
    GGML_ASSERT(weights->type == GGML_TYPE_I32 && scales->type == GGML_TYPE_F32);
    GGML_ASSERT(acts->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->ne[0] == words && scales->ne[0] == m && acts->ne[0] == k);
    GGML_ASSERT(bits == 1 || bits == 4 || bits == 8 || bits == 16);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(scales));
    GGML_ASSERT(ggml_is_contiguous(acts) && ggml_is_contiguous(dst));
    GGML_ASSERT((m - 1)/4 + 1 <= INT_MAX);

    static std::atomic<unsigned> logged{0};
    const char * a4_kernel = getenv("GGML_W1AX_A4_KERNEL");
    GGML_ASSERT(bits != 4 || !a4_kernel || !*a4_kernel ||
            strcmp(a4_kernel, "bitserial") == 0 || strcmp(a4_kernel, "conventional") == 0);
    const bool bitserial = bits == 4 && (!a4_kernel || strcmp(a4_kernel, "conventional") != 0);
    const unsigned flag = bits == 1 ? 1u : bits == 4 ? (bitserial ? 2u : 4u) : bits == 8 ? 8u : 16u;
    if (!(logged.fetch_or(flag) & flag)) {
        const char * marker = bits == 1 ? "CUDA packed W1A1 XOR/POPCOUNT dispatch" :
            bits == 4 ? (bitserial ? "CUDA packed W1A4 BITSERIAL dispatch" : "CUDA packed W1A4 CONVENTIONAL dispatch") :
            bits == 8 ? "CUDA packed W1A8 INT8 dispatch" : "CUDA packed W1A16 FP16 SIGNADD dispatch";
        GGML_LOG_INFO("%s: %s (K=%lld, rows=%lld, tokens=%lld)\n", __func__, marker,
                (long long) k, (long long) m, (long long) n);
    }

    ggml_cuda_pool_alloc<uint32_t> packed(ctx.pool(), (size_t) n*words);
    ggml_cuda_pool_alloc<float> act_scales(ctx.pool(), n);
    const cudaStream_t stream = ctx.stream();
    const unsigned token_blocks = (unsigned) (n < 65535 ? n : 65535);
    GGML_ASSERT(n <= 65535);
    w1ax_capture_activations(stream, weights, acts, k, m, n, bits);
    if (bits == 16) {
        w1a16_signadd<<<dim3((unsigned) ((m + 127)/128), token_blocks), 128, 0, stream>>>(
                (const uint32_t *) weights->data, (const float *) scales->data,
                (const float *) acts->data, m, k, words, (float *) dst->data);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (bits == 4 || bits == 8) {
        static const bool check_integer_dots = getenv("GGML_W1AX_ASSERT_INT_DOT") &&
            strcmp(getenv("GGML_W1AX_ASSERT_INT_DOT"), "0") != 0;
        ggml_cuda_pool_alloc<int8_t> codes(ctx.pool(), (size_t) n*k);
        ggml_cuda_pool_alloc<uint32_t> planes(ctx.pool(), bits == 4 ? (size_t) n*words*4 : 1);
        ggml_cuda_pool_alloc<int32_t> raw_dots(ctx.pool(), check_integer_dots ? (size_t) n*m : 1);
        w1ax_quantize<<<dim3(token_blocks), 256, 0, stream>>>(
                (const float *) acts->data, k, words, bits, codes.ptr, planes.ptr, act_scales.ptr);
        CUDA_CHECK(cudaGetLastError());
        w1ax_integer_dot<<<dim3((unsigned) ((m - 1)/4 + 1), token_blocks), dim3(32, 4), 0, stream>>>(
                (const uint32_t *) weights->data, (const float *) scales->data,
                codes.ptr, planes.ptr, act_scales.ptr, m, k, words, bits, bitserial, (float *) dst->data,
                check_integer_dots ? raw_dots.ptr : nullptr);
        CUDA_CHECK(cudaGetLastError());
        if (check_integer_dots) {
            w1ax_validate_integer_dots<<<dim3((unsigned) ((m*n + 127)/128)), 128, 0, stream>>>(
                    (const uint32_t *) weights->data, codes.ptr, raw_dots.ptr, m, n, k, words);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        return;
    }
    GGML_ASSERT(bits == 1);
    w1a1_pack_activations<<<dim3(token_blocks), 256, 0, stream>>>(
            (const float *) acts->data, k, words, n, packed.ptr, act_scales.ptr);
    CUDA_CHECK(cudaGetLastError());
    w1a1_xor_popc<<<dim3((unsigned) ((m - 1)/4 + 1), token_blocks), dim3(32, 4), 0, stream>>>(
            (const uint32_t *) weights->data, (const float *) scales->data,
            packed.ptr, act_scales.ptr, m, n, k, words, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}
