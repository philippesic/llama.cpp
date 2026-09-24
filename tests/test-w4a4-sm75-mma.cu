// Standalone, opt-in layout probe. Build on a CUDA host with:
// nvcc -std=c++17 -arch=sm_75 tests/test-w4a4-sm75-mma.cu -o test-w4a4-sm75-mma
// The production W4A4 CUDA dispatch does not call this kernel.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static __device__ __forceinline__ uint32_t nibble_at(
        const uint8_t * packed, int row, int k_index, int rows, int k) {
    if (row >= rows || k_index >= k) return 0;
    const uint8_t byte = packed[(size_t) row * ((k + 1)/2) + k_index/2];
    return (byte >> (4 * (k_index & 1))) & 0x0f;
}

static __global__ void signed_i4_mma_probe(
        const uint8_t * weights, const uint8_t * activations,
        int32_t * output, int m, int n, int k) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int row = blockIdx.x * 8 + group;
    const int token_for_b = blockIdx.y * 8 + group;
    int d0 = 0;
    int d1 = 0;
    for (int k_base = 0; k_base < k; k_base += 32) {
        uint32_t a = 0;
        uint32_t b = 0;
        for (int i = 0; i < 8; ++i) {
            const int ki = k_base + thread_in_group * 8 + i;
            a |= nibble_at(weights, row, ki, m, k) << (4 * i);
            b |= nibble_at(activations, token_for_b, ki, n, k) << (4 * i);
        }
        const int c0 = d0;
        const int c1 = d1;
        asm volatile(
            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 "
            "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
            : "=r"(d0), "=r"(d1)
            : "r"(a), "r"(b), "r"(c0), "r"(c1));
    }
    const int token0 = blockIdx.y * 8 + 2 * thread_in_group;
    if (row < m && token0 < n) output[token0 * m + row] = d0;
    if (row < m && token0 + 1 < n) output[(token0 + 1) * m + row] = d1;
}

static void require_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

static void pack_code(std::vector<uint8_t> & packed, int row, int i, int k, int code) {
    packed[(size_t) row * ((k + 1)/2) + i/2] |= (uint8_t) (code & 0x0f) << (4 * (i & 1));
}

static bool run_case(int m, int n, int k, bool basis) {
    const size_t packed_bytes = (size_t) (k + 1)/2;
    std::vector<uint8_t> weights((size_t) m * packed_bytes, 0);
    std::vector<uint8_t> activations((size_t) n * packed_bytes, 0);
    std::vector<int8_t> weight_codes((size_t) m * k);
    std::vector<int8_t> activation_codes((size_t) n * k);
    for (int row = 0; row < m; ++row) {
        for (int i = 0; i < k; ++i) {
            const int code = basis ? (i == row ? (row & 1 ? -7 : 7) : 0) : ((row * 11 + i * 3) % 15) - 7;
            weight_codes[(size_t) row * k + i] = (int8_t) code;
            pack_code(weights, row, i, k, code);
        }
    }
    for (int token = 0; token < n; ++token) {
        for (int i = 0; i < k; ++i) {
            const int code = basis ? (i == (token + 1) % k ? (token & 1 ? -7 : 7) : 0) :
                (token == n - 1 ? 0 : ((token * 5 + i * 7) % 15) - 7);
            activation_codes[(size_t) token * k + i] = (int8_t) code;
            pack_code(activations, token, i, k, code);
        }
    }
    std::vector<int32_t> expected((size_t) m * n, 0);
    for (int token = 0; token < n; ++token) {
        for (int row = 0; row < m; ++row) {
            int32_t dot = 0;
            for (int i = 0; i < k; ++i) {
                dot += (int32_t) weight_codes[(size_t) row * k + i] *
                       (int32_t) activation_codes[(size_t) token * k + i];
            }
            expected[(size_t) token * m + row] = dot;
        }
    }

    uint8_t * d_weights = nullptr;
    uint8_t * d_activations = nullptr;
    int32_t * d_output = nullptr;
    require_cuda(cudaMalloc(&d_weights, weights.size()), "cudaMalloc weights");
    require_cuda(cudaMalloc(&d_activations, activations.size()), "cudaMalloc activations");
    require_cuda(cudaMalloc(&d_output, expected.size() * sizeof(int32_t)), "cudaMalloc output");
    require_cuda(cudaMemcpy(d_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice), "copy weights");
    require_cuda(cudaMemcpy(d_activations, activations.data(), activations.size(), cudaMemcpyHostToDevice), "copy activations");
    require_cuda(cudaMemset(d_output, 0x55, expected.size() * sizeof(int32_t)), "clear output");
    signed_i4_mma_probe<<<dim3((m + 7)/8, (n + 7)/8), 32>>>(d_weights, d_activations, d_output, m, n, k);
    require_cuda(cudaGetLastError(), "launch signed_i4_mma_probe");
    require_cuda(cudaDeviceSynchronize(), "synchronize signed_i4_mma_probe");
    std::vector<int32_t> actual(expected.size());
    require_cuda(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy output");
    require_cuda(cudaFree(d_weights), "free weights");
    require_cuda(cudaFree(d_activations), "free activations");
    require_cuda(cudaFree(d_output), "free output");

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL M=%d N=%d K=%d basis=%d output[%zu]: got %d, expected %d\n",
                    m, n, k, basis, i, actual[i], expected[i]);
            return false;
        }
    }
    std::printf("PASS SM75 signed s4 MMA M=%d N=%d K=%d basis=%d (%zu exact I32 outputs)\n",
            m, n, k, basis, expected.size());
    return true;
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;
    }
    cudaDeviceProp prop;
    require_cuda(cudaGetDeviceProperties(&prop, 0), "get device properties");
    if (prop.major != 7 || prop.minor != 5) {
        std::fprintf(stderr, "SKIP: requires SM75, found SM%d%d\n", prop.major, prop.minor);
        return 77;
    }
    return run_case(8, 8, 32, true) &&
           run_case(8, 8, 32, false) &&
           run_case(9, 1, 9, false) &&
           run_case(9, 9, 33, false) &&
           run_case(9, 1, 9728, false) &&
           run_case(9, 9, 9728, false) ? 0 : 1;
}
