// Standalone exact-I32 lane-layout check for the SM75 W8A8 MMA candidate.
// nvcc -std=c++17 -arch=sm_75 -O2 tools/w8a8-mma-probe.cu -o w8a8-mma-probe
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

__device__ int pack4(const int8_t * values, int64_t first, int k) {
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (first + i < k) packed |= (uint32_t) (uint8_t) values[first + i] << (8 * i);
    }
    return (int) packed;
}

__global__ void probe(const int8_t * weights, const int8_t * acts, int m, int n, int k, int32_t * dots) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x * 8 + lane / 4;
    const int operand_token = blockIdx.y * 8 + lane / 4;
    int d0 = 0;
    int d1 = 0;
    for (int first = 0; first < k; first += 16) {
        const int offset = first + (lane % 4) * 4;
        const int a = row < m ? pack4(weights + (int64_t) row * k, offset, k) : 0;
        const int b = operand_token < n ? pack4(acts + (int64_t) operand_token * k, offset, k) : 0;
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
    }
    const int token0 = blockIdx.y * 8 + 2 * (lane % 4);
    if (row < m && token0 < n) dots[token0 * m + row] = d0;
    if (row < m && token0 + 1 < n) dots[(token0 + 1) * m + row] = d1;
}

static int8_t code(int row, int i, int salt) {
    const int value = (row * 53 + i * 29 + salt) % 255 - 127;
    return (int8_t) value;
}

static void run(int m, int n, int k) {
    std::vector<int8_t> weights((size_t) m * k);
    std::vector<int8_t> acts((size_t) n * k);
    std::vector<int32_t> got((size_t) m * n, INT32_MIN);
    for (int row = 0; row < m; ++row) {
        for (int i = 0; i < k; ++i) weights[(size_t) row * k + i] = code(row, i, 0);
    }
    for (int token = 0; token < n; ++token) {
        for (int i = 0; i < k; ++i) acts[(size_t) token * k + i] = token == 1 ? 0 : code(token, i, 71);
    }
    if (k > 0) {
        weights[0] = -127;
        acts[0] = 127;
    }
    int8_t * d_weights;
    int8_t * d_acts;
    int32_t * d_dots;
    check(cudaMalloc(&d_weights, weights.size()));
    check(cudaMalloc(&d_acts, acts.size()));
    check(cudaMalloc(&d_dots, got.size() * sizeof(int32_t)));
    check(cudaMemcpy(d_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_acts, acts.data(), acts.size(), cudaMemcpyHostToDevice));
    probe<<<dim3((m + 7) / 8, (n + 7) / 8), 32>>>(d_weights, d_acts, m, n, k, d_dots);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(), d_dots, got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (int token = 0; token < n; ++token) {
        for (int row = 0; row < m; ++row) {
            int32_t expected = 0;
            for (int i = 0; i < k; ++i) expected += (int32_t) weights[(size_t) row * k + i] * acts[(size_t) token * k + i];
            if (got[(size_t) token * m + row] != expected) {
                std::fprintf(stderr, "m=%d n=%d k=%d row=%d token=%d: got %d expected %d\n",
                        m, n, k, row, token, got[(size_t) token * m + row], expected);
                std::exit(1);
            }
        }
    }
    check(cudaFree(d_weights));
    check(cudaFree(d_acts));
    check(cudaFree(d_dots));
}

int main() {
    cudaDeviceProp props;
    check(cudaGetDeviceProperties(&props, 0));
    if (props.major != 7 || props.minor != 5) {
        std::fprintf(stderr, "SM75 required; found SM%d%d\n", props.major, props.minor);
        return 2;
    }
    for (int m : {1, 8, 9}) {
        for (int n : {1, 8, 9}) {
            for (int k : {1, 15, 16, 17, 33, 2560, 9728}) run(m, n, k);
        }
    }
    std::puts("W8A8 SM75 MMA exact I32 dots: 63/63 cases passed");
}
