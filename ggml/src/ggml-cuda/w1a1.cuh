#include "common.cuh"

void ggml_cuda_w1a1_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_eagle_capture_dense(ggml_backend_cuda_context & ctx, const ggml_tensor * weights, const ggml_tensor * acts);
