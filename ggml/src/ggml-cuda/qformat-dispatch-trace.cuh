#pragma once

#include "ggml.h"

// Opt-in, process-wide diagnostic for the Q4_0/Q8_0 CUDA matmul paths.
// GGML_CUDA_QFORMAT_DISPATCH_TRACE=1 logs the first call for each
// weight type, kernel/compute type, MUL_MAT_ID mode, and fusion mode.
enum class ggml_cuda_qformat_kernel {
    mmvq,
    mmq,
    cublas,
};

void ggml_cuda_trace_qformat_dispatch(
    ggml_cuda_qformat_kernel kernel,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst,
    ggml_type activation_kernel_type,
    bool activation_quantized,
    bool matmul_id,
    bool fused);
