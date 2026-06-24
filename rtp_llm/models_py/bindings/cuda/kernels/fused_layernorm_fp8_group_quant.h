#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace rtp_llm {

// dtype_flag: 0 = BFloat16, 1 = Float16
void invokeAddBiasResLayerNormFP8GroupQuant(
    void*        normed_output,    // [tokens, hidden_dim] BF16/FP16 normed output (for residual)
    void*        residual_output,  // [tokens, hidden_dim] BF16/FP16 pre-norm residual (in-place)
    void*        fp8_output,       // [tokens, hidden_dim] FP8 E4M3 quantized normed output
    float*       group_scales,     // column-major [hidden_dim/group_size, tokens] float32
    const void*  input,
    const void*  residual,
    const void*  bias,
    const void*  gamma,
    const void*  beta,
    float        eps,
    int          tokens,
    int          hidden_dim,
    int          group_size,
    int          scale_stride,
    int          dtype_flag,
    cudaStream_t stream);

}  // namespace rtp_llm
