#pragma once

#if USING_CUDA
#include <cuda_runtime.h>

namespace rtp_llm {

// Fused length-predictor encoder:
//   out[b][j] = gelu_erf( dot(standardize(hidden[b]), weight_t[:, j]) + bias[j] )
//
// - standardize: per-row (x - mean) / sqrt(var + eps), biased variance,
//   matching torch::layer_norm. LayerNorm's gamma/beta are expected to be
//   pre-folded into weight_t/bias by the weight loader, so the kernel never
//   sees them.
// - weight_t layout is [hidden_dim, feature_dim] (the transpose of the
//   nn.Linear weight): the inner rank-1 loop then reads feature_dim
//   consecutive floats per step, fully coalesced across threads.
// - gelu_erf: 0.5 * y * (1 + erf(y / sqrt(2))), bit-matching torch's default
//   (approximate='none') GELU up to erff precision.
//
// One thread block per row; requires hidden_dim * 4 bytes of dynamic shared
// memory (the launcher raises the max-dynamic-smem attribute when needed).
// T is float, __half, or __nv_bfloat16. Output is always fp32.
template<typename T>
void invokeLengthEncoderForward(const T* __restrict__ hidden,        // [batch, hidden_dim]
                                const float* __restrict__ weight_t,  // [hidden_dim, feature_dim]
                                const float* __restrict__ bias,      // [feature_dim]
                                float* __restrict__ out,             // [batch, feature_dim]
                                int          batch,
                                int          hidden_dim,
                                int          feature_dim,
                                float        eps,
                                cudaStream_t stream);

}  // namespace rtp_llm

#endif  // USING_CUDA
