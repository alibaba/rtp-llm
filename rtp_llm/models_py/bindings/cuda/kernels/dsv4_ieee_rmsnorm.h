#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

namespace rtp_llm {

// IEEE-precise RMSNorm (vendored from flashinfer norm.cuh, with the
// `math::rsqrt = rsqrt.approx.ftz.f32` PTX path replaced by IEEE
// `__frcp_rn(__fsqrt_rn(x))`).
//
// Why a separate copy: DeepSeek-V4's per-block residual RMSNorm runs
// 60+ times with attn_norm.weight ≈ 0.03; the ~2 ULP error from the
// approx PTX rsqrt accumulates across layers and at S=64K shifts the
// first-token logit of one specific token (id=271 for the smoke prompt)
// down by ~0.5 vs the IEEE chain — flipping argmax onto an adjacent
// candidate (id=223) and producing context-leakage gibberish.
// See compare.py logs in /tmp/dsv4_logits/ from 2026-05-03 bisect.
//
// Contract:
//   input  : [num_tokens, dim] bf16 contiguous
//   weight : [dim] bf16
//   output : [num_tokens, dim] bf16 contiguous (may alias input)
//
// Mirrors `flashinfer::norm::RMSNorm<__nv_bfloat16>` host signature.
void invokeDsv4IeeeRmsNorm(__nv_bfloat16*       output,
                           const __nv_bfloat16* input,
                           const __nv_bfloat16* weight,
                           int                  num_tokens,
                           int                  dim,
                           float                eps,
                           cudaStream_t         stream);

}  // namespace rtp_llm

namespace torch_ext {

// Pybind entry: same shape as `rtp_llm_ops.rmsnorm` (bf16 inputs).
//   output: [N, D] bf16
//   input : [N, D] bf16
//   weight: [D]    bf16
void dsv4_ieee_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream);

}  // namespace torch_ext
