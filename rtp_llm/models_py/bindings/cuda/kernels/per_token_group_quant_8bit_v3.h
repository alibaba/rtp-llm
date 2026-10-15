#pragma once

#include <ATen/ATen.h>

namespace rtp_llm {

// V4.1: finite contiguous BF16 [M,K], K a positive multiple of 128.
// Q is contiguous E4M3 [M,K]. S is int32 [M,K/128], column-major with
// stride (1, round_up(M,4)). Four group32 UE8M0 codes are packed per word.
// Input/Q pointers must be 16-byte aligned; input, Q and S byte spans
// (including S padding) must not overlap. M=0 is a no-op.
// Matches legacy quantization's 1e-10 scale floor; not v2's tiny-value policy.
void per_token_group_quant_fp8_v3(const at::Tensor& input, at::Tensor& output_q, at::Tensor& output_s);

}  // namespace rtp_llm
