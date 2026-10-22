#pragma once

#include <torch/all.h>

namespace torch_ext {

// Whether this build contains the SM100/SM103 DeepSelect implementation.
bool deepselect_bf16_available();

// BF16 K512 selection, without an intermediate FP32 score tensor.
// logits: CUDA BF16 [M,N], N >= 512 and divisible by 512; unit inner stride,
//         row stride a multiple of 1024 bytes, base aligned to 16 bytes.
// ends:   CUDA int32 contiguous [M], clamped to [0,N] inside the selecting CTA.
// output: CUDA int32 [M,512], unit inner stride, row stride/base aligned to 32B.
// All tensors must share a device; output must not alias an input.
// Output order/tie membership is unspecified. Short rows return prefix indices
// followed by -1. Long rows containing NaN are entirely -1; this deliberately
// differs from torch.topk's NaN ordering. The caller filters selected +/-inf
// and short-row NaN in its fused remapping kernel.
void deepselect_bf16(const torch::Tensor& logits, const torch::Tensor& ends, torch::Tensor& output);

// Dense token K512 with the same storage/bounds contract as above. NaN ranks
// like +inf instead of invalidating the whole row. When filter_finite is true,
// selected NaN/+inf/-inf become -1, including short rows. When false, callers
// must filter selected nonfinite scores before publishing token indices.
void deepselect_bf16_dense(const torch::Tensor& logits,
                           const torch::Tensor& ends,
                           torch::Tensor&       output,
                           bool                 filter_finite);

// Complete candidate-block K2048: ascending distinct IDs in the valid prefix,
// followed by -1 for short rows. NaN ranks like +inf; no finite filtering.
// The last visible block is always included (replaces the greatest selected ID
// if absent), matching complete decode candidate publication. Same BF16 input
// alignment/ends contract as above; output is int32 [M,2048].
void deepselect_bf16_candidates(const torch::Tensor& scores,
                                const torch::Tensor& ends,
                                torch::Tensor&       output);

}  // namespace torch_ext
