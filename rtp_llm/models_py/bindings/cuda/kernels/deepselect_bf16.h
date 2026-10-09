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

}  // namespace torch_ext
