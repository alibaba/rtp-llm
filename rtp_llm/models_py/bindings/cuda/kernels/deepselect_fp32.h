#pragma once

#include <torch/all.h>

namespace torch_ext {

bool deepselect_fp32_available();

// FP32 K512 token selection, without an intermediate score copy/workspace.
// logits: CUDA FP32 [M,N], 512 <= N < 2^23; unit inner stride, row stride a
//         multiple of 1024B, base aligned to 16B. Storage must back the final
//         row through ceil(N / 32) * 32 elements for TMA, including view offsets.
// ends:   CUDA int32 contiguous [M], clamped to [0,N] inside the selecting CTA.
// output: CUDA int32 [M,512], unit inner stride, row stride/base aligned to 32B.
// All tensors share a device; output must not alias an input. Output order and
// membership within ties are unspecified. NaN occupies top-k slots as +inf.
// With filter_finite=true, selected NaN/+inf/-inf becomes -1, also in short
// rows. Otherwise the caller must apply its existing finite-filter epilogue.
void deepselect_fp32(const torch::Tensor& logits, const torch::Tensor& ends, torch::Tensor& output, bool filter_finite);

}  // namespace torch_ext
