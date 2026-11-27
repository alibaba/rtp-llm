#pragma once

#include <torch/all.h>

namespace torch_ext {

// Vectorized mHC post-residual merge:
//   out[t, o, d] = bf16(post[t, o] * x[t, d] + sum_{i=0}^{H-1} comb[t, i, o] * residual[t, i, d])
// Supports contiguous BF16 ``x`` ([..., D]), ``residual`` ([..., H, D]), and ``out`` ([..., H, D]),
// with FP32 or BF16 contiguous ``post`` ([..., H] or [..., H, 1]) and ``comb`` ([..., H, H]).
// In-place aliasing (``out.data_ptr() == residual.data_ptr()``) is race-free.
void dsv4_mhc_post_fwd(const torch::Tensor& x,
                       const torch::Tensor& residual,
                       const torch::Tensor& post,
                       const torch::Tensor& comb,
                       torch::Tensor&       out);

}  // namespace torch_ext
