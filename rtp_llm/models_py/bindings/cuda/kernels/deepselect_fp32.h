#pragma once

#include <torch/all.h>
#include <tuple>

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
void deepselect_fp32(const torch::Tensor& logits,
                     const torch::Tensor& ends,
                     torch::Tensor&       output,
                     bool                 filter_finite,
                     torch::Tensor*       output_values);

// Materialize the DSpARK logits + native-dtype Markov bias, divided by
// per-row temperature, as contiguous FP32 input for candidate selection.
torch::Tensor
markov_scaled_logits(const torch::Tensor& logits, const torch::Tensor& markov_bias, const torch::Tensor& temperature);

// K<=512 and K4096 use DeepSelect on supported Blackwell inputs; other
// shapes use ATen topk. Returned indices are int64.
std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_topk(const torch::Tensor& logits, int64_t k);

// Compatibility entry points for DSV4.1 K4096 selection. Both use the
// general selector, including after the optional Markov transform.
std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_topk_prefilter(const torch::Tensor& logits);
std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_markov_topk_prefilter(const torch::Tensor& logits,
                                                                               const torch::Tensor& markov_bias,
                                                                               const torch::Tensor& temperature);

// Exact joint top-k/top-p support from the original full-vocabulary probabilities.
// top_k/top_p are CUDA row vectors; max_top_k is their largest active top-k.
void deepselect_fp32_joint_topk_topp(const torch::Tensor& probs,
                                     torch::Tensor&       output,
                                     const torch::Tensor& top_k,
                                     const torch::Tensor& top_p,
                                     int64_t              max_top_k);

}  // namespace torch_ext
