#pragma once

#include <cstdint>
#include <torch/all.h>
#include <tuple>

namespace torch_ext {

bool deepselect_fp32_available();

// Draft only. Returns temperature-scaled FP32 values and int64 vocabulary IDs.
std::tuple<torch::Tensor, torch::Tensor>
draft_markov_topk(const torch::Tensor& logits, const torch::Tensor& bias, const torch::Tensor& temperature, int64_t k);

// Candidate widths the FP32 selector instantiates natively (the v3_fp32 kernel
// has 512 / 1024 / 4096 capacity buckets). Any other k falls back to ATen topk
// inside deepselect_fp32_topk and is rejected by the DSpARK prefilter entries.
inline constexpr bool deepselect_fp32_candidate_width_supported(int64_t k) {
    return k == 512 || k == 1024 || k == 4096;
}

// Historical K4096 default of the prefilter compatibility entry points.
// Production DSpARK callers pass the live RTP_LLM_DSPARK_FORCE_TOPK width
// explicitly; the no-argument form keeps the legacy K4096 contract.
inline constexpr int64_t kDeepSelectPrefilterDefaultWidth = 4096;

// FP32 candidate selection, without an intermediate score copy/workspace.
// logits: CUDA FP32 [M,N], 512 <= N < 2^23; unit inner stride, row stride a
//         multiple of 1024B, base aligned to 16B. Storage must back the final
//         row through ceil(N / 32) * 32 elements for TMA, including view offsets.
// ends:   CUDA int32 contiguous [M], clamped to [0,N] inside the selecting CTA.
// output: CUDA int32 [M,K] for K in {512,1024,4096}, unit inner stride, row
//         stride/base aligned to 32B. The K1024 bucket is emitted without the
//         finite-filtered epilogue (see deepselect_fp32.cu's dispatch).
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

// K<=512, K=1024 and K=4096 use DeepSelect on supported Blackwell inputs; other
// shapes use ATen topk. Returned indices are int64. K1024 and K4096 return the
// selected values straight from the selector; K<=512 gathers them after the
// wider bucket is trimmed.
std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_topk(const torch::Tensor& logits, int64_t k);

// Compatibility entry points for DSV4.1 candidate selection. Both use the
// general selector, including after the optional Markov transform. `k` is the
// candidate width; it must be one of 512 / 1024 / 4096.
std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk_prefilter(const torch::Tensor& logits, int64_t k = kDeepSelectPrefilterDefaultWidth);
std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_markov_topk_prefilter(const torch::Tensor& logits,
                                      const torch::Tensor& markov_bias,
                                      const torch::Tensor& temperature,
                                      int64_t              k = kDeepSelectPrefilterDefaultWidth);

// Exact joint top-k/top-p support from the original full-vocabulary probabilities.
// top_k/top_p are CUDA row vectors; max_top_k is their largest active top-k.
void deepselect_fp32_joint_topk_topp(const torch::Tensor& probs,
                                     torch::Tensor&       output,
                                     const torch::Tensor& top_k,
                                     const torch::Tensor& top_p,
                                     int64_t              max_top_k);

}  // namespace torch_ext
