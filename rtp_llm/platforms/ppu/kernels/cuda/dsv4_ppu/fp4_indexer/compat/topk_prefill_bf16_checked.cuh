#pragma once
#include "topk_prefill_bf16.cuh"

// Streaming scatter uses independent warp work stealing. Every warp must
// finish before counters or the aliased tie buffer can be consumed.
__global__ __launch_bounds__(prefill_bf16::kBlockSize, 1)
void topk_prefill_bf16_checked_kernel(TopKPrefillBF16Params params, int32_t* fallback) {
  using namespace prefill_bf16;
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const int32_t start = params.row_starts[row];
  const int32_t end = params.row_ends[row];
  const auto* scores = params.scores + static_cast<int64_t>(row) * params.score_stride;
  // The original vector loader requires 16-byte alignment and emits offsets
  // relative to row_start. The generic radix path handles other layouts.
  if (start != 0 || (reinterpret_cast<uintptr_t>(scores) & 15u) != 0u) {
    if (tx == 0) fallback[row] = 1;
    return;
  }
  const uint32_t length = end > 0 ? static_cast<uint32_t>(end) : 0u;
  auto* out = params.page_indices + static_cast<int64_t>(row) * kTopK;
  const auto* pages = params.page_table + static_cast<int64_t>(row) * params.page_table_stride;
  if (length <= kTopK) {
    naive_transform_bf16(pages, out, nullptr, length, params.page_bits);
    if (tx == 0) fallback[row] = 0;
    return;
  }
  extern __shared__ char storage[];
  auto* smem = reinterpret_cast<PrefillBF16Smem*>(storage);
  auto* indices = reinterpret_cast<int32_t*>(storage + sizeof(PrefillBF16Smem));
  streaming_topk_bf16(scores, length, indices, smem);
  __syncthreads();
  // FP16 coarse bins separate signed zero; radix canonicalizes both signs.
  const bool retry = smem->match.bin == 2047u || smem->match.bin == 2048u ||
                     smem->counter_eq > kMaxTies ||
                     smem->counter_gt != smem->match.above_count ||
                     smem->counter_eq != smem->match.equal_count;
  if (tx == 0) fallback[row] = retry ? 1 : 0;
  if (retry) return;
  tie_handle_and_transform_bf16(params, indices, out, nullptr, pages, smem);
}

struct TopKPrefillBF16CheckedKernel {
  static void transform(const tvm::ffi::TensorView scores,
                        const tvm::ffi::TensorView starts,
                        const tvm::ffi::TensorView ends,
                        const tvm::ffi::TensorView pages,
                        const tvm::ffi::TensorView out,
                        const tvm::ffi::TensorView fallback) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<bf16_t>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(starts);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(ends);
    TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device).verify(pages);
    TensorMatcher({B, prefill_bf16::kTopK}).with_dtype<int32_t>().with_device(device).verify(out);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(fallback);
    const auto params = TopKPrefillBF16Params{
        .scores = static_cast<const __nv_bfloat16*>(scores.data_ptr()),
        .row_starts = static_cast<const int32_t*>(starts.data_ptr()),
        .row_ends = static_cast<const int32_t*>(ends.data_ptr()),
        .page_table = static_cast<const int32_t*>(pages.data_ptr()),
        .page_indices = static_cast<int32_t*>(out.data_ptr()),
        .raw_indices = nullptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = 31,
    };
    constexpr auto kernel = topk_prefill_bf16_checked_kernel;
    constexpr auto bytes = sizeof(PrefillBF16Smem) + prefill_bf16::kTopK * sizeof(int32_t);
    setup_kernel_smem_once_pf<kernel, bytes>();
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), prefill_bf16::kBlockSize, device.unwrap(), bytes)(
        kernel, params, static_cast<int32_t*>(fallback.data_ptr()));
  }
};
