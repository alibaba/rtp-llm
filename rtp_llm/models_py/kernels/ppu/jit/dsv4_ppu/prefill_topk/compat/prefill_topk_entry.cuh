// SPDX-License-Identifier: Apache-2.0
// RTP host-only export adapter. Device kernels remain the pinned SG originals.
#pragma once
#include "../csrc/deepseek_v4/topk_v2.cuh"

namespace sglang {
struct PrefillTopKEntry {
  static void forward(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> blocks,
      const tvm::ffi::TensorView out_offsets,
      const tvm::ffi::TensorView output) {
    if (blocks.has_value()) {
      TopKKernel<false>::transform_candidates(scores, seq_lens, blocks.value(), out_offsets, output);
    } else {
      TopKKernel<false>::transform_ragged(
          scores, seq_lens, tvm::ffi::Optional<tvm::ffi::TensorView>(), out_offsets, output);
    }
  }
};
}  // namespace sglang
