#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

// Rebuild the tail of a captured prefill padding_offset buffer after the live
// batch shrank below the CUDA graph capture width.
//
// Background: MTP draft prefill CUDA graph replay only refreshes the live
// prefix of the captured pinned padding_offset mirror (copyToHostMirror(...,
// current_seq_len * sizeof(int))). Tail rows inside the capture capacity keep
// the slot mapping left behind by the previous, larger batch. Consumers such
// as add_fusedQKV_bias_transpose_prefill_kernel_v1 decode every captured row
// as tgt_token_idx = row + padding_offset[row] and batch_idx =
// tgt_token_idx / seq_len, then dereference per-batch page-table entries, so a
// stale tail row resolves to a batch_idx beyond the live page table and the
// replay crashes with WARP_ILLEGAL_ADDRESS.
//
// This helper rewrites only the tail rows [live_tokens, padding_offset_capacity)
// so every tail row decodes to last_live_tgt, the unpacked target index of the
// last live row:
//
//   padding_offset[row] = last_live_tgt - row
//   => row + padding_offset[row] == last_live_tgt
//
// Anchoring every tail row on the last live row is idempotent and safe for
// consumers that do not carry a live-row-count guard: the worst such a
// consumer can do is re-process the final live token, and no tail row can ever
// decode past the live page table. The live prefix [0, live_tokens) is never
// modified.
inline bool rebuildPrefillPaddingOffsetTail(int32_t* padding_offset, size_t padding_offset_capacity, int live_tokens) {
    if (padding_offset == nullptr || padding_offset_capacity == 0 || live_tokens <= 0
        || static_cast<size_t>(live_tokens) > padding_offset_capacity) {
        return false;
    }

    if (static_cast<size_t>(live_tokens) == padding_offset_capacity) {
        // No tail rows: the captured buffer is fully live, nothing to rebuild.
        return true;
    }

    const int32_t last_live_row = static_cast<int32_t>(live_tokens) - 1;
    const int32_t last_live_tgt = last_live_row + padding_offset[last_live_row];
    for (size_t row = static_cast<size_t>(live_tokens); row < padding_offset_capacity; ++row) {
        padding_offset[row] = last_live_tgt - static_cast<int32_t>(row);
    }
    return true;
}

}  // namespace rtp_llm
