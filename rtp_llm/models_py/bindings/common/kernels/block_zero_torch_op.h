#pragma once

#include <torch/extension.h>
#include <optional>

namespace rtp_llm {

/// Zero the latest incomplete KV cache block for each (batch, layer).
///
/// Only blocks at a "new block boundary" — where (token_count - 1) % seq_size_per_block == 0 —
/// are zeroed.  Mid-block positions are skipped inside the kernel.
///
/// Dimensions derived from tensor shapes:
///   layer_num            = layer_base_addrs.size(0)
///   batch_size           = token_counts.size(0)
///   batch_dim            = kv_cache_block_id.size(1)
///   max_blocks_per_batch = kv_cache_block_id.size(2)
void zero_incomplete_kv_cache_blocks(const torch::Tensor&                layer_base_addrs,     // [L] int64
                                     const torch::Tensor&                kv_cache_block_id,    // [G, B, M] int32
                                     const torch::Tensor&                token_counts,         // [N] int32
                                     const std::optional<torch::Tensor>& layer_to_group,       // [L] int32
                                     int64_t                             block_stride_bytes,
                                     int64_t                             seq_size_per_block);

}  // namespace rtp_llm
