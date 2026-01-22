#pragma once

#include <torch/extension.h>

namespace torch_ext {

// Indexer K quantization and cache operation
// Quantizes key tensor to FP8 and stores in paged KV cache with scales
void indexer_k_quant_and_cache(at::Tensor&        k,                 // [num_tokens, head_dim]
                               at::Tensor&        kv_cache,          // [num_blocks, block_size, cache_stride]
                               at::Tensor&        slot_mapping,      // [num_tokens]
                               int64_t            quant_block_size,  // quantization block size (e.g., 128)
                               const std::string& scale_fmt  // scale format: "ue8m0" for power-of-2, "" for direct
);

// Gather indexer K quantized cache operation
// Retrieves quantized keys and scales from paged cache
void cp_gather_indexer_k_quant_cache(const at::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
                                     at::Tensor&       dst_k,        // [num_tokens, head_dim]
                                     at::Tensor&       dst_scale,    // [num_tokens, head_dim / quant_block_size * 4]
                                     const at::Tensor& block_table,  // [batch_size, num_blocks]
                                     const at::Tensor& cu_seq_lens   // [batch_size + 1]
);

}  // namespace torch_ext
