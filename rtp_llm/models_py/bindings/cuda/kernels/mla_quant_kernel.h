#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

// Indexer K quantization and cache function
// Quantizes key tensor to FP8 and stores both quantized values and scales in kv_cache
void indexer_k_quant_and_cache(torch::Tensor&     k,                 // [num_tokens, head_dim]
                               torch::Tensor&     kv_cache,          // [num_blocks, block_size, cache_stride]
                               torch::Tensor&     slot_mapping,      // [num_tokens]
                               int64_t            quant_block_size,  // quantization block size (e.g., 128)
                               const std::string& scale_fmt  // scale format: "ue8m0" for power-of-2, "" for direct
);

// Gather quantized K cache for indexer
// Retrieves quantized keys and scales from paged cache for computation
void cp_gather_indexer_k_quant_cache(const torch::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
                                     torch::Tensor&       dst_k,        // [num_tokens, head_dim]
                                     torch::Tensor&       dst_scale,    // [num_tokens, head_dim / quant_block_size * 4]
                                     const torch::Tensor& block_table,  // [batch_size, num_blocks]
                                     const torch::Tensor& cu_seq_lens   // [batch_size + 1]
);

// Indexer K FP4 quantization and cache (Blackwell-only).
// Packs BF16 K into FP4 e2m1 + UE8M0 scales matching
// deep_gemm.utils.per_token_cast_to_fp4(use_ue8m0=True, gran_k=32,
// use_packed_ue8m0=True). cache_stride must be >= head_dim/2 + head_dim/gran_k.
void indexer_k_quant_and_cache_fp4(torch::Tensor& k,             // [num_tokens, head_dim] bf16
                                   torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
                                   torch::Tensor& slot_mapping,  // [num_tokens] int64
                                   int64_t        gran_k         // FP4 quant group size (32)
);

// Gather FP4 indexer K from paged cache into contiguous (dst_k, dst_scale)
// buffers. dst_k: int8 [num_tokens, head_dim/2]. dst_scale: int32
// [num_tokens, head_dim/gran_k/4] holding packed UE8M0 (4 bytes per int32).
void cp_gather_indexer_k_quant_cache_fp4(const torch::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
                                         torch::Tensor&       dst_k,        // [num_tokens, head_dim/2] int8
                                         torch::Tensor&       dst_scale,    // [num_tokens, head_dim/gran_k/4] int32
                                         const torch::Tensor& block_table,  // [batch_size, num_blocks]
                                         const torch::Tensor& cu_seq_lens   // [batch_size + 1]
);

// Concat and cache MLA (Multi-Head Latent Attention)
// Concatenates kv_c and k_pe and stores in paged KV cache
void concat_and_cache_mla(torch::Tensor&     kv_c,          // [num_tokens, kv_lora_rank]
                          torch::Tensor&     k_pe,          // [num_tokens, pe_dim]
                          torch::Tensor&     kv_cache,      // [num_blocks, block_size, (kv_lora_rank + pe_dim)]
                          torch::Tensor&     slot_mapping,  // [num_tokens] or [num_actual_tokens]
                          const std::string& kv_cache_dtype,
                          torch::Tensor&     scale);

// Gather and upconvert FP8 KV cache to BF16 workspace (MLA DeepSeek V3 layout)
// src_cache: [num_blocks, block_size, 656] uint8 (512 fp8 + 16 scale + 128 rope bf16)
// dst_compressed_kv: [total_tokens, 512] bfloat16
// dst_k_pe: [total_tokens, 64] bfloat16
void cp_gather_and_upconvert_fp8_kv_cache(const torch::Tensor& src_cache,          // [NUM_BLOCKS, BLOCK_SIZE, 656]
                                          torch::Tensor&       dst_compressed_kv,  // [TOT_TOKENS, 512]
                                          torch::Tensor&       dst_k_pe,           // [TOT_TOKENS, 64]
                                          const torch::Tensor& block_table,        // [BATCH, BLOCK_INDICES]
                                          const torch::Tensor& seq_lens,           // [BATCH]
                                          const torch::Tensor& workspace_starts,   // [BATCH]
                                          int64_t              batch_size);

// V2: Gather and upconvert FP8 KV cache to a single fused BF16 buffer [total_tokens, 576]
// One CUDA block per token for full GPU utilization.
void cp_gather_and_upconvert_fp8_kv_cache_v2(const torch::Tensor& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
                                             torch::Tensor&       dst_fused,         // [TOT_TOKENS, 576]
                                             const torch::Tensor& block_table,       // [BATCH, BLOCK_INDICES]
                                             const torch::Tensor& seq_lens,          // [BATCH]
                                             const torch::Tensor& workspace_starts,  // [BATCH]
                                             int64_t              batch_size,
                                             int64_t              total_tokens);

}  // namespace rtp_llm
