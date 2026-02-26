#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <torch/extension.h>

namespace rtp_llm {

/**
 * Unified high-performance in-place parameter update for all three
 * FlashInferTRTLLM*Impl::prepare_cuda_graph paths.
 *
 * The three modes are selected by which optional arguments are provided:
 *
 * ── Decode mode (lengths_b = nullopt, cu_kv_seqlens = nullopt) ──────────────
 *   seq_lens ← copy(lengths_a)          # lengths_a = sequence_lengths_plus_1_d
 *   kv_cache_offset ← block_id_to_offset(kv_cache_block_id)
 *
 * ── SpecDecode mode (lengths_b provided, cu_kv_seqlens = nullopt) ───────────
 *   seq_lens ← lengths_a + lengths_b    # prefix_lengths + input_lengths (avoids .item() sync)
 *   kv_cache_offset ← block_id_to_offset(kv_cache_block_id)
 *
 * ── Prefill mode (lengths_b provided, cu_kv_seqlens provided) ───────────────
 *   seq_lens        ← lengths_a + lengths_b          # input_lengths + prefix_lengths
 *   cu_kv_seqlens[1:] ← cumsum(ceil(seq_lens / page_size))
 *   kv_cache_offset ← block_id_to_offset(kv_cache_block_id)
 *
 * Parameters
 * ----------
 * seq_lens         [batch_size]              int32 CUDA tensor, updated IN-PLACE
 * kv_cache_offset  [B_cap, 1, 2, M_cap]     int32 CUDA tensor, updated IN-PLACE
 * kv_cache_block_id [batch_size, max_blocks] int32 CUDA tensor, read-only
 * lengths_a        [batch_size]              int32 CUDA tensor
 *                  decode:    sequence_lengths_plus_1_d
 *                  spec/prefill: first addend (prefix_lengths_d / input_lengths_d)
 * lengths_b        [batch_size]              int32 CUDA tensor  (optional)
 *                  spec/prefill: second addend (input_lengths_d / prefix_lengths_d)
 * cu_kv_seqlens    [batch_size+1]            int32 CUDA tensor  (optional, prefill only)
 * page_size        tokens per KV-cache block (required when cu_kv_seqlens is provided)
 */
void trtllm_gen_update_cuda_graph_params(torch::Tensor                seq_lens,
                                         torch::Tensor                kv_cache_offset,
                                         torch::Tensor                kv_cache_block_id,
                                         torch::Tensor                lengths_a,
                                         c10::optional<torch::Tensor> lengths_b,
                                         c10::optional<torch::Tensor> cu_kv_seqlens,
                                         int64_t                      page_size);

void registerTRTLLMGenCudaGraphOp(py::module& m);

}  // namespace rtp_llm
