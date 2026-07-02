#pragma once

#include <torch/all.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <vector>

namespace torch_ext {

// Semantic CUDA vertical slice for the future DSv4 CP distributed prefill
// attention op. The production implementation will replace the inside with
// symmetric-buffer communication and fused cache writes; this launcher fixes
// the Python/C++ ABI and validates rank-local HCA/CSA attention semantics.
torch::Tensor dsv4_cp_distributed_prefill_attention(const torch::Tensor& q,
                                                    const torch::Tensor& kv,
                                                    const torch::Tensor& indexer_q,
                                                    const torch::Tensor& indexer_k,
                                                    const torch::Tensor& attn_sink,
                                                    const torch::Tensor& req_id_per_token,
                                                    const torch::Tensor& position_ids,
                                                    const torch::Tensor& prefix_lengths,
                                                    const torch::Tensor& input_lengths,
                                                    const torch::Tensor& local_rows,
                                                    int64_t              compress_ratio,
                                                    int64_t              window_size,
                                                    int64_t              compressed_topk,
                                                    int64_t              compressed_region_width = -1,
                                                    int64_t              cp_rank = 0,
                                                    int64_t              cp_size = 1,
                                                    int64_t              comm_ptr = 0,
                                                    int64_t              buffer_handle = -1,
                                                    int64_t              signal_handle = -1,
                                                    int64_t              per_rank_buffer_bytes = 0,
                                                    std::vector<int64_t> rank_offsets = {},
                                                    const std::optional<torch::Tensor>& swa_k = std::nullopt,
                                                    const std::optional<torch::Tensor>& swa_k_cache = std::nullopt,
                                                    const std::optional<torch::Tensor>& swa_slot_mapping = std::nullopt,
                                                    const std::optional<torch::Tensor>& symm_buffer = std::nullopt,
                                                    int64_t symm_buffer_ptrs_dev = 0,
                                                    int64_t symm_signal_pad_ptrs_dev = 0,
                                                    pybind11::object symm_handle = pybind11::none(),
                                                    const std::optional<torch::Tensor>& compressor_kv = std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_score = std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_ape = std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_positions = std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_state_cache = std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_state_slots = std::nullopt,
                                                    int64_t compressor_ratio = 0,
                                                    const std::optional<torch::Tensor>& compressor_token_to_req =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_state_block_table =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_norm_weight =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_cos_sin_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_kv_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_kv_slots =
                                                        std::nullopt,
                                                    int64_t compressor_seq_start = 0,
                                                    bool    compressor_disable_raw_path = false,
                                                    double  compressor_rms_norm_eps = 1.0e-6,
                                                    int64_t compressor_head_dim = 0,
                                                    int64_t compressor_rope_head_dim = 0,
                                                    bool    compressor_overlap = false,
                                                    int64_t compressor_state_tokens_per_block = 0,
                                                    const std::optional<torch::Tensor>& compressor_seq_start_per_req =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_cu_seq_per_req =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& compressor_unpad_restore =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_kv =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_score =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_ape =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_positions =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_state_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_state_slots =
                                                        std::nullopt,
                                                    int64_t csa_indexer_compressor_ratio = 0,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_token_to_req =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_state_block_table =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_norm_weight =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_cos_sin_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_kv_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_kv_slots =
                                                        std::nullopt,
                                                    int64_t csa_indexer_compressor_seq_start = 0,
                                                    bool    csa_indexer_compressor_disable_raw_path = false,
                                                    double  csa_indexer_compressor_rms_norm_eps = 1.0e-6,
                                                    int64_t csa_indexer_compressor_head_dim = 0,
                                                    int64_t csa_indexer_compressor_rope_head_dim = 0,
                                                    bool    csa_indexer_compressor_overlap = false,
                                                    int64_t csa_indexer_compressor_state_tokens_per_block = 0,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_seq_start_per_req =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_cu_seq_per_req =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_compressor_unpad_restore =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_k_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_weights =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_cu_lens =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_k_pool =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_block_table =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& csa_indexer_seq_lens =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_cmp_k_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_cmp_cu_lens =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_cmp_k_pool =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_cmp_block_table =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_cmp_seq_lens =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_swa_k_cache =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_swa_cu_lens =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_swa_k_pool =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_swa_slot_mapping =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& attention_swa_gather_lens =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& kv_unpad_restore =
                                                        std::nullopt,
                                                    const std::optional<torch::Tensor>& kv_cu_lens =
                                                        std::nullopt);

}  // namespace torch_ext
