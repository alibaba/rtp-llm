
#include <vector>
#include <torch/extension.h>
#include <torch/all.h>

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t layout, int64_t cuda_stream);

void append_paged_mla_kv_cache(at::Tensor append_ckv, at::Tensor append_kpe, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor ckv_cache, at::Tensor kpe_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t cuda_stream);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, int64_t cuda_stream);

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q,
                                      at::Tensor k,
                                      at::Tensor q_rope,
                                      at::Tensor k_rope,
                                      at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids,
                                      bool       interleave,
                                      int64_t    cuda_stream);

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
    bool enable_cuda_graph, int64_t window_left, double logits_soft_cap, int64_t head_dim_qk,
    int64_t head_dim_vo, at::Tensor empty_q_data, at::Tensor empty_kv_data, int64_t cuda_stream);


void BatchDecodeWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    int64_t kv_layout_code, int64_t window_left, std::optional<at::Tensor> maybe_alibi_slopes,
    double logits_soft_cap, double sm_scale, double rope_rcp_scale, double rope_rcp_theta, int64_t cuda_stream);


at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
    bool enable_cuda_graph, int64_t head_dim_qk, int64_t head_dim_vo, bool causal,
    int64_t cuda_stream);


void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left , std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_mask_indptr,
    std::optional<at::Tensor> maybe_alibi_slopes, double logits_soft_cap, double sm_scale, double rope_rcp_scale, double rope_rcp_theta, int64_t cuda_stream);


at::Tensor BatchMLAPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                      at::Tensor int_workspace_buffer,
                                      at::Tensor page_locked_int_workspace_buffer,
                                      at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len,
                                      int64_t num_heads, int64_t head_dim_o, bool causal,
                                      int64_t cuda_stream);

void BatchMLAPagedAttentionRun(at::Tensor                float_workspace_buffer,
                               at::Tensor                int_workspace_buffer,
                               at::Tensor                plan_info_vec,
                               at::Tensor                q_nope,
                               at::Tensor                q_pe,
                               at::Tensor                ckv_cache,
                               at::Tensor                kpe_cache,
                               at::Tensor                kv_indices,
                               at::Tensor                o,
                               std::optional<at::Tensor> maybe_lse,
                               int64_t                   mask_mode_code,
                               int64_t                   num_heads,
                               int64_t                   page_size,
                               double                    sm_scale,
                               int64_t                   cuda_stream);

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_k_arr,
                               int64_t top_k_val, bool deterministic, int64_t cuda_stream);

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_p_arr,
                               double top_p_val, bool deterministic, int64_t cuda_stream);

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples,
                                     at::Tensor samples, at::Tensor success,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, int64_t cuda_stream);

void top_p_renorm_probs(at::Tensor                probs,
                        at::Tensor                renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr,
                        double                    top_p_val,
                        int64_t                   cuda_stream);

void top_k_renorm_probs(at::Tensor                probs,
                        at::Tensor                renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr,
                        int64_t                   top_k_val,
                        int64_t                   cuda_stream);

void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
    at::Tensor uniform_samples, at::Tensor target_probs,
    at::Tensor output_token_ids, at::Tensor output_accepted_token_num,
    at::Tensor output_emitted_token_num, bool deterministic,
    int64_t cuda_stream);

void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
             int64_t cuda_stream);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
                       int64_t cuda_stream);

// std::vector<int64_t> BatchPrefillWithKVCacheSM90Plan(
//     unsigned int head_dim, bool causal, at::Tensor float_workspace_buffer,
//     at::Tensor int_workspace_buffer, at::Tensor page_locked_int_workspace_buffer,
//     at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len_arr, unsigned int total_num_rows,
//     unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
//     unsigned int page_size, bool enable_cuda_graph, int64_t cuda_stream);

// void BatchPrefillWithRaggedKVCacheSM90Run(
//     unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
//     std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
//     std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes,
//     at::Tensor qo_indptr, at::Tensor kv_indptr, std::optional<at::Tensor> maybe_qk_indptr,
//     at::Tensor o, unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
//     float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);
