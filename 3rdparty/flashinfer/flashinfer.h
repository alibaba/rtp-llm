
#include <vector>
#include <torch/extension.h>
#include <torch/all.h>

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           unsigned int layout, int64_t cuda_stream);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                        float rope_scale, float rope_theta, int64_t cuda_stream);

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    bool use_logits_soft_cap, unsigned int head_dim, at::Tensor empty_q_data,
    at::Tensor empty_kv_data, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph, int64_t cuda_stream);

void BatchDecodeWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, std::optional<at::Tensor> alibi_slopes, at::Tensor o,
    unsigned int kv_layout_code, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    unsigned int head_dim, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    unsigned int total_num_rows, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph,
    int64_t cuda_stream);

void BatchPrefillWithPagedKVCacheRun(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, std::optional<at::Tensor> maybe_custom_mask,
    std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    std::optional<at::Tensor> maybe_qk_indptr, at::Tensor o, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

std::vector<int64_t> BatchPrefillWithKVCacheSM90Plan(
    unsigned int head_dim, bool causal, at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer, at::Tensor page_locked_int_workspace_buffer,
    at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len_arr, unsigned int total_num_rows,
    unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
    unsigned int page_size, bool enable_cuda_graph, int64_t cuda_stream);

void BatchPrefillWithRaggedKVCacheSM90Run(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes,
    at::Tensor qo_indptr, at::Tensor kv_indptr, std::optional<at::Tensor> maybe_qk_indptr,
    at::Tensor o, unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);
