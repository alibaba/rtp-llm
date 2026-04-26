
#include <vector>
#include <torch/extension.h>
#include <torch/all.h>

// Sampling APIs
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
