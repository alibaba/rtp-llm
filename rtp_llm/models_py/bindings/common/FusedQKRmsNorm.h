namespace rtp_llm {
void FusedQKRMSNorm(at::Tensor& input,
                    at::Tensor& q_gamma,
                    at::Tensor& k_gamma,
                    const double layernorm_eps,
                    const int64_t q_group_num,
                    const int64_t k_group_num,
                    const int64_t m,
                    const int64_t n,
                    const int64_t norm_size);
}
