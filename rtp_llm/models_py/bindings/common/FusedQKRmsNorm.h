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

#if USING_ROCM
// Warp-per-(token, head) wave64 single-pass V2 kernel (ROCm only).
// Same semantics as FusedQKRMSNorm. When norm_size != 256 or bias is present
// the V2 path internally falls back to the baseline kernel.
void FusedQKRMSNormV2(at::Tensor&   input,
                      at::Tensor&   q_gamma,
                      at::Tensor&   k_gamma,
                      const double  layernorm_eps,
                      const int64_t q_group_num,
                      const int64_t k_group_num,
                      const int64_t m,
                      const int64_t n,
                      const int64_t norm_size);
#endif
}
