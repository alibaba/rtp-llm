namespace torch_ext {

void per_token_group_quant_int8(at::Tensor& input,
                                at::Tensor& output_q,
                                at::Tensor& output_s,
                                int64_t     group_size,
                                double      eps,
                                double      int8_min,
                                double      int8_max,
                                bool        scale_ue8m0);

void per_token_group_quant_fp8(at::Tensor& input,
                               at::Tensor& output_q,
                               at::Tensor& output_s,
                               int64_t     group_size,
                               double      eps,
                               double      fp8_min,
                               double      fp8_max,
                               bool        scale_ue8m0);

void per_token_group_quant_fp8_v2(at::Tensor&                         input,
                                  at::Tensor&                         output_q,
                                  at::Tensor&                         output_s,
                                  int64_t                             group_size,
                                  double                              eps,
                                  double                              fp8_min,
                                  double                              fp8_max,
                                  bool                                scale_ue8m0,
                                  bool                                fuse_silu_and_mul,
                                  const std::optional<torch::Tensor>& masked_m);
}  // namespace torch_ext
