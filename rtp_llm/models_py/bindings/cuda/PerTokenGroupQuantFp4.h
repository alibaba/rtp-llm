namespace torch_ext {

void per_token_group_quant_fp4(at::Tensor& input,
                               at::Tensor& output_q,
                               at::Tensor& output_s,
                               int64_t     group_size,
                               double      eps,
                               bool        use_packed_ue8m0);

}  // namespace torch_ext
