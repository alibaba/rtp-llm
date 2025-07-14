namespace torch_ext {
void layernorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps, int64_t cuda_stream);

void fused_add_layernorm(at::Tensor& input,
                         at::Tensor& residual,
                         at::Tensor& bias,
                         at::Tensor& weight,
                         at::Tensor& beta,
                         double      eps,
                         int64_t     cuda_stream);
}  // namespace torch_ext