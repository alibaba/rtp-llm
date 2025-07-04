namespace torch_ext {
void layernorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps);

void fused_add_layernorm(at::Tensor& input, at::Tensor& residual, at::Tensor& bias, at::Tensor& weight, at::Tensor& beta, double eps);
}
