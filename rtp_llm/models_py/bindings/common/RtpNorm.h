namespace torch_ext {
void layernorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps);

void fused_add_layernorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& bias, at::Tensor& weight, at::Tensor& beta, double eps);
void fused_add_layernorm_quant_fp8(at::Tensor& input,
                                   at::Tensor& residual,
                                   at::Tensor& bias,
                                   at::Tensor& weight,
                                   at::Tensor& beta,
                                   at::Tensor& output,
                                   at::Tensor& scales,
                                   double      eps);
void fused_bias_add(at::Tensor& input, at::Tensor& bias);
void fused_bias_gelu(at::Tensor& input, at::Tensor& bias);
void fused_bias_gelu_quant_fp8(at::Tensor& input, at::Tensor& bias, at::Tensor& output, at::Tensor& scales);
}  // namespace torch_ext
