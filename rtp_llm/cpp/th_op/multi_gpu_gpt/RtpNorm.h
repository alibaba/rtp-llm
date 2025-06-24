namespace torch_ext {
void layernorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps, int64_t cuda_stream);
}