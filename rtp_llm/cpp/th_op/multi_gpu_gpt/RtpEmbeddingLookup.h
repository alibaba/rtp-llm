namespace torch_ext {
void embedding(at::Tensor& output, at::Tensor& input, at::Tensor& weight, int64_t cuda_stream);
}