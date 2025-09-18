namespace torch_ext {

void per_token_group_quant_int8(at::Tensor& input, at::Tensor& output_q, at::Tensor& output_s, int64_t group_size, double eps, double int8_min, double int8_max);

void per_token_group_quant_fp8(at::Tensor& input, at::Tensor& output_q, at::Tensor& output_s, int64_t group_size, double eps, double fp8_min, double fp8_max);

}