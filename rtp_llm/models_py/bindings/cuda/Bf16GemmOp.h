#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight);
at::Tensor gemma4_gather_rows_bf16(const at::Tensor& input, const at::Tensor& indices);
at::Tensor gemma4_logit_softcap_fp32(const at::Tensor& input, double cap);
at::Tensor gemma4_qk_bmm_8192_bf16(const at::Tensor& q, const at::Tensor& k);
at::Tensor gemma4_qk_bmm_8192_bf16_key_len(const at::Tensor& q, const at::Tensor& k, int64_t key_len);
at::Tensor gemma4_pv_bmm_8192_bf16(const at::Tensor& probabilities, const at::Tensor& v);
void       gemma4_pv_bmm_8192_bf16_out(const at::Tensor& probabilities, const at::Tensor& v, at::Tensor& output);
void       gemma4_pv_bmm_8192_bf16_out_key_len(const at::Tensor& probabilities,
                                               const at::Tensor& v,
                                               at::Tensor&       output,
                                               int64_t           key_len);
at::Tensor gemma4_swa_pv_bmm_8192_bf16(const at::Tensor& probabilities, const at::Tensor& v);
void       gemma4_swa_pv_bmm_8192_bf16_out(const at::Tensor& probabilities, const at::Tensor& v, at::Tensor& output);

}  // namespace torch_ext
