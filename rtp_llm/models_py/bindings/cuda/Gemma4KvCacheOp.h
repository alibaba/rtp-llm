#pragma once

#include <torch/extension.h>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor, at::Tensor> gemma4_gather_paged_kv_bf16(const at::Tensor& k_cache,
                                                                           const at::Tensor& v_cache,
                                                                           const at::Tensor& page_indices,
                                                                           int64_t           first_offset,
                                                                           int64_t           token_count,
                                                                           int64_t           page_size);

void gemma4_append_swa_kv_cache_bf16(const at::Tensor& key,
                                     const at::Tensor& value,
                                     const at::Tensor& batch_indices,
                                     const at::Tensor& positions,
                                     const at::Tensor& k_cache,
                                     const at::Tensor& v_cache,
                                     const at::Tensor& page_indices,
                                     const at::Tensor& page_indptr,
                                     int64_t           page_size);

}  // namespace torch_ext
