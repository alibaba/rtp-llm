#pragma once

#include <torch/all.h>
#include <vector>

namespace rtp_llm {

int64_t custom_ar_init(const std::vector<int64_t>& ipc_ptrs, torch::Tensor& rank_data, int64_t rank, bool full_nvlink);

void custom_ar_all_reduce(
    int64_t fa, torch::Tensor& inp, torch::Tensor& out, int64_t reg_buffer, int64_t reg_buffer_sz_bytes);

void custom_ar_dispose(int64_t fa);

int64_t custom_ar_meta_size();

void custom_ar_register_buffer(int64_t fa, const std::vector<int64_t>& ipc_ptrs);

std::tuple<std::vector<int64_t>, std::vector<int64_t>> custom_ar_get_graph_buffer_ipc_meta(int64_t fa);

void custom_ar_register_graph_buffers(int64_t                                  fa,
                                      const std::vector<std::vector<int64_t>>& handles,
                                      const std::vector<std::vector<int64_t>>& offsets);

}  // namespace rtp_llm
