#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// Adapted from atrex trtllm_all_reduce_fusion for rtp-llm ROCm backend.

#include <cstdint>
#include <vector>
#include <torch/all.h>

namespace rtp_llm {

using fptr_t = int64_t;

fptr_t init_ar_fusion(int64_t device_id,
                      int64_t rank,
                      int64_t world_size,
                      int64_t max_size_in_bytes,
                      int64_t comm_ptrs_buf_len);

void destroy_ar_fusion(fptr_t fptr);

torch::Tensor get_ar_fusion_barrier_handle(fptr_t fptr);

torch::Tensor get_ar_fusion_data_handle(fptr_t fptr);

void open_ar_fusion_barrier_handles(fptr_t fptr, std::vector<torch::Tensor> handles);

void open_ar_fusion_data_handles(fptr_t fptr, std::vector<torch::Tensor> handles);

void ar_fusion_capture_clear(fptr_t fptr);

std::vector<torch::Tensor> get_ar_fusion_captured_handles(fptr_t fptr);

torch::Tensor get_ar_fusion_captured_offsets(fptr_t fptr);

void open_ar_fusion_captured_handles(fptr_t fptr,
                                     std::vector<torch::Tensor> handles,
                                     std::vector<int64_t> offsets,
                                     int64_t ptr_idx);

void allreduce_rms(fptr_t fptr,
                   torch::Tensor& allreduce_in,
                   torch::Tensor& residual_in,
                   torch::Tensor& rms_gamma,
                   torch::Tensor& residual_out,
                   torch::Tensor& norm_out,
                   torch::Tensor& scale_out,
                   double eps,
                   int64_t quant_type);

void allreduce(fptr_t fptr,
               torch::Tensor& allreduce_in,
               torch::Tensor& allreduce_out);

}  // namespace rtp_llm
