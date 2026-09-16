// Copyright 2026 Alibaba Group Holding Limited.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace rtp_llm {
void invokePpuSiluMulMxfp4(const void* input,
                           uint8_t* output,
                           uint8_t* scale,
                           int64_t stride_input,
                           int64_t stride_output,
                           int64_t stride_scale_p_bytes,
                           int64_t stride_scale_n_bytes,
                           int hidden_size,
                           int num_tokens,
                           bool apply_swiglu_limit,
                           float swiglu_limit,
                           cudaStream_t stream);
}  // namespace rtp_llm
