#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"

namespace rtp_llm {

/// @brief Debug kernel to print 2D data blocks
/// @param data Input tensor to debug
/// @param start_row Starting row index
/// @param start_col Starting column index
/// @param m Number of rows to print
/// @param n Number of columns to print
/// @param row_len Length of each row (stride)
/// @param info_id Debug identifier
void debugKernel(const torch::Tensor& data,
                 int64_t              start_row,
                 int64_t              start_col,
                 int64_t              m,
                 int64_t              n,
                 int64_t              row_len,
                 int64_t              info_id);

}  // namespace rtp_llm