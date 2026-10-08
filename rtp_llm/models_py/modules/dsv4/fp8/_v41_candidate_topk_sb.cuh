// SPDX-License-Identifier: BSD-3-Clause
// SB15 selection body, factored into a same-CTA device function.
// Adapted from PyTorch TensorTopK.cu, commit 70d99e998b4955e0049d13a98d77ae1b14db1f45.
// See _v41_candidate_topk.LICENSE for source attribution and BSD notices.
#pragma once
#include <math_constants.h>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>

namespace v41_candidate_topk {
struct Add {
    __device__ __forceinline__ int operator()(int a, int b) const { return a + b; }
};

__device__ __forceinline__ void sb_fallback(
    const float* row, int32_t* dst, int width, int out_width, int* scratch) {
    const int k = width < 2048 ? width : 2048;
    for (int col = k + threadIdx.x; col < out_width; col += blockDim.x) dst[col] = -1;
    float cutoff = 0.0f;
    at::native::radixSelect<float, uint32_t, int>(row, k, true, width, 1, scratch, &cutoff);
    const uint32_t cutoff_key = at::native::TopKTypeConfig<float>::convert(cutoff);
    const int iterations = ((width + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int start = 0;
    for (int col = threadIdx.x; col < iterations; col += blockDim.x) {
        const bool in_range = col < width;
        const float value = in_range ? doLdg(row + col) : 0.0f;
        const bool selected = in_range && at::native::TopKTypeConfig<float>::convert(value) > cutoff_key;
        int offset, count;
        at::cuda::exclusiveBinaryPrefixScan<int, true>(scratch, selected, &offset, &count, Add{});
        if (selected) dst[start + offset] = value > -CUDART_INF_F ? col : -1;
        start += count;
    }
    int remaining = k - start;
    for (int col = threadIdx.x; col < iterations; col += blockDim.x) {
        const bool in_range = col < width;
        const float value = in_range ? doLdg(row + col) : 0.0f;
        const bool equal = in_range && at::native::TopKTypeConfig<float>::convert(value) == cutoff_key;
        int offset, count;
        at::cuda::exclusiveBinaryPrefixScan<int, true>(scratch, equal, &offset, &count, Add{});
        if (equal && offset < remaining) dst[start + offset] = value > -CUDART_INF_F ? col : -1;
        if (count >= remaining) break;
        remaining -= count;
        start += count;
    }
}
}  // namespace v41_candidate_topk
