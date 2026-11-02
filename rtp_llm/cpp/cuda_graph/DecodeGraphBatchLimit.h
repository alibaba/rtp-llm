#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// The scheduler must not admit a decode batch which has no captured graph.
// Graph buffers are sized separately by concurrency_limit, even when an
// explicit capture list deliberately uses a smaller maximum batch.
inline int decodeGraphBatchLimit(int buffer_batch_size, const std::vector<int>& capture_batch_sizes) {
    if (buffer_batch_size <= 0) {
        throw std::invalid_argument("CUDA graph concurrency_limit must be positive");
    }
    if (capture_batch_sizes.empty()) {
        return buffer_batch_size;
    }
    for (int batch_size : capture_batch_sizes) {
        if (batch_size <= 0 || batch_size > buffer_batch_size) {
            throw std::invalid_argument(
                "decode_capture_batch_sizes must be positive and must not exceed CUDA graph concurrency_limit");
        }
    }
    return *std::max_element(capture_batch_sizes.begin(), capture_batch_sizes.end());
}

}  // namespace rtp_llm
