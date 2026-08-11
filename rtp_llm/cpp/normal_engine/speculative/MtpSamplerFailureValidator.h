#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <torch/all.h>

namespace rtp_llm::speculative {

inline std::vector<size_t> findFailedSamplerStreamIndices(const torch::Tensor&       success,
                                                          const std::vector<size_t>& stream_row_counts) {
    if (!success.defined()) {
        return {};
    }
    if (success.scalar_type() != torch::kBool) {
        throw std::invalid_argument("sampler success tensor must have bool dtype");
    }
    if (success.dim() != 1) {
        throw std::invalid_argument("sampler success tensor must be one-dimensional");
    }
    if (!success.is_contiguous()) {
        throw std::invalid_argument("sampler success tensor must be contiguous");
    }

    size_t expected_rows = 0;
    for (const size_t row_count : stream_row_counts) {
        if (row_count > std::numeric_limits<size_t>::max() - expected_rows) {
            throw std::invalid_argument("sampler success row count overflow");
        }
        expected_rows += row_count;
    }
    if (static_cast<size_t>(success.numel()) != expected_rows) {
        throw std::invalid_argument("sampler success row count does not match the executor batch");
    }

    const auto success_cpu = success.is_cuda() ? success.cpu() : success;
    const auto success_ptr = success_cpu.data_ptr<bool>();

    std::vector<size_t> failed_stream_indices;
    size_t              row_offset = 0;
    for (size_t stream_index = 0; stream_index < stream_row_counts.size(); ++stream_index) {
        bool stream_failed = false;
        for (size_t row = 0; row < stream_row_counts[stream_index]; ++row) {
            stream_failed |= !success_ptr[row_offset + row];
        }
        if (stream_failed) {
            failed_stream_indices.push_back(stream_index);
        }
        row_offset += stream_row_counts[stream_index];
    }
    return failed_stream_indices;
}

}  // namespace rtp_llm::speculative
