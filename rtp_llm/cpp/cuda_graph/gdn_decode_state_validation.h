#pragma once

#include <string>

#include <torch/torch.h>

namespace rtp_llm {

inline std::string validateGdnDecodeStateBlockTable(const torch::Tensor& block_map,
                                                    const torch::Tensor& sequence_lengths,
                                                    int64_t              real_batch_size,
                                                    int64_t              seq_size_per_block,
                                                    int64_t              state_pool_size) {
    if (state_pool_size <= 0) {
        return {};
    }
    if (!block_map.defined() || block_map.is_cuda() || block_map.dim() != 2
        || block_map.scalar_type() != torch::kInt32 || block_map.stride(1) != 1) {
        return "GDN decode host block table must be a column-contiguous CPU int32 matrix";
    }
    if (!sequence_lengths.defined() || sequence_lengths.is_cuda() || sequence_lengths.dim() != 1
        || sequence_lengths.scalar_type() != torch::kInt32 || sequence_lengths.stride(0) != 1) {
        return "GDN decode sequence lengths must be a contiguous CPU int32 vector";
    }
    if (real_batch_size < 0 || real_batch_size > block_map.size(0)
        || real_batch_size > sequence_lengths.numel()) {
        return "GDN decode real batch exceeds host metadata rows";
    }
    if (seq_size_per_block <= 0) {
        return "GDN decode sequence block size must be positive";
    }

    const auto* lengths = sequence_lengths.data_ptr<int32_t>();
    const auto* blocks  = block_map.data_ptr<int32_t>();
    for (int64_t row = 0; row < real_batch_size; ++row) {
        const int64_t sequence_length = lengths[row];
        if (sequence_length < 1) {
            return "GDN decode real request has a non-positive sequence length at row " + std::to_string(row);
        }
        const int64_t read_pos  = (sequence_length - 1) / seq_size_per_block;
        const int64_t write_pos = sequence_length / seq_size_per_block;
        if (read_pos >= block_map.size(1) || write_pos >= block_map.size(1)) {
            return "GDN decode real request exceeds the host block-table width at row " + std::to_string(row);
        }

        const auto row_offset = row * block_map.stride(0);
        const auto read_id    = blocks[row_offset + read_pos];
        const auto write_id   = blocks[row_offset + write_pos];
        if (read_id <= 0 || write_id <= 0 || read_id >= state_pool_size || write_id >= state_pool_size) {
            return "GDN decode real request has an invalid state block ID at row " + std::to_string(row);
        }
    }
    return {};
}

}  // namespace rtp_llm
