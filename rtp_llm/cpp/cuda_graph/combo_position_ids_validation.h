#pragma once

#include <cstddef>

#include <torch/torch.h>

namespace rtp_llm {

inline bool validateComboPositionIdsForReplay(int                  position_id_len_factor,
                                              int                  token_count,
                                              const torch::Tensor& position_ids,
                                              const torch::Tensor& captured_position_ids,
                                              size_t&              copy_numel) {
    copy_numel = 0;
    if (position_id_len_factor <= 0) {
        return true;
    }

    if (!position_ids.defined() || !position_ids.has_storage() || position_ids.numel() <= 0
        || !captured_position_ids.defined() || !captured_position_ids.has_storage()
        || captured_position_ids.numel() <= 0) {
        return false;
    }
    if (position_ids.scalar_type() != torch::kInt32 || captured_position_ids.scalar_type() != torch::kInt32
        || !position_ids.is_cuda() || !captured_position_ids.is_cuda() || !position_ids.is_contiguous()
        || !captured_position_ids.is_contiguous() || position_ids.numel() % position_id_len_factor != 0) {
        return false;
    }

    if (token_count <= 0) {
        return false;
    }
    copy_numel = static_cast<size_t>(token_count) * static_cast<size_t>(position_id_len_factor);
    return static_cast<size_t>(position_ids.numel()) >= copy_numel
           && static_cast<size_t>(captured_position_ids.numel()) >= copy_numel;
}

}  // namespace rtp_llm
