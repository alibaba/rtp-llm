#pragma once

#include <cstddef>
#include <cstdint>

#include <torch/torch.h>

namespace rtp_llm {

enum class ReplayIdDimRequirement {
    kAny,
    kOneDimensional,
};

enum class ReplayIdDeviceRequirement {
    kAny,
    kSameDevice,
};

inline constexpr ReplayIdDimRequirement    kBertReplayIdDimRequirement     = ReplayIdDimRequirement::kOneDimensional;
inline constexpr ReplayIdDeviceRequirement kBertReplayIdDeviceRequirement  = ReplayIdDeviceRequirement::kSameDevice;
inline constexpr ReplayIdDimRequirement    kComboReplayIdDimRequirement    = ReplayIdDimRequirement::kAny;
inline constexpr ReplayIdDeviceRequirement kComboReplayIdDeviceRequirement = ReplayIdDeviceRequirement::kAny;

inline bool satisfiesReplayIdDeviceRequirement(ReplayIdDeviceRequirement requirement,
                                               const c10::Device&        source,
                                               const c10::Device&        destination) {
    return requirement == ReplayIdDeviceRequirement::kAny || source == destination;
}

inline bool
satisfiesReplayIdDimRequirement(ReplayIdDimRequirement requirement, int64_t source_dim, int64_t destination_dim) {
    return requirement == ReplayIdDimRequirement::kAny || (source_dim == 1 && destination_dim == 1);
}

struct RequestOwnedMultimodalSignals {
    bool multimodal_features{false};
    bool multimodal_locs{false};
    bool multimodal_extra{false};
    bool text_tokens_mask{false};
};

inline bool hasRequestOwnedMultimodalSignals(const RequestOwnedMultimodalSignals& signals) {
    return signals.multimodal_features || signals.multimodal_locs || signals.multimodal_extra
           || signals.text_tokens_mask;
}

inline bool hasReplayIdBufferContract(const torch::Tensor&      source,
                                      const torch::Tensor&      destination,
                                      ReplayIdDimRequirement    dim_requirement,
                                      ReplayIdDeviceRequirement device_requirement) {
    if (!source.defined() || !source.has_storage() || source.numel() <= 0 || !destination.defined()
        || !destination.has_storage() || destination.numel() <= 0) {
        return false;
    }
    if (source.scalar_type() != torch::kInt32 || destination.scalar_type() != torch::kInt32 || !source.is_cuda()
        || !destination.is_cuda() || !source.is_contiguous() || !destination.is_contiguous()) {
        return false;
    }
    if (!satisfiesReplayIdDimRequirement(dim_requirement, source.dim(), destination.dim())) {
        return false;
    }
    if (!satisfiesReplayIdDeviceRequirement(device_requirement, source.device(), destination.device())) {
        return false;
    }
    return true;
}

inline bool validateReplayIdBufferForCopy(const torch::Tensor&      source,
                                          const torch::Tensor&      destination,
                                          size_t                    required_numel,
                                          ReplayIdDimRequirement    dim_requirement,
                                          ReplayIdDeviceRequirement device_requirement) {
    return required_numel > 0 && hasReplayIdBufferContract(source, destination, dim_requirement, device_requirement)
           && static_cast<size_t>(source.numel()) >= required_numel
           && static_cast<size_t>(destination.numel()) >= required_numel;
}

inline bool validateBertReplayIdBuffersForCopy(const torch::Tensor& source_position_ids,
                                               const torch::Tensor& destination_position_ids,
                                               const torch::Tensor& source_token_type_ids,
                                               const torch::Tensor& destination_token_type_ids,
                                               size_t               required_numel) {
    return validateReplayIdBufferForCopy(source_position_ids,
                                         destination_position_ids,
                                         required_numel,
                                         kBertReplayIdDimRequirement,
                                         kBertReplayIdDeviceRequirement)
           && validateReplayIdBufferForCopy(source_token_type_ids,
                                            destination_token_type_ids,
                                            required_numel,
                                            kBertReplayIdDimRequirement,
                                            kBertReplayIdDeviceRequirement);
}

inline bool hasBothBertEmbeddingTables(const torch::Tensor& position_encoding,
                                       const torch::Tensor& token_type_embedding) {
    return position_encoding.defined() && position_encoding.has_storage() && position_encoding.numel() > 0
           && token_type_embedding.defined() && token_type_embedding.has_storage() && token_type_embedding.numel() > 0;
}

inline bool shouldCaptureBertEmbeddingInputs(bool                 is_prefill_cuda_graph_mode,
                                             const torch::Tensor& position_encoding,
                                             const torch::Tensor& token_type_embedding) {
    return is_prefill_cuda_graph_mode && hasBothBertEmbeddingTables(position_encoding, token_type_embedding);
}

inline bool validateComboPositionIdsForReplay(int                  position_id_len_factor,
                                              int                  token_count,
                                              const torch::Tensor& position_ids,
                                              const torch::Tensor& captured_position_ids,
                                              size_t&              copy_numel) {
    copy_numel = 0;
    if (position_id_len_factor <= 0) {
        return true;
    }

    if (!position_ids.defined() || position_ids.numel() % position_id_len_factor != 0) {
        return false;
    }

    if (token_count <= 0) {
        return false;
    }
    const size_t required_numel = static_cast<size_t>(token_count) * static_cast<size_t>(position_id_len_factor);
    // Preserve legacy MRoPE kAny-dimension/kAny-device-index eligibility. This
    // predicate does not make a cross-device D2D copy valid; production callers
    // must provide tensors compatible with the selected graph device. Bert
    // replay IDs use the stricter one-dimensional/same-device contract.
    if (!validateReplayIdBufferForCopy(position_ids,
                                       captured_position_ids,
                                       required_numel,
                                       kComboReplayIdDimRequirement,
                                       kComboReplayIdDeviceRequirement)) {
        return false;
    }
    copy_numel = required_numel;
    return true;
}

}  // namespace rtp_llm
