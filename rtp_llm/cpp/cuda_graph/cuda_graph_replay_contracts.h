#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace rtp_llm {

struct PrefillMoeSelfCheck {
    bool   passed          = false;
    double eager_relative  = 0;
    double graph_relative  = 0;
    double graph_row_max   = 0;
};

// Only for the attested ROCm atomic-accumulation MoE implementation. Routing
// amplifies BF16 accumulation noise across layers, so whole-model allclose is
// not an appropriate capture check. Compare independent repeats to an eager
// envelope instead. This is a startup compatibility check, not a golden
// accuracy test or a determinism guarantee. Stable eager outputs still have
// a tight tolerance, and a bad token cannot hide in a batch-wide norm.
inline PrefillMoeSelfCheck checkPrefillMoeReplay(const std::vector<torch::Tensor>& eager_outputs,
                                               const std::vector<torch::Tensor>& graph_outputs) {
    PrefillMoeSelfCheck result;
    if (eager_outputs.size() < 2 || graph_outputs.size() < 2) {
        return result;
    }
    const auto& reference = eager_outputs.front();
    if (!reference.defined() || reference.dim() != 2 || reference.numel() == 0) {
        return result;
    }
    auto valid = [&](const torch::Tensor& value) {
        return value.defined() && value.sizes() == reference.sizes()
               && value.scalar_type() == reference.scalar_type() && value.device() == reference.device()
               && torch::isfinite(value).all().item<bool>();
    };
    for (const auto& value : eager_outputs) {
        if (!valid(value)) {
            return result;
        }
    }
    for (const auto& value : graph_outputs) {
        if (!valid(value)) {
            return result;
        }
    }
    auto eager      = torch::stack(eager_outputs).to(torch::kFloat32);
    auto graph      = torch::stack(graph_outputs).to(torch::kFloat32);
    auto center     = eager.mean(0);
    auto eager_diff = eager - center;
    auto graph_diff = graph - center;
    auto norm       = center.norm().clamp_min(1e-6);
    result.eager_relative = (eager_diff.flatten(1).norm(2, 1).max() / norm).item<double>();
    result.graph_relative = (graph_diff.flatten(1).norm(2, 1).max() / norm).item<double>();
    auto center_row_norm = center.norm(2, 1);
    auto eager_radius = std::get<0>(eager_diff.norm(2, 2).max(0));
    auto graph_radius = std::get<0>(graph_diff.norm(2, 2).max(0));
    result.graph_row_max = (graph_radius / center_row_norm.clamp_min(1e-6)).max().item<double>();
    // The larger eager sample estimates normal routing variation; graph
    // repeats may exceed it by 50% globally and 2x for individual tokens.
    // The additive floor only covers roundoff when eager is stable.
    auto row_limit = 2 * eager_radius + 1e-3 * (center_row_norm + 1);
    auto min_eager_norm = std::get<0>(eager.norm(2, 2).min(0));
    auto max_eager_norm = std::get<0>(eager.norm(2, 2).max(0));
    auto graph_norm = graph.norm(2, 2);
    const bool valid_norms = (graph_norm >= 0.5 * min_eager_norm - 1e-3).all().item<bool>()
                             && (graph_norm <= 2 * max_eager_norm + 1e-3).all().item<bool>();
    result.passed = result.graph_relative <= 1.5 * result.eager_relative + 1e-3
                    && (graph_radius <= row_limit).all().item<bool>() && valid_norms;
    return result;
}

// Normalize a dynamic packed prefill batch to the fixed graph contract:
// Bmax real slots followed by one sentinel slot, with exactly token_capacity
// packed tokens. The first real_request_count entries of input_lengths must
// already contain the live request lengths. All output buffers are caller
// owned and therefore keep stable addresses across CUDA graph replays.
inline bool prepareFullPrefillReplayMetadata(int32_t* input_lengths,
                                             int32_t* cu_seqlens,
                                             int32_t* padding_offset,
                                             int      real_request_count,
                                             int      max_request_count,
                                             int      real_token_count,
                                             int      token_capacity) {
    if (input_lengths == nullptr || cu_seqlens == nullptr || padding_offset == nullptr || real_request_count <= 0
        || real_request_count > max_request_count || max_request_count <= 0 || real_token_count <= 0
        || real_token_count > token_capacity) {
        return false;
    }

    int computed_real_tokens = 0;
    for (int slot = 0; slot < real_request_count; ++slot) {
        if (input_lengths[slot] <= 0 || input_lengths[slot] > token_capacity - computed_real_tokens) {
            return false;
        }
        computed_real_tokens += input_lengths[slot];
    }
    if (computed_real_tokens != real_token_count) {
        return false;
    }

    for (int slot = real_request_count; slot < max_request_count; ++slot) {
        input_lengths[slot] = 0;
    }
    input_lengths[max_request_count] = token_capacity - real_token_count;

    int packed_offset = 0;
    cu_seqlens[0]     = 0;
    for (int slot = 0; slot <= max_request_count; ++slot) {
        const int length       = input_lengths[slot];
        const int fixed_offset = slot * token_capacity - packed_offset;
        for (int token = 0; token < length; ++token) {
            padding_offset[packed_offset + token] = fixed_offset;
        }
        packed_offset += length;
        cu_seqlens[slot + 1] = packed_offset;
    }
    return packed_offset == token_capacity;
}

enum class ReplayIdDimRequirement {
    kAny,
    kOneDimensional,
};

enum class ReplayIdDeviceRequirement {
    kAny,
    kSameDevice,
};

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
    if (dim_requirement == ReplayIdDimRequirement::kOneDimensional && (source.dim() != 1 || destination.dim() != 1)) {
        return false;
    }
    if (device_requirement == ReplayIdDeviceRequirement::kSameDevice && source.device() != destination.device()) {
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
    constexpr auto dim_requirement    = ReplayIdDimRequirement::kOneDimensional;
    constexpr auto device_requirement = ReplayIdDeviceRequirement::kSameDevice;
    return validateReplayIdBufferForCopy(
               source_position_ids, destination_position_ids, required_numel, dim_requirement, device_requirement)
           && validateReplayIdBufferForCopy(
               source_token_type_ids, destination_token_type_ids, required_numel, dim_requirement, device_requirement);
}

inline bool hasBothBertEmbeddingTables(const torch::Tensor& position_encoding,
                                       const torch::Tensor& token_type_embedding) {
    return position_encoding.defined() && position_encoding.has_storage() && position_encoding.numel() > 0
           && token_type_embedding.defined() && token_type_embedding.has_storage() && token_type_embedding.numel() > 0;
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

    if (!hasReplayIdBufferContract(
            position_ids, captured_position_ids, ReplayIdDimRequirement::kAny, ReplayIdDeviceRequirement::kAny)
        || position_ids.numel() % position_id_len_factor != 0) {
        return false;
    }

    if (token_count <= 0) {
        return false;
    }
    const size_t required_numel = static_cast<size_t>(token_count) * static_cast<size_t>(position_id_len_factor);
    // Legacy mrope callers may provide multi-dimensional buffers and rely on
    // the copy path, rather than this eligibility predicate, to select the
    // active CUDA device. Preserve that historical kAny/kAny contract here;
    // Bert replay IDs use the stricter one-dimensional/same-device contract.
    if (static_cast<size_t>(position_ids.numel()) < required_numel
        || static_cast<size_t>(captured_position_ids.numel()) < required_numel) {
        return false;
    }
    copy_numel = required_numel;
    return true;
}

}  // namespace rtp_llm
