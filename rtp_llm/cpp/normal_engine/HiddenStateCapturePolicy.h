#pragma once

#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/config/RoleTypes.h"

namespace rtp_llm {

enum class HiddenStateCaptureModelRole {
    TARGET,
    DRAFT,
};

inline std::vector<int64_t> selectHiddenStateCaptureLayerIds(HiddenStateCaptureModelRole model_role,
                                                             RoleType                    role_type,
                                                             bool                        warm_up,
                                                             const std::vector<int64_t>& configured_layer_ids) {
    const bool is_target_prefill = model_role == HiddenStateCaptureModelRole::TARGET
                                   && (role_type == RoleType::PREFILL || role_type == RoleType::PDFUSION);
    if (warm_up || !is_target_prefill) {
        return {};
    }
    return configured_layer_ids;
}

}  // namespace rtp_llm
