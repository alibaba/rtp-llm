#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <string>

namespace rtp_llm {

enum class LayerNormType {
    pre_layernorm,
    post_layernorm,
    invalid_type
};

enum class NormType {
    layernorm,
    rmsnorm,
    alphanorm,
    add_bias,
    invalid_type
};

inline LayerNormType getLayerNormType(std::string layernorm_type_str) {
    if (layernorm_type_str == "pre_layernorm") {
        return LayerNormType::pre_layernorm;
    } else if (layernorm_type_str == "post_layernorm") {
        return LayerNormType::post_layernorm;
    } else {
        RTP_LLM_FAIL("Layernorm Type: " + layernorm_type_str + " not supported !");
    }
    return LayerNormType::invalid_type;
}

inline NormType getNormType(std::string norm_type_str) {
    if (norm_type_str == "layernorm") {
        return NormType::layernorm;
    } else if (norm_type_str == "rmsnorm") {
        return NormType::rmsnorm;
    } else if (norm_type_str == "alphanorm") {
        return NormType::alphanorm;
    } else {
        RTP_LLM_FAIL("Norm Type: " + norm_type_str + " not supported !");
    }
    return NormType::invalid_type;
}

inline std::string getLayerNormTypeStr(LayerNormType layernorm_type) {
    switch (layernorm_type) {
        case LayerNormType::pre_layernorm:
            return "pre_layernorm";
        case LayerNormType::post_layernorm:
            return "post_layernorm";
        default:
            throw std::runtime_error("Invalid LayerNormType: " + std::to_string(static_cast<int>(layernorm_type)));
    }
}

inline std::string getNormTypeStr(NormType norm_type) {
    switch (norm_type) {
        case NormType::layernorm:
            return "layernorm";
        case NormType::rmsnorm:
            return "rmsnorm";
        case NormType::alphanorm:
            return "alphanorm";
        case NormType::add_bias:
            return "add_bias";
        default:
            throw std::runtime_error("Invalid NormType: " + std::to_string(static_cast<int>(norm_type)));
    }
}

}  // namespace rtp_llm
