/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    GeluNoneApproximate,
    GeGluNoneApproximate,
    Sigmoid,
    InvalidType
};

inline ActivationType getActivationType(std::string activation_type_str) {
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return ActivationType::Gelu;
    } else if (activation_type_str == "GeluNoneApproximate" || activation_type_str == "gelu-none-approximate") {
        return ActivationType::GeluNoneApproximate;
    } else if (activation_type_str == "GeGLU" || activation_type_str == "geglu"
               || activation_type_str == "gated-gelu") {
        return ActivationType::Geglu;
    } else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return ActivationType::Relu;
    } else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return ActivationType::Silu;
    } else if (activation_type_str == "SiGLU" || activation_type_str == "gated-silu") {
        return ActivationType::Swiglu;
    } else if (activation_type_str == "GeGluNoneApproximate" || activation_type_str == "geglu-none-approximate") {
        return ActivationType::GeGluNoneApproximate;
    } else {
        RTP_LLM_FAIL("Activation Type: " + activation_type_str + " not supported !");
    }
    return ActivationType::InvalidType;
}

inline bool isGatedActivation(ActivationType activaiton_type) {
    return activaiton_type == ActivationType::Geglu || activaiton_type == ActivationType::Swiglu
           || activaiton_type == ActivationType::GeGluNoneApproximate;
}

}  // namespace rtp_llm
