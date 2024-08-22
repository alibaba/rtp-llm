#pragma once

#include <torch/torch.h>
#include "src/fastertransformer/devices/testing/TestBase.h"

using half = arm_compute::half;
template <>
struct c10::CppTypeToScalarType<half>
    : std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};

template <>
struct c10::CppTypeToScalarType<float16_t>
    : std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};
