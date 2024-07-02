#pragma once

#include <torch/torch.h>
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/rocm/hip_utils.h"

template <>
struct c10::CppTypeToScalarType<half>
    : std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};
