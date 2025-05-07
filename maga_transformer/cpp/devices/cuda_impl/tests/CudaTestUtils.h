#pragma once

#include <torch/torch.h>
#include "maga_transformer/cpp/devices/testing/TestBase.h"
#include "maga_transformer/cpp/cuda/cuda_utils.h"

template <>
struct c10::CppTypeToScalarType<half>
    : std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};
