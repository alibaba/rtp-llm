#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/cuda/cuda_utils.h"

template<>
struct c10::CppTypeToScalarType<half>: std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};

template<>
struct c10::CppTypeToScalarType<__nv_bfloat16>: std::integral_constant<c10::ScalarType, c10::ScalarType::BFloat16> {};
