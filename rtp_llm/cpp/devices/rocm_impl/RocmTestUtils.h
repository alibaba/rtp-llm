#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

template<>
struct c10::CppTypeToScalarType<half>: std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};
