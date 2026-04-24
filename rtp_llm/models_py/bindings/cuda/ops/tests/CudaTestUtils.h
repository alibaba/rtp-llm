#pragma once

#include <torch/torch.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

template<>
struct c10::CppTypeToScalarType<half>: std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};

template<>
struct c10::CppTypeToScalarType<__nv_bfloat16>: std::integral_constant<c10::ScalarType, c10::ScalarType::BFloat16> {};
