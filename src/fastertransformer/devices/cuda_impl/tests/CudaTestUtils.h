#pragma once

#include <torch/torch.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

template <>
struct c10::CppTypeToScalarType<half>
    : std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};
