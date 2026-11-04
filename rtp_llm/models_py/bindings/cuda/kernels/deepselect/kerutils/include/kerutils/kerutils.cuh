#pragma once

#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/kerutils/include/kerutils/common/common.h"

#include "host/host.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "device/device.cuh"
#endif
