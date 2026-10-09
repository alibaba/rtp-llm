#pragma once

#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/kerutils/include/kerutils/common/common.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "cuda/common.h"
#include "cuda/sm80/intrinsics.cuh"
#include "cuda/sm80/helpers.cuh"
#include "cuda/sm90/intrinsics.cuh"
#include "cuda/sm100/intrinsics.cuh"
#endif
