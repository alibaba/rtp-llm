#pragma once

#if USING_ROCM
#include <hip/hip_fp8.h>
typedef __hip_fp8_e4m3_fnuz __nv_fp8_e4m3;
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
