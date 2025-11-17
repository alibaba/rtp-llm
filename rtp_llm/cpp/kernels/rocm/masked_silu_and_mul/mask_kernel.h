#pragma once

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bfloat16.h>
#else
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>

#include "rtp_llm/cpp/kernels/rocm/masked_silu_and_mul/utils/array_utils.h"


namespace rtp_llm {
using utils::fp8_e4m3_t;
#ifndef ACTIVATION_THREADS_PER_BLOCK
#define ACTIVATION_THREADS_PER_BLOCK 256
#endif
extern "C"
void launch_doActivationMaskedKernelHIP(fp8_e4m3_t*       output,
                                        float*            output_fp8_scale,
                                        const hip_bfloat16* gemm_result,
                                        int64_t           expert_num,
                                        int64_t           token_num,
                                        int64_t           inter_size,
                                        bool              gated,
                                        const int*        masked_m,
                                        hipStream_t       stream);
}  // namespace rtp_llm


