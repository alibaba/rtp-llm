// SPDX-License-Identifier: MIT
 // Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 
 #include "mha_common.h"
 
 namespace aiter {
 __global__ void ParsePhiloxCudaState(at::PhiloxCudaState arg, uint64_t* rng_state)
 {
     // Imitate from PyTorch
     // https://github.com/pytorch/pytorch/blob/8b61daaf7349e9102117e1aeefaa51666d887547/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L17
     if (arg.captured_) {
         rng_state[0] = static_cast<uint64_t>(*arg.seed_.ptr);
         rng_state[1] = static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_);
     } else {
         rng_state[0] = arg.seed_.val;
         rng_state[1] = arg.offset_.val;
     }
 }
 
 } // namespace aiter