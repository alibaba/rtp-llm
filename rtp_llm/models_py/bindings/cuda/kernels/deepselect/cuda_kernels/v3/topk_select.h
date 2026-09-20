#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/structs.h"

namespace topk_select_bf16_normal {

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs& args);

}  // namespace topk_select_bf16_normal
