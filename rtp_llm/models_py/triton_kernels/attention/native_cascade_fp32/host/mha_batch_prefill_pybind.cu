// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "torch/mha_batch_prefill.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHA_BATCH_PREFILL_PYBIND;
}