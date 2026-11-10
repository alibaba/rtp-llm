// Copyright 2026 Alibaba Group Holding Limited.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef USE_PPU
#include <torch/extension.h>
namespace rtp_llm {
std::tuple<torch::Tensor, torch::Tensor> PpuSiluAndMulPostQuantMxfp4(
    torch::Tensor gate_up, double swiglu_limit, bool apply_swiglu_limit);
}  // namespace rtp_llm
#endif
