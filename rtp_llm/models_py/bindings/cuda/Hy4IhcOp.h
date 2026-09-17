// Copyright 2026 Tencent
// HY4 iHC AOT binding adapted from hpc-ops (MIT).
#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

#include <tuple>

namespace torch_ext {

at::Tensor fuse_hy4_ihc_head(const at::Tensor& channels,
                             const at::Tensor& fn_weight,
                             const at::Tensor& scale,
                             const at::Tensor& base,
                             double norm_eps,
                             double hc_eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> fuse_hy4_ihc_post_pre(
    const at::Tensor& block_output,
    const at::Tensor& residual,
    const at::Tensor& post_gate,
    const at::Tensor& next_fn_weight,
    const at::Tensor& next_scale,
    const at::Tensor& next_base,
    double ihc_norm_eps,
    double hc_eps,
    double magnitude,
    const at::Tensor& rms_weight,
    double rms_eps,
    bool cast_bfloat_for_norm = true);

}  // namespace torch_ext
