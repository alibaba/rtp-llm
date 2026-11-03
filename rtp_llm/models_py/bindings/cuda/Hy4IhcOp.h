// Copyright 2026 Tencent
// HY4 iHC AOT binding adapted from hpc-ops (MIT).
#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace torch_ext {

at::Tensor fuse_hy4_ihc_head(const at::Tensor& channels,
                             const at::Tensor& fn_weight,
                             const at::Tensor& scale,
                             const at::Tensor& base,
                             double norm_eps,
                             double hc_eps);

}  // namespace torch_ext
