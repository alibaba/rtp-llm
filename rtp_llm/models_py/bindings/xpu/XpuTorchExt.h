#pragma once
// XpuTorchExt.h — XPU-specific includes and macros for rtp_llm_ops bindings.
// Equivalent of common/Torch_ext.h but without CUDA/HIP dependencies.

#include <vector>
#include <torch/extension.h>
#include <torch/all.h>

namespace py = pybind11;
