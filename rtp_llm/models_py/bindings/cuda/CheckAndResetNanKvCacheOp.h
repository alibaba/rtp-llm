#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace rtp_llm {

// Register check_and_reset_nan_kv_cache_* to py::module (called from RegisterBaseBindings).
// Implementation lives in nan_check_torch_op; same symbols are used for TORCH_LIBRARY and pybind11.
void registerCheckAndResetNanKvCacheOp(py::module& m);

}  // namespace rtp_llm
