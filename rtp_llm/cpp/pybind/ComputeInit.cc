#include <torch/library.h>
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#if USING_CUDA || USING_ROCM
#endif

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);
using namespace torch_ext;

PYBIND11_MODULE(librtp_compute_ops, m) {
#if USING_CUDA || USING_ROCM
    registerExecCtxOps(m);
#endif

    registerPyOpDefs(m);

    py::module rtp_ops_m = m.def_submodule("rtp_llm_ops", "rtp llm custom ops");
    registerPyModuleOps(rtp_ops_m);
}

}  // namespace rtp_llm
