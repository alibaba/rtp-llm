#include <torch/library.h>
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/cpp/kernels/moe/layout_convert.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

PYBIND11_MODULE(librtp_compute_ops, m) {

    registerDeviceOps(m);
    registerPyOpDefs(m);

    py::module rtp_ops_m = m.def_submodule("rtp_llm_ops", "rtp llm custom ops");
    registerPyModuleOps(rtp_ops_m);
}

}  // namespace rtp_llm
