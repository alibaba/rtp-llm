#include <torch/library.h>
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/comm/CollectiveBackend.h"
#include "rtp_llm/models_py/bindings/core/WeightPreprocess.h"
#endif
#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

namespace py = pybind11;

namespace rtp_llm {
using namespace torch_ext;

#if USING_CUDA || USING_ROCM
static void registerRuntimeBindings(py::module& m) {
    m.def("get_device_id", &getDeviceId);
    m.def("preprocess_gemm_weight_by_key",
          &preprocessGemmWeightByKey,
          py::arg("key"),
          py::arg("weight"),
          py::arg("user_arm_gemm_use_kai"));
    m.def("preprocess_weight_scale", &preprocessWeightScale, py::arg("weight"), py::arg("scale"));

    m.def(
        "init_exec_ctx",
        [](std::size_t device_id, bool trace_memory, bool enable_comm_overlap, int mla_ops_type) {
            (void)initRuntime(device_id, trace_memory, enable_comm_overlap, static_cast<MlaOpsType>(mla_ops_type));
        },
        py::arg("device_id"),
        py::arg("trace_memory"),
        py::arg("enable_comm_overlap"),
        py::arg("mla_ops_type"));

    registerCommPybindings(m);
}
#endif

PYBIND11_MODULE(librtp_compute_ops, m) {
#if USING_CUDA || USING_ROCM
    registerRuntimeBindings(m);
#endif

    registerPyOpDefs(m);

    py::module rtp_ops_m = m.def_submodule("rtp_llm_ops", "rtp llm custom ops");
    registerPyModuleOps(rtp_ops_m);
}

}  // namespace rtp_llm
