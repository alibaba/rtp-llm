#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/rocm/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/rocm/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/common/CpuTpStatus.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("cpu_status_size", &cpu_status_size);
    rtp_ops_m.def("cpu_status_initialize", [](uintptr_t pointer, size_t bytes) {
        return cpu_status_initialize(reinterpret_cast<void*>(pointer), bytes);
    });
    rtp_ops_m.def("cpu_status_break",
                  [](uintptr_t pointer) { return cpu_status_break(reinterpret_cast<void*>(pointer)); });
    rtp_ops_m.def(
        "cpu_status_reduce",
        [](uintptr_t pointer, int rank, uint64_t generation, int local_ok, int timeout_ms) {
            return cpu_status_reduce(reinterpret_cast<void*>(pointer), rank, generation, local_ok, timeout_ms);
        },
        py::call_guard<py::gil_scoped_release>());

    registerBaseRocmBindings(rtp_ops_m);
    registerAttnOpBindings(rtp_ops_m);
}

}
