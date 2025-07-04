#include "rtp_llm/models_py/bindings/rocm/Norm.h"

using namespace rtp_llm;

namespace torch_ext {

void registerBasicRocmOps(py::module &rtp_ops_m) {
    rtp_ops_m.def("layernorm", &layernorm, "LayerNorm kernel",
        py::arg("output"),
        py::arg("input"),
        py::arg("residual"),
        py::arg("weight"),
        py::arg("beta"),
        py::arg("eps"),
        py::arg("hip_stream") = 0);
}

void registerBaseRocmBindings(py::module &rtp_ops_m) {
    registerBasicRocmOps(rtp_ops_m);
}

}
