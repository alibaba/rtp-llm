#include "rtp_llm/models_py/bindings/rocm/Norm.h"
#include "rtp_llm/models_py/bindings/common/RtpEmbeddingLookup.h"
#include "rtp_llm/models_py/bindings/common/FusedQKRmsNorm.h"
#include "rtp_llm/models_py/bindings/rocm/Gemm.h"

using namespace rtp_llm;

namespace torch_ext {

void registerBasicRocmOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("fused_add_layernorm",
                  &fused_add_layernorm,
                  "Fused Add LayerNorm kernel",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("bias"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"),
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("layernorm",
                  &layernorm,
                  "LayerNorm kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"),
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def(
        "embedding", &embedding, "Embedding lookup kernel", py::arg("output"), py::arg("input"), py::arg("weight"));

    rtp_ops_m.def("fused_qk_rmsnorm",
                  &FusedQKRMSNorm,
                  "Fused QK RMSNorm kernel",
                  py::arg("IO"),
                  py::arg("q_gamma"),
                  py::arg("k_gamma"),
                  py::arg("layernorm_eps"),
                  py::arg("q_group_num"),
                  py::arg("k_group_num"),
                  py::arg("m"),
                  py::arg("n"),
                  py::arg("norm_size"));

    rtp_ops_m.def("gemm", &gemm, "Gemm kernel", py::arg("output"), py::arg("input"), py::arg("weight"));
}

void registerBaseRocmBindings(py::module& rtp_ops_m) {
    registerBasicRocmOps(rtp_ops_m);
}

}  // namespace torch_ext
