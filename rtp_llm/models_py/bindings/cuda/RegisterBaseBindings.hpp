#pragma once

#include "rtp_llm/models_py/bindings/cuda/RtpNorm.h"
#include "rtp_llm/models_py/bindings/cuda/RtpEmbeddingLookup.h"
#include "rtp_llm/models_py/bindings/cuda/FusedQKRmsNorm.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/models_py/bindings/cuda/FusedMoEOp.h"
#include "rtp_llm/models_py/bindings/cuda/SelectTopkOp.h"
#include "3rdparty/flashinfer/flashinfer.h"

using namespace rtp_llm;

namespace torch_ext {

void registerBasicCudaOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("rmsnorm",
                  &rmsnorm,
                  "RMSNorm kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("eps"),
                  py::arg("cuda_stream") = 0);

    rtp_ops_m.def("fused_add_rmsnorm",
                  &fused_add_rmsnorm,
                  "Fused Add RMSNorm kernel",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("weight"),
                  py::arg("eps"),
                  py::arg("cuda_stream") = 0);

    rtp_ops_m.def("silu_and_mul",
                  &silu_and_mul,
                  "SiLU and Multiply kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("cuda_stream") = 0);

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

    rtp_ops_m.def("layernorm",
                  &layernorm,
                  "LayerNorm kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"));

    rtp_ops_m.def("fused_add_layernorm",
                  &fused_add_layernorm,
                  "Fused Add LayerNorm kernel",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("bias"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"));

    rtp_ops_m.def(
        "embedding", &embedding, "Embedding lookup kernel", py::arg("output"), py::arg("input"), py::arg("weight"));
}

void registerBaseCudaBindings(py::module& rtp_ops_m) {
    registerBasicCudaOps(rtp_ops_m);
    registerFusedMoEOp(rtp_ops_m);
    registerSelectTopkOp(rtp_ops_m);
}

}  // namespace torch_ext
