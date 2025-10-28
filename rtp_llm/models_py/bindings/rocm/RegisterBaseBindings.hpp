#include "rtp_llm/models_py/bindings/rocm/Norm.h"
#include "rtp_llm/models_py/bindings/common/RtpEmbeddingLookup.h"
#include "rtp_llm/models_py/bindings/common/FusedQKRmsNorm.h"
#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/rocm/Gemm.h"
#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/rocm/RtpProcessGroup.h"

using namespace rtp_llm;

namespace torch_ext {

void registerBasicRocmOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("write_cache_store",
                  &WriteCacheStoreOp,
                  "WriteCacheStoreOp kernel",
                  py::arg("input_lengths"),
                  py::arg("prefix_lengths"),
                  py::arg("kv_cache_block_id_host"),
                  py::arg("cache_store_member"),
                  py::arg("kv_cache"));

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
    registerRtpProcessGroup(rtp_ops_m);
}

}  // namespace torch_ext
