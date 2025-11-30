#include "rtp_llm/models_py/bindings/rocm/Norm.h"
#include "rtp_llm/models_py/bindings/common/RtpEmbeddingLookup.h"
#include "rtp_llm/models_py/bindings/common/FusedQKRmsNorm.h"
#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/rocm/Gemm.h"
#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/common/CudaGraphPrefillCopy.h"

namespace rtp_llm {

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

    // CUDA Graph Copy Kernel Functions (also supported in ROCm)
    rtp_ops_m.def("cuda_graph_copy_small2large",
                  &cuda_graph_copy_small2large,
                  "CUDA Graph copy kernel: Small to Large tensor copy",
                  py::arg("input_tensor"),
                  py::arg("output_tensor"),
                  py::arg("batch_size"),
                  py::arg("max_batch_size"),
                  py::arg("max_seq_len"),
                  py::arg("input_lengths"),
                  py::arg("hidden_size"),
                  py::arg("cu_seq_len"));

    rtp_ops_m.def("cuda_graph_copy_large2small",
                  &cuda_graph_copy_large2small,
                  "CUDA Graph copy kernel: Large to Small tensor copy",
                  py::arg("input_tensor"),
                  py::arg("output_tensor"),
                  py::arg("batch_size"),
                  py::arg("max_batch_size"),
                  py::arg("max_seq_len"),
                  py::arg("input_lengths"),
                  py::arg("hidden_size"),
                  py::arg("cu_seq_len"));
}

void registerBaseRocmBindings(py::module& rtp_ops_m) {
    registerBasicRocmOps(rtp_ops_m);
    // RtpProcessGroup is deprecated, use rtp_llm.distribute.collective_torch instead
    // registerRtpProcessGroup(rtp_ops_m);
}

}  // namespace rtp_llm
