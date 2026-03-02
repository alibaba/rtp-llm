#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/dcu/Gemm.h"
#include "rtp_llm/models_py/bindings/common/CudaGraphPrefillCopy.h"
#include "rtp_llm/models_py/bindings/common/RtpEmbeddingLookup.h"

namespace rtp_llm {

void registerBasicDcuOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("write_cache_store",
                  &WriteCacheStoreOp,
                  "WriteCacheStoreOp kernel",
                  py::arg("input_lengths"),
                  py::arg("prefix_lengths"),
                  py::arg("kv_cache_block_id_host"),
                  py::arg("cache_store_member"),
                  py::arg("kv_cache"));

    rtp_ops_m.def("gemm", &gemm, "Gemm kernel", py::arg("output"), py::arg("input"), py::arg("weight"));

    rtp_ops_m.def("embedding", 
		  &embedding, 
		  "Embedding lookup kernel", 
		  py::arg("output"), 
		  py::arg("input"), 
		  py::arg("weight"),
                  py::arg("position_ids")     = py::none(),
                  py::arg("token_type_ids")   = py::none(),
                  py::arg("text_tokens_mask") = py::none());

    rtp_ops_m.def("embedding_bert",
                  &embeddingBert,
                  "EmbeddingBert lookup kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("combo_position_ids"),
                  py::arg("position_encoding"),
                  py::arg("combo_tokens_type_ids"),
                  py::arg("token_type_embedding"),
                  py::arg("input_embedding_scalar") = 1.0f);

    // CUDA Graph Copy Kernel Functions (also supported in ROCm)
    rtp_ops_m.def("cuda_graph_copy_small2large",
                  &torch_ext::cuda_graph_copy_small2large,
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
                  &torch_ext::cuda_graph_copy_large2small,
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

void registerBaseDcuBindings(py::module& rtp_ops_m) {
    registerBasicDcuOps(rtp_ops_m);
    // RtpProcessGroup is deprecated, use rtp_llm.distribute.collective_torch instead
    // registerRtpProcessGroup(rtp_ops_m);
}

}  // namespace rtp_llm
