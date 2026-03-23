#include "rtp_llm/models_py/bindings/cuda/CheckAndResetNanKvCacheOp.h"

#include "rtp_llm/cpp/kernels/nan_check_torch_op.h"

namespace rtp_llm {

void registerCheckAndResetNanKvCacheOp(py::module& m) {
    m.def("check_and_reset_nan_kv_cache_decode",
          &check_and_reset_nan_kv_cache_decode,
          "Check and reset NaN/Inf in KV cache for decode (last token per batch).",
          py::arg("layer_base_addrs"),
          py::arg("kv_cache_block_id"),
          py::arg("sequence_lengths"),
          py::arg("nan_flag"),
          py::arg("cache_dtype"),
          py::arg("batch_size"),
          py::arg("layer_num"),
          py::arg("num_groups"),
          py::arg("layer_to_group"),
          py::arg("group_types"),
          py::arg("batch_dim"),
          py::arg("batch_start"),
          py::arg("max_blocks_per_batch"),
          py::arg("local_head_num_kv"),
          py::arg("k_token_size"),
          py::arg("v_token_size"),
          py::arg("k_block_size_bytes"),
          py::arg("v_block_size_bytes"),
          py::arg("k_token_bytes"),
          py::arg("v_token_bytes"),
          py::arg("block_size_bytes"),
          py::arg("seq_size_per_block"));

    m.def("check_and_reset_nan_kv_cache_prefill",
          &check_and_reset_nan_kv_cache_prefill,
          "Check and reset NaN/Inf in KV cache for prefill (prefix_lengths..input_lengths per batch).",
          py::arg("layer_base_addrs"),
          py::arg("kv_cache_block_id"),
          py::arg("prefix_lengths"),
          py::arg("input_lengths"),
          py::arg("nan_flag"),
          py::arg("cache_dtype"),
          py::arg("batch_size"),
          py::arg("layer_num"),
          py::arg("num_groups"),
          py::arg("layer_to_group"),
          py::arg("group_types"),
          py::arg("batch_dim"),
          py::arg("batch_start"),
          py::arg("max_blocks_per_batch"),
          py::arg("local_head_num_kv"),
          py::arg("k_token_size"),
          py::arg("v_token_size"),
          py::arg("k_block_size_bytes"),
          py::arg("v_block_size_bytes"),
          py::arg("k_token_bytes"),
          py::arg("v_token_bytes"),
          py::arg("block_size_bytes"),
          py::arg("seq_size_per_block"));
}

}  // namespace rtp_llm
