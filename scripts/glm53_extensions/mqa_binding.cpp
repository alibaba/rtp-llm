// FP8 indexer ABI, isolated from the process-wide DeepGEMM installation.
#include <pybind11/pybind11.h>
#include <torch/python.h>
#include "csrc/apis/attention.hpp"
#include "csrc/apis/runtime.hpp"

PYBIND11_MODULE(_rtp_llm_fp8_mqa, m) {
  deep_gemm::runtime::register_apis(m);
  m.def("fp8_mqa_logits", &deep_gemm::attention::fp8_mqa_logits,
        py::arg("q"), py::arg("kv"), py::arg("weights"),
        py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
        py::arg("clean_logits") = true, py::arg("max_seqlen_k") = 0);
  m.def("get_paged_mqa_logits_metadata",
        &deep_gemm::attention::get_paged_mqa_logits_metadata,
        py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"),
        py::arg("indices") = std::nullopt);
  m.def("fp8_paged_mqa_logits", &deep_gemm::attention::fp8_paged_mqa_logits,
        py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
        py::arg("context_lens"), py::arg("block_table"),
        py::arg("schedule_meta"), py::arg("max_context_len"),
        py::arg("clean_logits") = false, py::arg("indices") = std::nullopt);
}
