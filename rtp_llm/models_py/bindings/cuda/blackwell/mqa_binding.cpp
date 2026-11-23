// FP8 indexer ABI, isolated from the process-wide DeepGEMM installation.
#include <pybind11/pybind11.h>
#include <torch/python.h>
#include "csrc/apis/attention.hpp"
#include "mqa_runtime.h"

PYBIND11_MODULE(_rtp_llm_fp8_mqa, m) {
    m.def("init", &rtp_kernel::blackwell::init_mqa_runtime);
    m.def("set_num_sms", [](int num_sms) { deep_gemm::device_runtime->set_num_sms(num_sms); });
    m.def("get_num_sms", []() { return deep_gemm::device_runtime->get_num_sms(); });
    m.def("set_pdl", [](bool enable) { deep_gemm::device_runtime->set_pdl(enable); });
    m.def("fp8_mqa_logits",
          &deep_gemm::attention::fp8_mqa_logits,
          py::arg("q"),
          py::arg("kv"),
          py::arg("weights"),
          py::arg("cu_seq_len_k_start"),
          py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0);
    m.def("get_paged_mqa_logits_metadata",
          &deep_gemm::attention::get_paged_mqa_logits_metadata,
          py::arg("context_lens"),
          py::arg("block_kv"),
          py::arg("num_sms"),
          py::arg("indices") = std::nullopt);
    m.def("fp8_paged_mqa_logits",
          &deep_gemm::attention::fp8_paged_mqa_logits,
          py::arg("q"),
          py::arg("kv_cache"),
          py::arg("weights"),
          py::arg("context_lens"),
          py::arg("block_table"),
          py::arg("schedule_meta"),
          py::arg("max_context_len"),
          py::arg("clean_logits") = false,
          py::arg("indices")      = std::nullopt);
}
