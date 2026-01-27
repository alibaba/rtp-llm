#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefillOp.h"

namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphPrefillOp::init(py::object       py_instance,
                              int64_t          max_context_batch_size,
                              int64_t          max_seq_len,
                              int64_t          tokens_per_block,
                              int64_t          max_prefill_cuda_graph_len,
                              std::vector<int> prefill_capture_seq_lens) {
    GraphParams params;
    params.enable_cuda_graph            = true;
    params.enable_cuda_graph_debug_mode = true;
    params.is_prefill_cuda_graph_mode   = true;
    params.max_seq_len                  = max_seq_len;
    params.tokens_per_block             = tokens_per_block;
    params.num_tokens_per_bs            = max_seq_len;  // Prefill mode
    params.max_context_batch_size       = max_context_batch_size;
    params.model_data_type              = c10::ScalarType::BFloat16;
    params.prefill_capture_seq_lens     = prefill_capture_seq_lens;

    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance), params);
    cuda_graph_runner_->setMaxPrefillCudaGraphLen(max_prefill_cuda_graph_len);
    cuda_graph_runner_->initCapture();
}

int CudaGraphPrefillOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
}

PYBIND11_MODULE(libtest_cuda_graph_prefill_ops, m) {
    py::class_<cuda_graph::CudaGraphPrefillOp>(m, "CudaGraphPrefillOp")
        .def(py::init<>())
        .def("init",
             &CudaGraphPrefillOp::init,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("max_prefill_cuda_graph_len"),
             py::arg("prefill_capture_seq_lens"))
        .def("forward", &CudaGraphPrefillOp::forward)
        .def("getCurrentRealGraphSize", &cuda_graph::CudaGraphPrefillOp::getCurrentRealGraphSize);
}

}  // namespace cuda_graph
