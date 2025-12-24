#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefillOp.h"

namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphPrefillOp::init(py::object       py_instance,
                              int64_t          max_context_batch_size,
                              int64_t          max_seq_len,
                              int64_t          tokens_per_block,
                              int64_t          max_prefill_cuda_graph_len,
                              std::vector<int> prefill_capture_seq_lens) {
    CudaGraphRunnerConfig config;
    config.is_prefill_mode          = true;
    config.enable_debug_mode        = true;
    config.max_seq_len              = max_seq_len;
    config.tokens_per_block         = tokens_per_block;
    config.max_context_batch_size   = max_context_batch_size;
    config.prefill_capture_seq_lens = prefill_capture_seq_lens;
    config.data_type                = torch::kBFloat16;

    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance), config);
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
