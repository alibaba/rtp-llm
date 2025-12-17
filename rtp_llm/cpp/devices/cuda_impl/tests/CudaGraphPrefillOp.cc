#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefillOp.h"
namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphPrefillOp::init(py::object       py_instance,
                              int64_t          max_context_batch_size,
                              int64_t          hidden_size,
                              int64_t          max_seq_len,
                              int64_t          tokens_per_block,
                              int64_t          max_prefill_cuda_graph_len,
                              std::vector<int> prefill_capture_seq_lens) {
    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance),
                                               max_context_batch_size,
                                               hidden_size,
                                               max_seq_len,
                                               tokens_per_block,
                                               prefill_capture_seq_lens);
    cuda_graph_runner_->setMaxPrefillCudaGraphLen(max_prefill_cuda_graph_len);
    cuda_graph_runner_->initCapture();
}

int CudaGraphPrefillOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
}

CudaGraphRunnerPtr CudaGraphPrefillOp::createCudaGraphRunner(py::object       py_instance,
                                                             int64_t          max_context_batch_size,
                                                             int64_t          hidden_size,
                                                             int64_t          max_seq_len,
                                                             int64_t          tokens_per_block,
                                                             std::vector<int> prefill_capture_seq_lens) {
    GraphParams graph_params;
    graph_params.enable_cuda_graph            = true;
    graph_params.enable_cuda_graph_debug_mode = true;
    graph_params.is_prefill_cuda_graph_mode   = true;
    graph_params.max_seq_len                  = max_seq_len;
    graph_params.tokens_per_block             = tokens_per_block;
    graph_params.kv_cache_block_offset        = 0;
    graph_params.max_context_batch_size       = max_context_batch_size;
    graph_params.prefill_capture_seq_lens     = prefill_capture_seq_lens;

    CudaGraphRunnerPtr cuda_graph_runner_ptr = CudaGraphRunner::create(graph_params, std::move(py_instance));
    return cuda_graph_runner_ptr;
}

PYBIND11_MODULE(libtest_cuda_graph_prefill_ops, m) {
    py::class_<cuda_graph::CudaGraphPrefillOp>(m, "CudaGraphPrefillOp")
        .def(py::init<>())
        .def("init",
             &CudaGraphPrefillOp::init,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("max_prefill_cuda_graph_len"),
             py::arg("prefill_capture_seq_lens"))
        .def("forward", &cuda_graph::CudaGraphPrefillOp::forward)
        .def("getCurrentRealGraphSize", &cuda_graph::CudaGraphPrefillOp::getCurrentRealGraphSize);
    // buildInputs is now implemented in Python (CudaGraphPrefill.py)
}

}  // namespace cuda_graph
