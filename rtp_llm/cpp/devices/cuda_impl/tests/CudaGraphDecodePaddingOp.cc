#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphDecodePaddingOp.h"
#include "c10/core/ScalarType.h"
namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphDecodePaddingOp::init(py::object       py_instance,
                                    int64_t          hidden_size,
                                    int64_t          max_seq_len,
                                    int64_t          tokens_per_block,
                                    int64_t          kv_block_offset,
                                    std::vector<int> decode_capture_batch_sizes) {
    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance),
                                               hidden_size,
                                               max_seq_len,
                                               tokens_per_block,
                                               kv_block_offset,
                                               decode_capture_batch_sizes);
    cuda_graph_runner_->initCapture();
}

int CudaGraphDecodePaddingOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
}

CudaGraphRunnerPtr CudaGraphDecodePaddingOp::createCudaGraphRunner(py::object       py_instance,
                                                                   int64_t          hidden_size,
                                                                   int64_t          max_seq_len,
                                                                   int64_t          tokens_per_block,
                                                                   int64_t          kv_block_offset,
                                                                   std::vector<int> decode_capture_batch_sizes) {

    GraphParams graph_params;
    graph_params.enable_cuda_graph            = true;
    graph_params.enable_cuda_graph_debug_mode = false;
    graph_params.is_prefill_cuda_graph_mode   = false;
    graph_params.max_seq_len                  = max_seq_len;
    graph_params.tokens_per_block             = tokens_per_block;
    graph_params.kv_cache_block_offset        = kv_block_offset;
    graph_params.concurrency_limit            = 128;
    graph_params.decode_capture_batch_sizes   = decode_capture_batch_sizes;

    CudaGraphRunnerPtr cuda_graph_runner_ptr = CudaGraphRunner::create(graph_params, std::move(py_instance));
    return cuda_graph_runner_ptr;
}

PYBIND11_MODULE(libtest_cuda_graph_decode_ops, m) {
    py::class_<cuda_graph::CudaGraphDecodePaddingOp>(m, "CudaGraphDecodePaddingOp")
        .def(py::init<>())
        .def("init",
             &CudaGraphDecodePaddingOp::init,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kv_block_offset"),
             py::arg("decode_capture_batch_sizes"))
        .def("forward", &cuda_graph::CudaGraphDecodePaddingOp::forward)
        .def("getCurrentRealGraphSize", &cuda_graph::CudaGraphDecodePaddingOp::getCurrentRealGraphSize);
}

}  // namespace cuda_graph
