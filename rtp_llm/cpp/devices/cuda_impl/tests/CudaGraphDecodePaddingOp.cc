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
    DeviceInitParams params;
    params.hw_kernel_config.enable_cuda_graph            = true;
    params.concurrency_config.concurrency_limit          = 128;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = false;
    params.hidden_size                                   = 896;
    params.max_seq_len                                   = 64;
    params.tokens_per_block                              = 64;
    // int  layer_num                              = 24;
    // int  block_num                              = 26037;
    c10::ScalarType    dtype             = torch::kFloat16;
    int                num_tokens_per_bs = 1;  // decode mode
    CudaGraphRunnerPtr cuda_graph_runner_ptr =
        new CudaGraphRunner(params, std::move(py_instance), dtype, num_tokens_per_bs, false);
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
