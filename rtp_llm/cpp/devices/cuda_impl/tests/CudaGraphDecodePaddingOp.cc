#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphDecodePaddingOp.h"

namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphDecodePaddingOp::init(py::object       py_instance,
                                    int64_t          hidden_size,
                                    int64_t          max_seq_len,
                                    int64_t          tokens_per_block,
                                    std::vector<int> decode_capture_batch_sizes) {
    GraphParams params;
    params.enable_cuda_graph            = true;
    params.enable_cuda_graph_debug_mode = false;
    params.is_prefill_cuda_graph_mode   = false;
    params.max_seq_len                  = max_seq_len;
    params.tokens_per_block             = tokens_per_block;
    params.num_tokens_per_bs            = 1;  // Decode mode
    params.hidden_size                  = hidden_size;
    params.model_data_type              = c10::ScalarType::Half;
    params.concurrency_limit            = 128;
    params.decode_capture_batch_sizes   = decode_capture_batch_sizes;

    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance), params);
    cuda_graph_runner_->initCapture();
}

int CudaGraphDecodePaddingOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
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
             py::arg("decode_capture_batch_sizes"))
        .def("forward", &CudaGraphDecodePaddingOp::forward)
        .def("getCurrentRealGraphSize", &cuda_graph::CudaGraphDecodePaddingOp::getCurrentRealGraphSize);
}

}  // namespace cuda_graph
