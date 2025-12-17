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
    DeviceInitParams params;
    params.hw_kernel_config.enable_cuda_graph            = true;
    params.fifo_scheduler_config.max_context_batch_size  = max_context_batch_size;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = true;
    params.hw_kernel_config.prefill_capture_seq_lens     = {
        6,   10,  14,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75,  77,  80,  85,  90,  95,
        100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200,
        205, 210, 215, 220, 225, 230, 235, 240, 245, 248, 250, 252, 255, 256, 260, 265, 270, 275, 280, 285, 290,
        295, 300, 305, 310, 311, 315, 317, 320, 321, 325, 330, 335, 340, 345, 350, 355, 356, 360, 365, 370, 375,
        380, 385, 390, 395, 399, 400, 405, 410, 411, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470,
        475, 480, 485, 490, 495, 500, 512, 520, 540, 560, 576, 580, 600, 620, 629, 640, 660, 673, 680, 685, 697,
        700, 703, 720, 740, 760, 780, 793, 797, 800, 820, 837, 840, 844, 856, 860, 880, 889, 900, 920, 940, 960};
    // int  layer_num                              = 24;
    // int  block_num                              = 26037;
    c10::ScalarType    dtype             = torch::kBFloat16;
    int                num_tokens_per_bs = params.max_seq_len;  // prefill mode
    CudaGraphRunnerPtr cuda_graph_runner_ptr =
        new CudaGraphRunner(params, std::move(py_instance), dtype, num_tokens_per_bs, true);
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
