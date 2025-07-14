#include "rtp_llm/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "rtp_llm/cpp/embedding_engine/handlers/LinearSoftmaxHandler.h"

namespace torch_ext {

torch_ext::EmbeddingHandlerOp create_linear_softmax_handler(const rtp_llm::GptInitParameter& gpt_init_params) {
    auto                                  handler_op = torch_ext::EmbeddingHandlerOp();
    std::unique_ptr<rtp_llm::HandlerBase> linear_softmax_handler =
        std::make_unique<rtp_llm::LinearSoftmaxHandler>(gpt_init_params);
    handler_op.setHandler(linear_softmax_handler);
    return handler_op;
}

void registerEmbeddingHandler(py::module& m) {
    pybind11::class_<torch_ext::EmbeddingHandlerOp>(m, "EmbeddingHandlerOp")
        .def(pybind11::init<>())  // quant_pre_scales
        .def("load_tensor", &torch_ext::EmbeddingHandlerOp::loadTensor, py::arg("weights"))
        .def("forward",
             &torch_ext::EmbeddingHandlerOp::forward,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("hidden_states"),
             py::arg("input_lengths"));
    // register functions
    m.def("create_linear_softmax_handler", &create_linear_softmax_handler, py::arg("gpt_init_params"));
}

}  // namespace torch_ext
