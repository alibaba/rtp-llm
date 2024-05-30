#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "maga_transformer/cpp/embedding_engine/handlers/LinearSoftmaxHandler.h"

namespace torch_ext {


torch_ext::EmbeddingHandlerOp create_linear_softmax_handler(const ft::GptInitParameter& gpt_init_params) {
    auto handler_op = torch_ext::EmbeddingHandlerOp();
    std::unique_ptr<rtp_llm::HandlerBase> linear_softmax_handler = std::make_unique<rtp_llm::LinearSoftmaxHandler>(gpt_init_params);
    handler_op.setHandler(linear_softmax_handler);
    return std::move(handler_op);
}

void registerEmbeddingHandler(py::module& m) {
    pybind11::class_<torch_ext::EmbeddingHandlerOp>(m, "EmbeddingHandlerOp")
        .def(pybind11::init<>())  // quant_pre_scales
        .def("load_tensor", &torch_ext::EmbeddingHandlerOp::loadTensor)
        .def("forward", &torch_ext::EmbeddingHandlerOp::forward, py::call_guard<py::gil_scoped_release>());
    // register functions
    m.def("create_linear_softmax_handler", &create_linear_softmax_handler);
    
}

} // namespace torch_ext