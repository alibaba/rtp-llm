#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "maga_transformer/cpp/embedding_engine/handlers/LinearSoftmaxHandler.h"

namespace torch_ext {
}

// static auto rtpEmbeddinHandlergOpTHS =
//     torch::jit::class_<torch_ext::EmbeddingHandlerOp>("FasterTransformer", "EmbeddingHandlerOp")
//         .def(torch::jit::init<>());  // quant_pre_scales

// torch::intrusive_ptr<torch_ext::EmbeddingHandlerOp> create_linear_softmax_handler(const c10::intrusive_ptr<ft::GptInitParameter> gpt_init_params) {
//     auto handler_op = torch::make_intrusive<torch_ext::EmbeddingHandlerOp>();
//     std::unique_ptr<rtp_llm::HandlerBase> linear_softmax_handler = std::make_unique<rtp_llm::LinearSoftmaxHandler>(*gpt_init_params);
//     handler_op->setHandler(linear_softmax_handler);    
//     return handler_op;
// }

// static auto create_linear_softmax_handler_ths =
//     torch::RegisterOperators("fastertransformer::create_linear_softmax_handler", &create_linear_softmax_handler);