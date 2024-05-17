#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"

#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingQueryConverter.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

RtpEmbeddingOp::RtpEmbeddingOp(const c10::intrusive_ptr<ft::GptInitParameter> gpt_init_params, const c10::intrusive_ptr<EmbeddingHandlerOp> handler_op): handler_(handler_op->getHandler()) {
}

std::vector<std::string> RtpEmbeddingOp::handlerTensorInfo() {
    return handler_.tensorInfo();
}

void RtpEmbeddingOp::init(const c10::intrusive_ptr<ft::GptInitParameter>&               gpt_init_params,
                    const std::vector<std::unordered_map<std::string, th::Tensor>>& layer_weights,
                    const c10::Dict<std::string, th::Tensor>&                       weights) {
    rtp_llm::MagaInitParams params;
    params.gpt_init_parameter = gpt_init_params;
    ft::DeviceFactory::initDevices(ft::DeviceFactory::getDefaultGlobalDeviceParams());
    for (auto& it : weights) {
        global_weights_.emplace(it.key(), ft::torchTensor2Buffer(it.value()));
    }
    for (auto& weights : layer_weights) {
        std::unordered_map<std::string, ft::ConstBufferPtr> __weights;
        for (auto& it : weights) {
            __weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        layer_weights_.emplace_back(std::move(__weights));
    }
    THROW_IF_STATUS_ERROR(handler_.loadTensor(global_weights_));
    embedding_engine_.reset(new rtp_llm::EmbeddingEngine(params, layer_weights_, global_weights_, handler_));
}

void RtpEmbeddingOp::stop() {
    if (!is_server_shutdown_) {
        (void)embedding_engine_->stop();
        is_server_shutdown_ = true;
    }
}

th::Tensor RtpEmbeddingOp::decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    auto embedding_stream = rtp_llm::EmbeddingQueryConverter::convertEmbeddingInputs(token_ids, token_type_ids, input_lengths, request_id);
    THROW_IF_STATUS_ERROR(embedding_engine_->enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return rtp_llm::EmbeddingQueryConverter::convertEmbeddingOutputs(embedding_stream);
}

RtpEmbeddingOp::~RtpEmbeddingOp() {
    stop();
}

}  // namespace torch_ext


static auto rtpEmbeddingOpTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::RtpEmbeddingOp>("FasterTransformerRtpEmbeddingOp")
#else
    torch::jit::class_<torch_ext::RtpEmbeddingOp>("FasterTransformer", "RtpEmbeddingOp")
#endif
        .def(torch::jit::init<c10::intrusive_ptr<ft::GptInitParameter>, c10::intrusive_ptr<torch_ext::EmbeddingHandlerOp>>())  // quant_pre_scales
        .def("init", &torch_ext::RtpEmbeddingOp::init)
        .def("stop", &torch_ext::RtpEmbeddingOp::stop)
        .def("decode", &torch_ext::RtpEmbeddingOp::decode)
        .def("handler_tensor_info", &torch_ext::RtpEmbeddingOp::handlerTensorInfo);

