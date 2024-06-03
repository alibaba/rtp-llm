#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingQueryConverter.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/utils/pybind_utils.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
RtpEmbeddingOp::RtpEmbeddingOp() {}

void RtpEmbeddingOp::init(const ft::GptInitParameter& gpt_init_params, py::object handler_impl, py::object py_layers_weights, py::object py_global_weights) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    auto global_weights = rtp_llm::WeightsConverter::convertPyWeightsMap(py_global_weights);
    auto layers_weights = rtp_llm::WeightsConverter::convertPyWeightsMapVec(py_layers_weights);
    rtp_llm::EngineInitParams params(gpt_init_params, layers_weights, global_weights);
    // kmon metric init
    (void)rtp_llm::initKmonitorFactory();
    auto kmon_tags = rtp_llm::getHippoTags();
    params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
    embedding_engine_.reset(new rtp_llm::EmbeddingEngine(params, handler_impl));
}

void RtpEmbeddingOp::stop() {
    if (!is_server_shutdown_) {
        (void)embedding_engine_->stop();
    }
}

th::Tensor RtpEmbeddingOp::decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    auto embedding_stream = rtp_llm::EmbeddingQueryConverter::convertEmbeddingInputs(token_ids, token_type_ids, input_lengths, request_id);
    embedding_stream->setMetricReporter(metrics_reporter_);
    THROW_IF_STATUS_ERROR(embedding_engine_->enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return rtp_llm::EmbeddingQueryConverter::convertEmbeddingOutputs(embedding_stream);
}

RtpEmbeddingOp::~RtpEmbeddingOp() {
    stop();
}

void registerRtpEmbeddingOp(const py::module& m) {
    pybind11::class_<torch_ext::RtpEmbeddingOp>(m, "RtpEmbeddingOp")
        .def(pybind11::init<>())
        .def("init", &torch_ext::RtpEmbeddingOp::init)
        .def("stop", &torch_ext::RtpEmbeddingOp::stop)
        .def("decode", &torch_ext::RtpEmbeddingOp::decode, py::call_guard<py::gil_scoped_release>());
}

}  // namespace torch_ext
