#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
RtpEmbeddingOp::RtpEmbeddingOp() {}

void RtpEmbeddingOp::init(const ft::GptInitParameter& gpt_init_params, py::object py_render, py::object py_handler,
                          py::object py_layers_weights, py::object py_global_weights) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
    auto convert = rtp_llm::WeightsConverter(false, gpt_init_params.quant_algo_);
    rtp_llm::EngineInitParams params(gpt_init_params,
                                     std::move(*convert.createGptWeights(py_layers_weights, py_global_weights)));
    if (gpt_init_params.tp_rank_ == 0) {
        // kmon metric init
        (void)rtp_llm::initKmonitorFactory();
        auto kmon_tags = rtp_llm::getHippoTags();
        params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
    }
    embedding_engine_.reset(new rtp_llm::EmbeddingEngine(params, py_handler));        
    startRpcServer(gpt_init_params, py_render, params.metrics_reporter);
}

void RtpEmbeddingOp::stop() {
    if (embedding_rpc_service_) {
        embedding_rpc_service_->stop();
    }
    if (!is_server_shutdown_ && embedding_engine_) {
        (void)embedding_engine_->stop();
        is_server_shutdown_ = true;
    }
}

void RtpEmbeddingOp::startRpcServer(const ft::GptInitParameter& gpt_init_params, py::object py_render, kmonitor::MetricsReporterPtr reporter) {
    auto arpc_service = std::move(createEmbeddingArpcService(gpt_init_params, py_render, embedding_engine_, reporter));
    if (arpc_service) {        
        FT_LOG_INFO("creating arpc service");
        embedding_rpc_service_.reset(new rtp_llm::ArpcServerWrapper(std::move(arpc_service), gpt_init_params.model_rpc_port_));
        embedding_rpc_service_->start();
    } else {
        FT_LOG_INFO("Embedding RPC not supported, skip");
    }
}

th::Tensor RtpEmbeddingOp::decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    return embedding_engine_->decode(token_ids, token_type_ids, input_lengths, request_id);
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
