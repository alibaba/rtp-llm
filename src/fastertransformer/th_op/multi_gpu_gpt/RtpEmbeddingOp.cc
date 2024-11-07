#include <optional>
#include <pybind11/pytypes.h>
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
RtpEmbeddingOp::RtpEmbeddingOp() {}

void RtpEmbeddingOp::init(py::object model, py::object mm_process_engine) {
    try {
        auto [gpt_init_params, gpt_weight] = rtp_llm::prepareEngineInitParams(model);
        rtp_llm::EngineInitParams params(gpt_init_params, std::move(*gpt_weight));
        py::object py_render = model.attr("custom_module").attr("renderer");
        py::object py_handler = model.attr("custom_module").attr("handler");

        if (gpt_init_params.tp_rank_ == 0) {
            // kmon metric init
            (void)rtp_llm::initKmonitorFactory();
            auto kmon_tags = rtp_llm::getHippoTags();
            params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
        }
        embedding_engine_.reset(new rtp_llm::EmbeddingEngine(params, py_handler));
        if (!mm_process_engine.is_none()) {
            mm_processor_.reset(new rtp_llm::MultimodalProcessor(mm_process_engine,
                params.gpt_init_parameter.mm_sep_tokens_,
                params.gpt_init_parameter.include_sep_tokens_,
                params.gpt_init_parameter.max_seq_len_));
        }
        startRpcServer(gpt_init_params, py_render, params.metrics_reporter);
        startHttpServer(embedding_engine_, mm_processor_, gpt_init_params, py_render);
    } catch (const std::exception& e) {
        FT_FAIL("init embedding engine failed, error msg: %s", e.what());
    }
}

void RtpEmbeddingOp::stop() {
    if (embedding_rpc_service_) {
        embedding_rpc_service_->stop();
    }
    if (http_server_) {
        http_server_->stop();
    }
    if (!is_server_shutdown_ && embedding_engine_) {
        (void)embedding_engine_->stop();
        is_server_shutdown_ = true;
    }
}

void RtpEmbeddingOp::startHttpServer(std::shared_ptr<rtp_llm::EmbeddingEngine>     embedding_engine,
                                     std::shared_ptr<rtp_llm::MultimodalProcessor> mm_processor,
                                     const ft::GptInitParameter&                   gpt_init_params,
                                     py::object                                    py_render) {
    http_server_.reset(new rtp_llm::HttpApiServer(embedding_engine, mm_processor, gpt_init_params, py_render));
    std::string http_server_address("tcp:0.0.0.0:" + std::to_string(gpt_init_params.http_port_));
    if (http_server_->start(http_server_address)) {
        FT_LOG_INFO("embedding HTTP Server listening on %s", http_server_address.c_str());
    } else {
        throw std::runtime_error("embedding HTTP Server start fail.");
    }
}

void RtpEmbeddingOp::startRpcServer(const ft::GptInitParameter& gpt_init_params,
                                    py::object py_render,
                                    kmonitor::MetricsReporterPtr reporter) {
    auto arpc_service = std::move(createEmbeddingArpcService(gpt_init_params, py_render, embedding_engine_, reporter));
    if (arpc_service) {
        FT_LOG_INFO("creating arpc service");
        embedding_rpc_service_.reset(new rtp_llm::ArpcServerWrapper(std::move(arpc_service),
                                                                    gpt_init_params.model_rpc_port_));
        embedding_rpc_service_->start();
    } else {
        FT_LOG_INFO("Embedding RPC not supported, skip");
    }
}

th::Tensor RtpEmbeddingOp::decode(th::Tensor token_ids,
                                  th::Tensor token_type_ids,
                                  th::Tensor input_lengths,
                                  int64_t    request_id,
                                  std::vector<rtp_llm::MultimodalInput> multimodal_inputs) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    std::optional<rtp_llm::MultimodalFeature> multimodal_features = std::nullopt;
    if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
        auto mm_res = mm_processor_->getMultimodallFeatures(ft::torchTensor2Buffer(token_ids), multimodal_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.status().ToString());
        }
        token_ids = ft::Buffer2torchTensor(mm_res.value().expanded_ids, true);
        multimodal_features.emplace(mm_res.value());
    }
    return embedding_engine_->decode(token_ids, token_type_ids, input_lengths, request_id, multimodal_features);
}

RtpEmbeddingOp::~RtpEmbeddingOp() {
    stop();
}

void registerRtpEmbeddingOp(const py::module& m) {
    rtp_llm::registerMultimodalInput(m);
    pybind11::class_<torch_ext::RtpEmbeddingOp>(m, "RtpEmbeddingOp")
        .def(pybind11::init<>())
        .def("init", &torch_ext::RtpEmbeddingOp::init)
        .def("stop", &torch_ext::RtpEmbeddingOp::stop)
        .def("decode", &torch_ext::RtpEmbeddingOp::decode, py::call_guard<py::gil_scoped_release>());
}

}  // namespace torch_ext
