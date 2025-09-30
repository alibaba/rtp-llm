#include <optional>
#include <pybind11/pytypes.h>

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"

#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"

using namespace std;

namespace th = torch;

namespace rtp_llm {

std::tuple<GptInitParameter, std::unique_ptr<Weights>> prepareEngineInitParams(py::object model, bool sp_model);

RtpEmbeddingOp::RtpEmbeddingOp() {}

void RtpEmbeddingOp::init(py::object model, py::object mm_process_engine) {
    try {
        auto [gpt_init_params, gpt_weight] = prepareEngineInitParams(model, false);
        auto                      py_model = model.attr("py_model");
        EngineInitParams params(0, gpt_init_params, std::move(*gpt_weight), py_model);
        py::object                custom_module = model.attr("custom_module");
        py::object                py_render     = model.attr("custom_module").attr("renderer");
        py::object                py_tokenizer  = model.attr("tokenizer");
        py::object                py_handler    = model.attr("custom_module").attr("handler");

        if (gpt_init_params.tp_rank_ == 0) {
            // kmon metric init
            (void)initKmonitorFactory();
            auto kmon_tags = kmonitor::MetricsTags();
            kmon_tags.AddTag("dp_rank", std::to_string(gpt_init_params.dp_rank_));
            params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
        }
        embedding_engine_.reset(new EmbeddingEngine(params, py_handler));
        if (!mm_process_engine.is_none()) {
            mm_processor_.reset(new LocalMultimodalProcessor(mm_process_engine, params.gpt_init_parameter));
        }
        startRpcServer(gpt_init_params, py_render, py_tokenizer, params.metrics_reporter, mm_processor_);
        startHttpServer(embedding_engine_, mm_processor_, params, custom_module);
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("init embedding engine failed, error msg: %s", e.what());
    }
}

void RtpEmbeddingOp::stop() {
    if (embedding_rpc_service_) {
        embedding_rpc_service_->stop();
        embedding_rpc_service_.reset();
    }
    if (http_server_) {
        http_server_->stop();
        http_server_.reset();
    }
    if (!is_server_shutdown_ && embedding_engine_) {
        (void)embedding_engine_->stop();
        embedding_engine_.reset();
        is_server_shutdown_ = true;
    }
    stopKmonitorFactory();
}

void RtpEmbeddingOp::startHttpServer(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                                     std::shared_ptr<MultimodalProcessor> mm_processor,
                                     const EngineInitParams&              params,
                                     py::object                                    custom_module) {
    http_server_.reset(new HttpApiServer(embedding_engine, mm_processor, params, custom_module));
    std::string http_server_address("tcp:0.0.0.0:" + std::to_string(params.gpt_init_parameter.http_port_));
    if (http_server_->start(http_server_address)) {
        RTP_LLM_LOG_INFO("embedding HTTP Server listening on %s", http_server_address.c_str());
    } else {
        throw std::runtime_error("embedding HTTP Server start fail.");
    }
}

void RtpEmbeddingOp::startRpcServer(const GptInitParameter&              gpt_init_params,
                                    py::object                                    py_render,
                                    py::object                                    py_tokenizer,
                                    kmonitor::MetricsReporterPtr                  reporter,
                                    std::shared_ptr<MultimodalProcessor> mm_processor) {
    auto arpc_service = std::move(createEmbeddingArpcService(
        gpt_init_params, py_render, py_tokenizer, mm_processor, embedding_engine_, reporter));
    if (arpc_service) {
        RTP_LLM_LOG_INFO("creating arpc service");
        embedding_rpc_service_.reset(new ArpcServerWrapper(std::move(arpc_service),
                                                                    gpt_init_params.arpc_config.threadNum,
                                                                    gpt_init_params.arpc_config.queueNum,
                                                                    gpt_init_params.arpc_config.ioThreadNum,
                                                                    gpt_init_params.model_rpc_port_));
        embedding_rpc_service_->start();
    } else {
        RTP_LLM_LOG_INFO("Embedding RPC not supported, skip");
    }
}

py::object RtpEmbeddingOp::decode(th::Tensor                            token_ids,
                                  th::Tensor                            token_type_ids,
                                  th::Tensor                            input_lengths,
                                  int64_t                               request_id,
                                  std::vector<MultimodalInput> multimodal_inputs) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    std::optional<MultimodalFeature> multimodal_features = std::nullopt;
    auto                                      embedding_input     = std::make_shared<EmbeddingInput>(
        token_ids, token_type_ids, input_lengths, request_id, multimodal_features);
    if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(embedding_input, multimodal_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.ToString());
        }
    }
    auto                   embedding_output = embedding_engine_->decode(embedding_input);
    py::gil_scoped_acquire acquire;
    if (embedding_output->output.isTensor) {
        RTP_LLM_CHECK_WITH_INFO(embedding_output->output.t.has_value(), "embedding output has null tensor value");
        return convertTensorToObject(embedding_output->output.t.value());
    } else {
        RTP_LLM_CHECK_WITH_INFO(embedding_output->output.map.has_value(), "embedding output has null map value");
        return convertTensorMapVectorToObject(embedding_output->output.map.value());
    }
}

RtpEmbeddingOp::~RtpEmbeddingOp() {
    stop();
}

void registerRtpEmbeddingOp(const py::module& m) {
    pybind11::class_<RtpEmbeddingOp>(m, "RtpEmbeddingOp")
        .def(pybind11::init<>())
        .def("init", &RtpEmbeddingOp::init, py::arg("model"), py::arg("mm_process_engine"))
        .def("stop", &RtpEmbeddingOp::stop)
        .def("decode",
             &RtpEmbeddingOp::decode,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("token_ids"),
             py::arg("token_type_ids"),
             py::arg("input_lengths"),
             py::arg("request_id"),
             py::arg("multimodal_inputs"));
}

}  // namespace rtp_llm
