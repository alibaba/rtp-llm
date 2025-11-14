#pragma once

#include <atomic>
#include <string>

#include "autil/AtomicCounter.h"

#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpServer.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"
#include "rtp_llm/cpp/api_server/TokenProcessor.h"
#include "rtp_llm/cpp/api_server/EmbeddingEndpoint.h"
#include "rtp_llm/cpp/api_server/ChatService.h"
#include "rtp_llm/cpp/api_server/InferenceService.h"
#include "rtp_llm/cpp/api_server/EmbeddingService.h"
#include "rtp_llm/cpp/api_server/LoraService.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class HealthService;
class WorkerStatusService;
class ModelStatusService;
class SysCmdService;
class TokenizerService;

class HttpApiServer {
public:
    // normal engine
    HttpApiServer(std::shared_ptr<EngineBase>          engine,
                  std::shared_ptr<MultimodalProcessor> mm_processor,
                  std::string                          address,
                  const EngineInitParams&              params,
                  py::object                           token_processor):
        engine_(engine),
        mm_processor_(mm_processor),
        addr_(address),
        params_(params),
        token_processor_(new TokenProcessor(token_processor)),
        metrics_reporter_(params.metrics_reporter) {
        is_embedding_ = false;
        active_request_count_.reset(new autil::AtomicCounter());
        request_counter_.reset(new autil::AtomicCounter());
        init_controller(params_.concurrency_config, params_.parallelism_config);
    }

    // embedding engine
    HttpApiServer(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                  std::shared_ptr<MultimodalProcessor> mm_processor,
                  const EngineInitParams&              params,
                  py::object                           custom_module):
        params_(params), metrics_reporter_(params.metrics_reporter) {
        is_embedding_       = true;
        embedding_endpoint_ = std::make_shared<EmbeddingEndpoint>(embedding_engine, mm_processor, custom_module);
        active_request_count_.reset(new autil::AtomicCounter());
        request_counter_.reset(new autil::AtomicCounter());
        init_controller(params_.concurrency_config, params_.parallelism_config);
    }

    ~HttpApiServer() = default;

public:
    bool        start();
    bool        start(const std::string& address);
    bool        start(py::object model_weights_loader,
                      py::object lora_infos,
                      py::object gang_info,
                      py::object tokenizer,
                      py::object render);
    void        stop();
    bool        isStoped() const;
    std::string getListenAddr() const {
        return addr_;
    }

private:
    void init_controller(const ConcurrencyConfig& concurrency_config, const ParallelismConfig& parallelism_config);

private:
    bool registerServices();
    bool registerHealthService();
    bool registerWorkerStatusService();
    bool registerModelStatusService();
    bool registerSysCmdService();
    bool registerTokenizerService();
    bool registerChatService();
    bool registerInferenceService();
    bool registerEmbedingService();
    bool registerLoraService();

private:
    bool                                  is_embedding_;
    std::atomic_bool                      is_stopped_{true};
    std::shared_ptr<autil::AtomicCounter> active_request_count_;
    std::shared_ptr<autil::AtomicCounter> request_counter_;

    std::shared_ptr<EngineBase>          engine_;
    std::shared_ptr<MultimodalProcessor> mm_processor_;
    std::string                          addr_;

    const EngineInitParams&               params_;
    std::shared_ptr<ConcurrencyController> controller_;
    std::shared_ptr<TokenProcessor>        token_processor_;

    std::shared_ptr<EmbeddingEndpoint> embedding_endpoint_;
    std::shared_ptr<Tokenizer>         tokenizer_;
    std::shared_ptr<ChatRender>        render_;

    std::unique_ptr<http_server::HttpServer> http_server_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;
    kmonitor::MetricsReporterPtr             metrics_reporter_;
    std::shared_ptr<GangServer>              gang_server_;
    std::shared_ptr<WeightsLoader>           weights_loader_;
    std::map<std::string, std::string>       lora_infos_;

    std::shared_ptr<HealthService>       health_service_;
    std::shared_ptr<WorkerStatusService> worker_status_service_;
    std::shared_ptr<ModelStatusService>  model_status_service_;
    std::shared_ptr<SysCmdService>       sys_cmd_service_;
    std::shared_ptr<TokenizerService>    tokenizer_service_;
    std::shared_ptr<ChatService>         chat_service_;
    std::shared_ptr<InferenceService>    inference_service_;
    std::shared_ptr<EmbeddingService>    embedding_service_;
    std::shared_ptr<LoraService>         lora_service_;
};

class CounterGuard {
public:
    CounterGuard(std::shared_ptr<autil::AtomicCounter> counter): counter_(counter) {
        if (counter_) {
            counter_->inc();
        }
    }
    ~CounterGuard() {
        if (counter_) {
            counter_->dec();
        }
    }

private:
    std::shared_ptr<autil::AtomicCounter> counter_;
};

}  // namespace rtp_llm
