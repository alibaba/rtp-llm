#pragma once

#include <atomic>
#include <string>

#include "autil/AtomicCounter.h"

#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"

#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "maga_transformer/cpp/api_server/ConcurrencyControllerUtil.h"
#include "maga_transformer/cpp/api_server/TokenProcessor.h"
#include "maga_transformer/cpp/api_server/EmbeddingEndpoint.h"
#include "maga_transformer/cpp/api_server/InferenceService.h"

namespace rtp_llm {

class HealthService;
class WorkerStatusService;
class ModelStatusService;
class SysCmdService;
class TokenizerService;

class HttpApiServer {
public:
    // normal engine
    HttpApiServer(std::shared_ptr<EngineBase> engine,
                  std::shared_ptr<MultimodalProcessor> mm_processor,
                  std::string                 address,
                  const ft::GptInitParameter& params,
                  py::object                  token_processor):
        engine_(engine),
        mm_processor_(mm_processor),
        addr_(address),
        params_(params),
        token_processor_(new TokenProcessor(token_processor)) {

        active_request_count_.reset(new autil::AtomicCounter());
        request_counter_.reset(new autil::AtomicCounter());
        init_controller(params);
    }

    // embedding engine
    HttpApiServer(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                  std::shared_ptr<MultimodalProcessor> mm_processor,
                  const ft::GptInitParameter&          params,
                  py::object                           py_render):
        params_(params), embedding_endpoint_(EmbeddingEndpoint(embedding_engine, mm_processor, py_render)) {

        active_request_count_.reset(new autil::AtomicCounter());
        request_counter_.reset(new autil::AtomicCounter());
        init_controller(params);
    }

    ~HttpApiServer() = default;

public:
    bool        start();
    bool        start(const std::string& address);
    void        stop();
    bool        isStoped() const;
    std::string getListenAddr() const {
        return addr_;
    }

private:
    void init_controller(const ft::GptInitParameter& params);

private:
    bool registerServices();
    bool registerHealthService();
    bool registerWorkerStatusService();
    bool registerModelStatusService();
    bool registerSysCmdService();
    bool registerTokenizerService();
    bool registerInferenceService();

private:
    std::atomic_bool                      is_stopped_{true};
    std::shared_ptr<autil::AtomicCounter> active_request_count_;
    std::shared_ptr<autil::AtomicCounter> request_counter_;

    std::shared_ptr<EngineBase>            engine_;
    std::shared_ptr<MultimodalProcessor>   mm_processor_;
    std::string                            addr_;
    ft::GptInitParameter                   params_;
    std::shared_ptr<ConcurrencyController> controller_;
    std::shared_ptr<TokenProcessor>              token_processor_;

    std::optional<EmbeddingEndpoint> embedding_endpoint_;

    std::unique_ptr<http_server::HttpServer> http_server_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;

    std::shared_ptr<HealthService>           health_service_;
    std::shared_ptr<WorkerStatusService>     worker_status_service_;
    std::shared_ptr<ModelStatusService>      model_status_service_;
    std::shared_ptr<SysCmdService>           sys_cmd_service_;
    std::shared_ptr<TokenizerService>        tokenizer_service_;
    std::shared_ptr<InferenceService>        inference_service_;
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
