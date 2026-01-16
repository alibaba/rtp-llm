#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/api_server/common/HealthService.h"
#include "rtp_llm/cpp/api_server/WorkerStatusService.h"
#include "rtp_llm/cpp/api_server/ModelStatusService.h"
#include "rtp_llm/cpp/api_server/SysCmdService.h"
#include "rtp_llm/cpp/api_server/TokenizerService.h"
#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void HttpApiServer::init_controller(const ConcurrencyConfig& concurrency_config,
                                    const ParallelismConfig& parallelism_config) {
    bool block = concurrency_config.concurrency_with_block;
    RTP_LLM_LOG_INFO("Get concurrency_with_block: %d from ConcurrencyConfig.",
                     concurrency_config.concurrency_with_block);
    if (parallelism_config.tp_rank == 0) {
        int limit = concurrency_config.concurrency_limit;
        RTP_LLM_LOG_INFO("CONCURRENCY_LIMIT to %d", limit);
        controller_ = std::make_shared<ConcurrencyController>(limit, block);
    } else /* if (parallelism_config.tp_size != 1) */ {
        RTP_LLM_LOG_INFO("use gang cluster and is worker, set CONCURRENCY_LIMIT to 99");
        controller_ = std::make_shared<ConcurrencyController>(99, block);
    }
}

bool HttpApiServer::start(const std::string& address) {
    // TODO: queueSize may interleave with controller :(
    http_server_.reset(new http_server::HttpServer(/*transport=*/nullptr,
                                                   /*threadNum=*/controller_->get_available_concurrency(),
                                                   /*queueSize=*/controller_->get_available_concurrency() * 5));
    metric_reporter_.reset(new ApiServerMetricReporter());
    if (!metric_reporter_->init()) {
        RTP_LLM_LOG_WARNING("HttpApiServer start init metric reporter failed.");
        return false;
    }

    if (!registerServices()) {
        RTP_LLM_LOG_ERROR("HttpApiServer start failed, register services failed, address is %s.", address.c_str());
        return false;
    }

    if (!http_server_->Start(address)) {
        RTP_LLM_LOG_ERROR("HttpApiServer start failed, start http server failed, address is %s.", address.c_str());
        return false;
    }

    is_stopped_.store(false);
    RTP_LLM_LOG_INFO("HttpApiServer start success, listen address is %s.", address.c_str());
    return true;
}

bool HttpApiServer::start() {
    return start(addr_);
}

bool HttpApiServer::start(py::object model_weights_loader,
                          py::object lora_infos,
                          py::object world_info,
                          py::object tokenizer,
                          py::object render) {
    if (lora_infos.is_none() == false) {
        lora_infos_ = lora_infos.cast<std::map<std::string, std::string>>();
    }
    tokenizer_.reset(new Tokenizer(tokenizer));
    if (render.is_none() == false) {
        render_.reset(new ChatRender(render));
    }
    return start(addr_);
}

bool HttpApiServer::registerServices() {
    // add uri:
    // GET: / /health /GraphService/cm2_status /SearchService/cm2_status
    // POST: /health /GraphService/cm2_status /SearchService/cm2_status /health_check
    if (!registerHealthService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register health service failed.");
        return false;
    }

    // add uri:
    // GET: /worker_status
    if (!registerWorkerStatusService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register worker status service failed.");
        return false;
    }

    // GET: /v1/models
    if (!registerModelStatusService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register model status service failed.");
        return false;
    }

    // POST: /set_log_level
    if (!registerSysCmdService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register sys cmd service failed.");
        return false;
    }

    // add uri:
    // POST: /tokenizer/encode
    if (!registerTokenizerService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register tokenizer service failed.");
        return false;
    }

    // add uri:
    // POST: /chat/completions /v1/chat/completions /chat/render /v1/chat/render
    if (!registerChatService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register chat service failed.");
        return false;
    }

    // add uri:
    // POST: /
    if (!is_embedding_ && !registerInferenceService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register inference service failed.");
        return false;
    }

    // add uri
    // POST / /v1/embeddings /v1/embeddings/similarity /v1/classifier /v1/rerank
    if (is_embedding_ && !registerEmbedingService()) {
        RTP_LLM_LOG_WARNING("HttpApiServer register embeding service failed.");
        return false;
    }

    RTP_LLM_LOG_INFO("HttpApiServer register services success.");
    return true;
}

bool HttpApiServer::registerHealthService() {
    if (!http_server_) {
        RTP_LLM_LOG_WARNING("register health service failed, http server is null");
        return false;
    }

    health_service_.reset(new HealthService());
    return registerHealthServiceStatic(*http_server_, health_service_);
}

bool HttpApiServer::registerWorkerStatusService() {
    if (!http_server_) {
        RTP_LLM_LOG_WARNING("register worker status service failed, http server is null");
        return false;
    }

    worker_status_service_.reset(new WorkerStatusService(engine_, controller_));
    auto callback = [worker_status_service =
                         worker_status_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                 const http_server::HttpRequest&                  request) -> void {
        worker_status_service->workerStatus(writer, request);
    };
    return http_server_->RegisterRoute("GET", "/worker_status", callback);
}

bool HttpApiServer::registerModelStatusService() {
    if (!http_server_) {
        RTP_LLM_LOG_WARNING("register model status service failed, http server is null");
        return false;
    }

    model_status_service_.reset(new ModelStatusService());
    auto callback = [model_status_service =
                         model_status_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                const http_server::HttpRequest&                  request) -> void {
        model_status_service->modelStatus(writer, request);
    };
    return http_server_->RegisterRoute("GET", "/v1/models", callback);
}

bool HttpApiServer::registerSysCmdService() {
    if (!http_server_) {
        RTP_LLM_LOG_WARNING("register sys cmd service failed, http server is null");
        return false;
    }

    sys_cmd_service_.reset(new SysCmdService());
    auto set_log_level_callback = [sys_cmd_service =
                                       sys_cmd_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                         const http_server::HttpRequest& request) -> void {
        sys_cmd_service->setLogLevel(writer, request);
    };
    return http_server_->RegisterRoute("POST", "/set_log_level", set_log_level_callback);
}

bool HttpApiServer::registerTokenizerService() {
    if (!http_server_) {
        RTP_LLM_LOG_WARNING("register tokenizer service failed, http server is null");
        return false;
    }

    tokenizer_service_.reset(new TokenizerService(token_processor_));
    auto tokenizer_encode_callback = [tokenizer_service =
                                          tokenizer_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                              const http_server::HttpRequest& request) -> void {
        tokenizer_service->tokenizerEncode(writer, request);
    };
    return http_server_->RegisterRoute("POST", "/tokenizer/encode", tokenizer_encode_callback);
}

bool HttpApiServer::registerChatService() {
    chat_service_.reset(new ChatService(
        engine_, mm_processor_, request_counter_, tokenizer_, render_, params_.model_config_, metric_reporter_));
    auto chat_completions_callback = [active_request_count = active_request_count_,
                                      chat_service         = chat_service_,
                                      controller           = controller_,
                                      request_counter      = request_counter_,
                                      metric_reporter =
                                          metric_reporter_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                            const http_server::HttpRequest& request) -> void {
        auto request_id = request_counter->incAndReturn();
        try {
            CounterGuard               counter_guard(active_request_count);
            ConcurrencyControllerGuard controller_guard(controller);
            if (controller_guard.isPassed() == false) {
                if (metric_reporter) {
                    metric_reporter->reportConflictQpsMetric();
                }
                throw HttpApiServerException(HttpApiServerException::CONCURRENCY_LIMIT_ERROR, "Too Many Requests");
            }
            chat_service->chatCompletions(writer, request, request_id);
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("chat completion failed, found python exception: [%s]", e.what());
            HttpApiServerException::handleException(e, request_id, metric_reporter, request, writer);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("chat completion failed, found cpp exception: [%s]", e.what());
            HttpApiServerException::handleException(e, request_id, metric_reporter, request, writer);
        }
    };

    auto chat_render_callback = [active_request_count = active_request_count_,
                                 request_counter      = request_counter_,
                                 metric_reporter      = metric_reporter_,
                                 chat_service = chat_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                               const http_server::HttpRequest& request) -> void {
        auto request_id = request_counter->incAndReturn();
        try {
            CounterGuard counter_guard(active_request_count);
            chat_service->chatRender(writer, request);
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("chat render failed, found python exception: [%s]", e.what());
            HttpApiServerException::handleException(e, request_id, metric_reporter, request, writer);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("called chat render route but found exception: [%s]", e.what());
            HttpApiServerException::handleException(e, request_id, metric_reporter, request, writer);
        }
    };
    return http_server_->RegisterRoute("POST", "/chat/completions", chat_completions_callback)
           && http_server_->RegisterRoute("POST", "/v1/chat/completions", chat_completions_callback)
           && http_server_->RegisterRoute("POST", "/chat/render", chat_render_callback)
           && http_server_->RegisterRoute("POST", "/v1/chat/render", chat_render_callback);
}

bool HttpApiServer::registerInferenceService() {
    if (!http_server_) {
        RTP_LLM_LOG_WARNING("register inference service failed, http server is null");
        return false;
    }
    inference_service_.reset(new InferenceService(engine_,
                                                  mm_processor_,
                                                  request_counter_,
                                                  token_processor_,
                                                  controller_,
                                                  params_.model_config_,
                                                  metric_reporter_));
    auto inference_internal_callback =
        [active_request_count = active_request_count_, inference_service = inference_service_](
            std::unique_ptr<http_server::HttpResponseWriter> writer, const http_server::HttpRequest& request) -> void {
        CounterGuard counter_guard(active_request_count);
        inference_service->inference(writer, request, /*isInternal=*/true);
    };
    auto inference_callback = [active_request_count = active_request_count_, inference_service = inference_service_](
                                  std::unique_ptr<http_server::HttpResponseWriter> writer,
                                  const http_server::HttpRequest&                  request) -> void {
        CounterGuard counter_guard(active_request_count);
        inference_service->inference(writer, request, /*isInternal=*/false);
    };

    return http_server_->RegisterRoute("POST", "/", inference_callback)
           && http_server_->RegisterRoute("POST", "/inference_internal", inference_internal_callback);
}

bool HttpApiServer::registerEmbedingService() {
    embedding_service_.reset(
        new EmbeddingService(embedding_endpoint_, request_counter_, controller_, metrics_reporter_));
    auto callback = [active_request_count = active_request_count_,
                     embedding_service    = embedding_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                             const http_server::HttpRequest& request) -> void {
        CounterGuard counter_guard(active_request_count);
        embedding_service->embedding(writer, request);
    };
    auto callback_dense = [active_request_count = active_request_count_, embedding_service = embedding_service_](
                              std::unique_ptr<http_server::HttpResponseWriter> writer,
                              const http_server::HttpRequest&                  request) -> void {
        CounterGuard counter_guard(active_request_count);
        embedding_service->embedding(writer, request, EmbeddingEndpoint::DENSE);
    };
    auto callback_sparse = [active_request_count = active_request_count_, embedding_service = embedding_service_](
                               std::unique_ptr<http_server::HttpResponseWriter> writer,
                               const http_server::HttpRequest&                  request) -> void {
        CounterGuard counter_guard(active_request_count);
        embedding_service->embedding(writer, request, EmbeddingEndpoint::SPARSE);
    };
    auto callback_colbert = [active_request_count = active_request_count_, embedding_service = embedding_service_](
                                std::unique_ptr<http_server::HttpResponseWriter> writer,
                                const http_server::HttpRequest&                  request) -> void {
        CounterGuard counter_guard(active_request_count);
        embedding_service->embedding(writer, request, EmbeddingEndpoint::COLBERT);
    };

    return http_server_->RegisterRoute("POST", "/v1/embeddings", callback)
           && http_server_->RegisterRoute("POST", "/v1/embeddings/similarity", callback)
           && http_server_->RegisterRoute("POST", "/v1/embeddings/dense", callback_dense)
           && http_server_->RegisterRoute("POST", "/v1/embeddings/sparse", callback_sparse)
           && http_server_->RegisterRoute("POST", "/v1/embeddings/colbert", callback_colbert)
           && http_server_->RegisterRoute("POST", "/v1/classifier", callback)
           && http_server_->RegisterRoute("POST", "/v1/reranker", callback)
           && http_server_->RegisterRoute("POST", "/", callback);
}

void HttpApiServer::stop() {
    RTP_LLM_LOG_WARNING("http api server stopped");
    is_stopped_.store(true);

    if (health_service_) {
        health_service_->stop();
    }

    if (worker_status_service_) {
        worker_status_service_->stop();
    }

    // wait all active request finished
    if (active_request_count_) {
        while (active_request_count_->getValue() > 0) {
            RTP_LLM_LOG_DEBUG("wait active request processed. active request count: %d",
                              active_request_count_->getValue());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    if (http_server_) {
        http_server_->Stop();
    }
}

bool HttpApiServer::isStoped() const {
    // TODO:
    return is_stopped_.load();
}

}  // namespace rtp_llm
