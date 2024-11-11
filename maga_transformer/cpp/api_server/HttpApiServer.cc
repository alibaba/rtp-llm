#include "maga_transformer/cpp/api_server/HttpApiServer.h"
#include "maga_transformer/cpp/api_server/HealthService.h"
#include "maga_transformer/cpp/api_server/WorkerStatusService.h"
#include "maga_transformer/cpp/api_server/ModelStatusService.h"
#include "maga_transformer/cpp/api_server/SysCmdService.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {

void HttpApiServer::init_controller(const ft::GptInitParameter& params) {
    bool block = autil::EnvUtil::getEnv("CONCURRENCY_WITH_BLOCK", false);
    if (params.tp_rank_ == 0) {
        int limit = autil::EnvUtil::getEnv("CONCURRENCY_LIMIT", 32);
        FT_LOG_INFO("CONCURRENCY_LIMIT to %d", limit);
        controller_ = std::make_shared<ConcurrencyController>(limit, block);
    } else /* if (params.tp_size_ != 1) */ {
        FT_LOG_INFO("use gang cluster and is worker, set CONCURRENCY_LIMIT to 99");
        controller_ = std::make_shared<ConcurrencyController>(99, block);
    }
}

bool HttpApiServer::start(const std::string& address) {
    http_server_.reset(new http_server::HttpServer());

    if (!registerServices()) {
        FT_LOG_ERROR("HttpApiServer start failed, register services failed, address is %s.", address.c_str());
        return false;
    }

    if (!http_server_->Start(address)) {
        FT_LOG_ERROR("HttpApiServer start failed, start http server failed, address is %s.", address.c_str());
        return false;
    }

    is_stopped_.store(false);
    FT_LOG_INFO("HttpApiServer start success, listen address is %s.", address.c_str());
    return true;
}

bool HttpApiServer::start() {
    return start(addr_);
}

bool HttpApiServer::registerServices() {
    // add uri:
    // GET: / /health /GraphService/cm2_status /SearchService/cm2_status
    // POST: /health /GraphService/cm2_status /SearchService/cm2_status /health_check
    if (!registerHealthService()) {
        FT_LOG_WARNING("HttpApiServer register health service failed.");
        return false;
    }

    // add uri:
    // GET: /worker_status
    if (!registerWorkerStatusService()) {
        FT_LOG_WARNING("HttpApiServer register worker status service failed.");
        return false;
    }

    // GET: /v1/models
    if (!registerModelStatusService()) {
        FT_LOG_WARNING("HttpApiServer register model status service failed.");
        return false;
    }

    // POST: /set_log_level
    if (!registerSysCmdService()) {
        FT_LOG_WARNING("HttpApiServer register sys cmd service failed.");
        return false;
    }

    FT_LOG_INFO("HttpApiServer register services success.");
    return true;
}

bool HttpApiServer::registerHealthService() {
    if (!http_server_) {
        FT_LOG_WARNING("register health service failed, http server is null");
        return false;
    }

    health_service_.reset(new HealthService());
    auto raw_resp_callback = [health_service = health_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                                const http_server::HttpRequest& request) -> void {
        health_service->healthCheck(writer, request);
    };

    auto json_resp_callback = [health_service =
                                   health_service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                    const http_server::HttpRequest&                  request) -> void {
        health_service->healthCheck2(writer, request);
    };

    return http_server_->RegisterRoute("GET", "/health",                    raw_resp_callback) &&
           http_server_->RegisterRoute("POST", "/health",                   raw_resp_callback) &&
           http_server_->RegisterRoute("GET", "/GraphService/cm2_status",   raw_resp_callback) &&
           http_server_->RegisterRoute("POST", "/GraphService/cm2_status",  raw_resp_callback) &&
           http_server_->RegisterRoute("GET", "/SearchService/cm2_status",  raw_resp_callback) &&
           http_server_->RegisterRoute("POST", "/SearchService/cm2_status", raw_resp_callback) &&
           http_server_->RegisterRoute("GET", "/status",                    raw_resp_callback) &&
           http_server_->RegisterRoute("POST", "/status",                   raw_resp_callback) &&
           http_server_->RegisterRoute("POST", "/health_check",             raw_resp_callback) &&
           http_server_->RegisterRoute("GET", "/",                         json_resp_callback);
}

bool HttpApiServer::registerWorkerStatusService() {
    if (!http_server_) {
        FT_LOG_WARNING("register worker status service failed, http server is null");
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
        FT_LOG_WARNING("register model status service failed, http server is null");
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
        FT_LOG_WARNING("register sys cmd service failed, http server is null");
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

void HttpApiServer::stop() {
    is_stopped_.store(true);

    if (health_service_) {
        health_service_->stop();
    }

    // stop http server
    // TODO: maybe wait all request finished

    if (http_server_) {
        http_server_->Stop();
    }
}

bool HttpApiServer::isStoped() const {
    // TODO:
    return is_stopped_.load();
}

}  // namespace rtp_llm
