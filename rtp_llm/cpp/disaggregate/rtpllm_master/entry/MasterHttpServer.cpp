#include "rtp_llm/cpp/disaggregate/rtpllm_master/entry/MasterHttpServer.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/entry/Response.h"
#include "rtp_llm/cpp/api_server/common/HealthService.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace rtp_llm_master {

bool MasterHttpServer::start() {
    if (!tokenize_service_ || !load_balancer_) {
        RTP_LLM_LOG_ERROR("MasterHttpServer start failed, tokenize service or load balancer is null.");
        return false;
    }
    http_server_.reset(new http_server::HttpServer(/*transport=*/nullptr, 64, 64));
    if (!registerServices()) {
        RTP_LLM_LOG_ERROR("MasterHttpServer start failed, register services failed");
        return false;
    }

    if (!http_server_->Start(address_)) {
        RTP_LLM_LOG_ERROR("MasterHttpServer start failed, start http server failed, address is %s.", address_.c_str());
        return false;
    }

    is_stopped_.store(false);
    RTP_LLM_LOG_INFO("MasterHttpServer start success, listen address is %s.", address_.c_str());
    return true;
}

void MasterHttpServer::stop() {
    RTP_LLM_LOG_WARNING("master http api server stopped");
    is_stopped_.store(true);

    if (health_service_) {
        health_service_->stop();
    }

    if (http_server_) {
        http_server_->Stop();
    }
}


void MasterHttpServer::handleError(const http_server::HttpRequest&                  request,
                                   std::unique_ptr<http_server::HttpResponseWriter> writer,
                                   const std::string&                               error_msg) {
    writer->SetStatus(500, "Internal Server Error");
    writer->Write(MasterErrorResponse::CreateJsonString(500, error_msg));
}

void MasterHttpServer::handleRequest(const http_server::HttpRequest&                  request,
                                     std::unique_ptr<http_server::HttpResponseWriter> writer) {
    auto start_time = autil::TimeUtility::currentTimeInMilliSeconds();
    MasterInfo info;
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    const auto body = request.GetBody();
    try {
        auto res = tokenize_service_->encodeRequest(body, biz_name_);
        auto tokenize_end_time = autil::TimeUtility::currentTimeInMilliSeconds();
        info.tokenize_cost_time_ms = tokenize_end_time - start_time;
        if (!res.ok()) {
            handleError(request, std::move(writer), std::string(res.status().message()));
            return;
        }
        // info.request_id = generator_.getRandomString();
        info.request_id = request_id_counter_.fetch_add(1);
        info.input_length = res.value()->input_length;
        info.prefix_length = res.value()->prefix_length;
        res.value()->task_id = std::to_string(info.request_id);
        auto ret = load_balancer_->chooseHostWithTask(biz_name_, *res.value());
        if (!ret.ok()) {
            handleError(request, std::move(writer), std::string(ret.status().message()));
            return;
        }        
        writer->SetStatus(200, "OK");
        info.expect_execute_time_ms = ret.value().expect_execute_time_ms;
        info.expect_wait_time_ms = ret.value().expect_wait_time_ms;
        info.estimate_cost_time_ms = autil::TimeUtility::currentTimeInMilliSeconds() - tokenize_end_time;
        info.machine_info = ret.value().machine_info;
        writer->Write(MasterSuccessResponse::CreateJsonString(ret.value().host->ip, ret.value().host->http_port, info));
    } catch (std::exception& e) {
        std::string error_msg = autil::StringUtil::formatString("MasterHttpServer handle request failed with exception: %s", e.what());
        handleError(request, std::move(writer), error_msg);
    }
}

bool MasterHttpServer::registerServices() {
    if (!http_server_) {
        RTP_LLM_LOG_ERROR("HttpApiServer register services failed, http server is null.");
        return false;
    }
    return registerHealthService() && registerHandleService();
}

bool MasterHttpServer::registerHealthService() {
    health_service_.reset(new HealthService());
    return registerHealthServiceStatic(*http_server_, health_service_);
}

bool MasterHttpServer::registerHandleService() {
    auto callback = [this](std::unique_ptr<http_server::HttpResponseWriter> writer, const http_server::HttpRequest& request) {
        this->handleRequest(request, std::move(writer));
    };

    return http_server_->RegisterRoute("POST", "/", callback);
}

bool MasterHttpServer::isStoped() const {
    return is_stopped_.load();
}

MasterHttpServer::~MasterHttpServer() {
    if (isStoped() == false) {
        stop();
    }
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm