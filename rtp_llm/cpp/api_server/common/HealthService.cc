#include "rtp_llm/cpp/api_server/common/HealthService.h"

#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HealthService::HealthService(ReadinessStateProvider readiness_state_provider):
    readiness_state_provider_(std::move(readiness_state_provider)) {}

void HealthService::healthCheck(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    if (is_stopped_.load()) {
        RTP_LLM_LOG_WARNING("called health route, but server has been shutdown");
        writer->SetStatus(503, "Service Unavailable");
        writer->Write(R"({"detail":"this server has been shutdown"})");
        return;
    }
    if (readiness_state_provider_) {
        const auto state = readiness_state_provider_();
        if (state.empty()) {
            writer->Write(R"("ok")");
            return;
        }
        RTP_LLM_LOG_WARNING("called health route, but engine is not ready: state=%s", state.c_str());
        writer->SetStatus(503, "Service Unavailable");
        writer->Write(R"({"detail":"engine is not ready","state":")" + state + R"("})");
        return;
    }
    writer->Write(R"("ok")");
}

void HealthService::healthCheck2(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const http_server::HttpRequest&                         request) {

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    if (is_stopped_.load()) {
        RTP_LLM_LOG_WARNING("called root route, but server has been shutdown");
        writer->SetStatus(503, "Service Unavailable");
        writer->Write(R"({"detail":"this server has been shutdown"})");
        return;
    }
    writer->Write(R"({"status":"home"})");
}

void HealthService::stop() {
    is_stopped_.store(true);
}

bool registerHealthServiceStatic(http_server::HttpServer& http_server, std::shared_ptr<HealthService> health_service) {
    auto raw_resp_callback = [health_service](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                              const http_server::HttpRequest&                  request) -> void {
        health_service->healthCheck(writer, request);
    };

    auto json_resp_callback = [health_service](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                               const http_server::HttpRequest&                  request) -> void {
        health_service->healthCheck2(writer, request);
    };

    return http_server.RegisterRoute("GET", "/health", raw_resp_callback)
           && http_server.RegisterRoute("POST", "/health", raw_resp_callback)
           && http_server.RegisterRoute("GET", "/GraphService/cm2_status", raw_resp_callback)
           && http_server.RegisterRoute("POST", "/GraphService/cm2_status", raw_resp_callback)
           && http_server.RegisterRoute("GET", "/SearchService/cm2_status", raw_resp_callback)
           && http_server.RegisterRoute("POST", "/SearchService/cm2_status", raw_resp_callback)
           && http_server.RegisterRoute("GET", "/status", raw_resp_callback)
           && http_server.RegisterRoute("POST", "/status", raw_resp_callback)
           && http_server.RegisterRoute("POST", "/health_check", raw_resp_callback)
           && http_server.RegisterRoute("GET", "/", json_resp_callback);
}

}  // namespace rtp_llm
