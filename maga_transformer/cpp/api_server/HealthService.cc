#include "maga_transformer/cpp/api_server/HealthService.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {

void HealthService::healthCheck(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    if (is_stopped_.load()) {
        FT_LOG_WARNING("called health route, but server has been shutdown");
        writer->SetStatus(503, "Service Unavailable");
        writer->Write(R"({"detail":"this server has been shutdown"})");
        return;
    }
    writer->Write(R"("ok")");
}

void HealthService::healthCheck2(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const http_server::HttpRequest&                         request) {

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    if (is_stopped_.load()) {
        FT_LOG_WARNING("called root route, but server has been shutdown");
        writer->SetStatus(503, "Service Unavailable");
        writer->Write(R"({"detail":"this server has been shutdown"})");
        return;
    }
    writer->Write(R"({"status":"home"})");
}

void HealthService::stop() {
    is_stopped_.store(true);
}

}  // namespace rtp_llm
