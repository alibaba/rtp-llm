#pragma once

#include <atomic>
#include <functional>
#include <utility>
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpServer.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

class HealthService {
public:
    explicit HealthService(std::function<bool()> check_health = {}): check_health_(std::move(check_health)) {}
    ~HealthService() = default;

public:
    void healthCheck(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                     const http_server::HttpRequest&                         request);
    void healthCheck2(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request);

    void stop();

private:
    const std::function<bool()> check_health_;
    std::atomic_bool is_stopped_{false};
};

bool registerHealthServiceStatic(http_server::HttpServer& http_server, std::shared_ptr<HealthService> health_service);

}  // namespace rtp_llm