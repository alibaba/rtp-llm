#pragma once

#include <atomic>
#include "maga_transformer/cpp/http_server/http_server/HttpResponseWriter.h"
#include "maga_transformer/cpp/http_server/http_server/HttpRequest.h"

namespace rtp_llm {

class HealthService {
public:
    HealthService()  = default;
    ~HealthService() = default;

public:
    void healthCheck(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                     const http_server::HttpRequest&                         request);
    void healthCheck2(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request);

    void stop();

private:
    std::atomic_bool is_stopped_{false};
};

}  // namespace rtp_llm