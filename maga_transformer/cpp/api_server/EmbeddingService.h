#pragma once

#include "autil/AtomicCounter.h"

#include "maga_transformer/cpp/http_server/http_server/HttpResponseWriter.h"
#include "maga_transformer/cpp/http_server/http_server/HttpRequest.h"

#include "maga_transformer/cpp/api_server/ApiServerMetrics.h"
#include "maga_transformer/cpp/api_server/EmbeddingEndpoint.h"
#include "maga_transformer/cpp/api_server/ConcurrencyControllerUtil.h"

namespace rtp_llm {

class EmbeddingService {
public:
    EmbeddingService(const std::shared_ptr<EmbeddingEndpoint>&       embedding_endpoint,
                     const std::shared_ptr<autil::AtomicCounter>&    request_counter,
                     const std::shared_ptr<ConcurrencyController>&   controller,
                     const std::shared_ptr<ApiServerMetricReporter>& metric_reporter);
    ~EmbeddingService() = default;
public:
    void embedding(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                   const http_server::HttpRequest&                         request,
                   std::optional<EmbeddingEndpoint::EmbeddingType>         type = std::nullopt);
private:
    std::string getSource(const std::string& raw_request);
    std::string getUsage(const std::string& raw_request);
private:
    std::shared_ptr<EmbeddingEndpoint>       embedding_endpoint_;
    std::shared_ptr<autil::AtomicCounter>    request_counter_;
    std::shared_ptr<ConcurrencyController>   controller_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;
};

}  // namespace rtp_llm
