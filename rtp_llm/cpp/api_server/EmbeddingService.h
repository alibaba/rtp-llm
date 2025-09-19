#pragma once

#include "autil/AtomicCounter.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"
#include "rtp_llm/cpp/api_server/EmbeddingEndpoint.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"

namespace rtp_llm {

class EmbeddingService {
public:
    EmbeddingService(const std::shared_ptr<EmbeddingEndpoint>&     embedding_endpoint,
                     const std::shared_ptr<autil::AtomicCounter>&  request_counter,
                     const std::shared_ptr<ConcurrencyController>& controller,
                     const kmonitor::MetricsReporterPtr&           metrics_reporter);
    ~EmbeddingService() = default;

public:
    void embedding(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                   const http_server::HttpRequest&                         request,
                   std::optional<EmbeddingEndpoint::EmbeddingType>         type = std::nullopt);

private:
    std::string getSource(const std::string& raw_request);
    std::string getUsage(const std::string& raw_request);
    void        report(const double                 value,
                       const std::string&           name,
                       const kmonitor::MetricsTags& tags = kmonitor::MetricsTags(),
                       const kmonitor::MetricType   type = kmonitor::MetricType::QPS);

private:
    std::shared_ptr<EmbeddingEndpoint>     embedding_endpoint_;
    std::shared_ptr<autil::AtomicCounter>  request_counter_;
    std::shared_ptr<ConcurrencyController> controller_;
    kmonitor::MetricsReporterPtr           metrics_reporter_;
};

}  // namespace rtp_llm
