#include "rtp_llm/cpp/api_server/EmbeddingService.h"

#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/AccessLogWrapper.h"
#include "rtp_llm/cpp/api_server/InferenceDataType.h"

namespace rtp_llm {

EmbeddingService::EmbeddingService(const std::shared_ptr<EmbeddingEndpoint>&     embedding_endpoint,
                                   const std::shared_ptr<autil::AtomicCounter>&  request_counter,
                                   const std::shared_ptr<ConcurrencyController>& controller,
                                   const kmonitor::MetricsReporterPtr&           metrics_reporter):
    embedding_endpoint_(embedding_endpoint),
    request_counter_(request_counter),
    controller_(controller),
    metrics_reporter_(metrics_reporter) {}

std::string EmbeddingService::getSource(const std::string& raw_request) {
    std::string source = "unknown";
    try {
        auto body    = ParseJson(raw_request);
        auto bodyMap = AnyCast<JsonMap>(body);
        auto it      = bodyMap.find("source");
        if (it == bodyMap.end()) {
            return source;
        }
        FromJson(source, it->second);
        return source;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_DEBUG("embedding getSource failed, error: %s", e.what());
    }
    return source;
}

std::string EmbeddingService::getUsage(const std::string& response) {
    std::string usage = "{}";
    try {
        auto body    = ParseJson(response);
        auto bodyMap = AnyCast<JsonMap>(body);
        auto it      = bodyMap.find("usage");
        if (it == bodyMap.end()) {
            return usage;
        }
        usage = ToJsonString(it->second, /*isCompact=*/true);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_DEBUG("embedding getUsage failed, error: %s", e.what());
    }
    return usage;
}

void EmbeddingService::report(const double                 value,
                              const std::string&           name,
                              const kmonitor::MetricsTags& tags,
                              const kmonitor::MetricType   type) {
    kmonitor::MetricsTags metric_tags = kmonitor::MetricsTags(tags);
    if (metrics_reporter_) {
        metrics_reporter_->report(value, name, type, &metric_tags, true);
    }
}

void EmbeddingService::embedding(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const http_server::HttpRequest&                         request,
                                 std::optional<EmbeddingEndpoint::EmbeddingType>         type) {

    const auto body = request.GetBody();

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    if (!ParallelInfo::globalParallelInfo().isMaster()) {
        RTP_LLM_LOG_WARNING("worker can't process embedding request.");
        writer->SetStatus(403, "Forbidden");
        writer->Write(R"({"detail": "worker can't process embedding request."})");
        auto tags = kmonitor::MetricsTags("error_code", std::to_string(HttpApiServerException::UNSUPPORTED_OPERATION));
        report(1, "py_rtp_framework_error_qps", tags);
        // metric_reporter_->reportErrorQpsMetric(getSource(body), HttpApiServerException::UNSUPPORTED_OPERATION);
        return;
    }

    if (!embedding_endpoint_) {
        RTP_LLM_LOG_WARNING("non-embedding model can't handle embedding request!");
        writer->SetStatus(501, "Not Implemented");
        writer->Write(R"({"detail": "non-embedding model can't handle embedding request!"})");
        auto tags = kmonitor::MetricsTags("error_code", std::to_string(HttpApiServerException::UNKNOWN_ERROR));
        report(1, "py_rtp_framework_error_qps", tags);
        // metric_reporter_->reportErrorQpsMetric(getSource(body), HttpApiServerException::UNKNOWN_ERROR);
        return;
    }
    if (!controller_ || !request_counter_) {
        RTP_LLM_LOG_WARNING("embedding model: controller or request_counter null!");
        writer->SetStatus(500, "Internal Server Error");
        writer->Write(R"({"detail": "embedding model: controller or request_counter null!"})");
        auto tags = kmonitor::MetricsTags("error_code", std::to_string(HttpApiServerException::UNKNOWN_ERROR));
        report(1, "py_rtp_framework_error_qps", tags);
        // metric_reporter_->reportErrorQpsMetric(getSource(body), HttpApiServerException::UNKNOWN_ERROR);
        return;
    }

    ConcurrencyControllerGuard controller_guard(controller_);

    EmbeddingRequest req;
    int64_t          request_id = request_counter_->incAndReturn();
    try {
        FromJsonString(req, body);
    } catch (autil::legacy::ExceptionBase& e) {
        RTP_LLM_LOG_WARNING("embedding request parse failed.");
        AccessLogWrapper::logExceptionAccess(body, request_id, e.what());
        writer->SetStatus(400, "Bad Request");
        writer->Write(R"({"detail": "embedding request parse failed."})");
        auto tags =
            kmonitor::MetricsTags("error_code", std::to_string(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR));
        report(1, "py_rtp_framework_error_qps", tags);
        // metric_reporter_->reportErrorQpsMetric(req.source, HttpApiServerException::ERROR_INPUT_FORMAT_ERROR);
        return;
    }

    try {
        // metric_reporter_->reportQpsMetric(req.source);
        report(1, "py_rtp_framework_qps");

        auto [response, logable_response] =
            embedding_endpoint_->handle(body, type, metrics_reporter_, request.getRecvTime());
        writer->AddHeader("USAGE", getUsage(response));
        writer->Write(response);

        report((autil::TimeUtility::currentTimeInMicroSeconds() - request.getRecvTime()) / 1000.0,
               "py_rtp_framework_rt",
               kmonitor::MetricsTags(),
               kmonitor::MetricType::GAUGE);
        report(1, "py_rtp_success_qps_metric");
        // metric_reporter_->reportSuccessQpsMetric(req.source);
        // metric_reporter_->reportResponseLatencyMs(autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms);

        AccessLogWrapper::logSuccessAccess(
            body, request_id, request.getRecvTime(), logable_response, req.private_request);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("embedding endpoint handle request failed, found exception: %s", e.what());
        HttpApiServerException::handleException(e, request_id, metrics_reporter_, request, writer);
    }
}

}  // namespace rtp_llm
