#pragma once

#include <string>
#include <exception>

#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"
#include "rtp_llm/cpp/api_server/ErrorResponse.h"
#include "rtp_llm/cpp/api_server/AccessLogWrapper.h"

namespace rtp_llm {

std::string getSource(const std::string& raw_request);
template<typename T>
void WriteExceptionResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer, const T& e);

class HttpApiServerException: public std::exception {
public:
    enum Type {
        CONCURRENCY_LIMIT_ERROR        = 409,
        CANCELLED_ERROR                = 499,
        ERROR_INPUT_FORMAT_ERROR       = 507,
        GPT_NOT_FOUND_ERROR            = 508,
        NO_PROMPT_ERROR                = 509,
        EMPTY_PROMPT_ERROR             = 510,
        LONG_PROMPT_ERROR              = 511,
        ERROR_STOP_LIST_FORMAT         = 512,
        UNKNOWN_ERROR                  = 514,
        UNSUPPORTED_OPERATION          = 515,
        ERROR_GENERATE_CONFIG_FORMAT   = 516,
        TOKENIZER_ERROR                = 517,
        MULTIMODAL_ERROR               = 518,
        UPDATE_ERROR                   = 601,
        MALLOC_ERROR                   = 602,
        GENERATE_TIMEOUT_ERROR         = 603,
        GET_HOST_ERROR                 = 604,
        GET_CONNECTION_ERROR           = 605,
        CONNECT_ERROR                  = 606,
        CONNECTION_RESET_BY_PEER_ERROR = 607,
        REMOTE_ALLOCATE_RESOURCE_ERROR = 608,
        REMOTE_LOAD_KV_CACHE_ERROR     = 609,
        REMOTE_GENERATE_ERROR          = 610,
    };
    HttpApiServerException(Type type, const std::string& message): type_(type), message_(message) {}

    virtual const char* what() const noexcept override {
        return message_.c_str();
    }
    Type getType() const {
        return type_;
    }
    std::string getMessage() const {
        return message_;
    }
    template<typename T>
    static void handleException(const T&                                                e,
                                int64_t                                                 request_id,
                                std::shared_ptr<ApiServerMetricReporter>                metric_reporter,
                                const http_server::HttpRequest&                         request,
                                const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
        const auto body = request.GetBody();
        if (metric_reporter) {
            std::string source     = getSource(body);
            int         error_code = -1;
            if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
                RTP_LLM_LOG_WARNING("dynamic_cast succ");
                error_code = he->getType();
            }
            metric_reporter->reportErrorQpsMetric(source, error_code);
        }
        RTP_LLM_LOG_WARNING("found exception: [%s]", e.what());
        AccessLogWrapper::logExceptionAccess(body, request_id, e.what());
        WriteExceptionResponse(writer, e);
    }
    template<typename T>
    static void handleException(const T&                                                e,
                                int64_t                                                 request_id,
                                kmonitor::MetricsReporterPtr                            metric_reporter,
                                const http_server::HttpRequest&                         request,
                                const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
        const auto body = request.GetBody();
        if (metric_reporter) {
            std::string source     = getSource(body);
            int         error_code = Type::UNKNOWN_ERROR;
            if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
                RTP_LLM_LOG_WARNING("dynamic_cast succ");
                error_code = he->getType();
            }
            std::map<std::string, std::string> tag_map;
            tag_map["source"]     = source;
            tag_map["error_code"] = std::to_string(error_code);
            auto tags             = kmonitor::MetricsTags(tag_map);
            metric_reporter->report(1, "py_rtp_framework_error_qps", kmonitor::MetricType::QPS, &tags, true);
        }
        RTP_LLM_LOG_WARNING("found exception: [%s]", e.what());
        AccessLogWrapper::logExceptionAccess(body, request_id, e.what());
        WriteExceptionResponse(writer, e);
    }

private:
    Type        type_;
    std::string message_;
};

template<typename T>
inline std::string formatException(const T& e) {
    std::string res;
    if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
        res = ErrorResponse::CreateErrorResponseJsonString(he->getType(), he->getMessage());
    } else {
        res = ErrorResponse::CreateErrorResponseJsonString(HttpApiServerException::UNKNOWN_ERROR,
                                                           std::string("http api server failed, exception occurred: ")
                                                               + e.what());
    }
    return res;
}

inline std::string getSource(const std::string& raw_request) {
    std::string source = "unknown";
    try {
        auto body    = autil::legacy::json::ParseJson(raw_request);
        auto bodyMap = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(body);
        auto it      = bodyMap.find("source");
        if (it == bodyMap.end()) {
            return source;
        }
        autil::legacy::FromJson(source, it->second);
        return source;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_DEBUG("getSource failed, error: %s", e.what());
    }
    return source;
}

template<typename T>
inline void WriteExceptionResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer, const T& e) {
    if (writer->isConnected() == false) {
        return;
    }
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    int status_code = 500;
    if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
        status_code = he->getType();
    }
    if (status_code >= 600) {
        status_code = 500;
    }
    writer->SetStatus(status_code, "Internal Server Error");

    writer->Write(formatException(e));
    return;
}

}  // namespace rtp_llm
