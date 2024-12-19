#include "maga_transformer/cpp/api_server/Exception.h"

#include "autil/legacy/jsonizable.h"

#include "maga_transformer/cpp/utils/Logger.h"

#include "maga_transformer/cpp/api_server/ErrorResponse.h"
#include "maga_transformer/cpp/api_server/AccessLogWrapper.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

inline std::string formatException(const std::exception& e) {
    std::string res;
    if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
        res = ErrorResponse::CreateErrorResponseJsonString(he->getType(), he->getMessage());
    } else {
        res = ErrorResponse::CreateErrorResponseJsonString(
                HttpApiServerException::UNKNOWN_ERROR,
                std::string("inference failed, exception occurred: ") + e.what());
    }
    return res;
}

inline std::string getSource(const std::string& raw_request) {
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
        FT_LOG_DEBUG("getSource failed, error: %s", e.what());
    }
    return source;
}

inline void WriteExceptionResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                   const std::exception& e) {
    if (writer->isConnected() == false) {
        return;
    }
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    int status_code = 500;
    if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
        status_code = he->getType();
    }
    writer->SetStatus(status_code, "Internal Server Error");

    writer->Write(formatException(e));
}

void HttpApiServerException::handleException(const std::exception& e,
                                             int64_t request_id,
                                             std::shared_ptr<ApiServerMetricReporter> metric_reporter,
                                             const http_server::HttpRequest& request,
                                             const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
    const auto body = request.GetBody();
    if (metric_reporter) {
        std::string source = getSource(body);
        int error_code = -1;
        if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
            FT_LOG_WARNING("dynamic_cast succ");
            error_code = he->getType();
        }
        metric_reporter->reportErrorQpsMetric(source, error_code);
    }
    FT_LOG_WARNING("found exception: [%s]", e.what());
    AccessLogWrapper::logExceptionAccess(body, request_id, e.what());
    WriteExceptionResponse(writer, e);
}

}  // namespace rtp_llm
