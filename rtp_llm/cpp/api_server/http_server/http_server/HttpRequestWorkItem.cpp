#include "http_server/HttpRequestWorkItem.h"

#include "http_server/HttpResponseWriter.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpRequestWorkItem);

void HttpRequestWorkItem::process() {
    if (!_func) {
        AUTIL_LOG(WARN, "process http request but route callback is null");
        return;
    }
    if (!_request) {
        AUTIL_LOG(WARN, "process http request but request is null, cannot call back");
        return;
    }
    if (!_writer)
        _writer = std::make_unique<HttpResponseWriter>(_conn);
    _func(std::move(_writer), *_request);
}

void HttpRequestWorkItem::reject(int status, const std::string& message) {
    if (!_writer)
        _writer = std::make_unique<HttpResponseWriter>(_conn);
    _writer->SetWriteType(HttpResponseWriter::WriteType::Normal);
    _writer->SetStatus(status, status == 503 ? "Service Unavailable" : "Internal Server Error");
    _writer->Write(message);
}

}  // namespace http_server