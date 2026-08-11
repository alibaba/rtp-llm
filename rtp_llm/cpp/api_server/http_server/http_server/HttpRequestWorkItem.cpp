#include "http_server/HttpRequestWorkItem.h"

#include <utility>

#include "http_server/HttpResponseWriter.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpRequestWorkItem);

void HttpRequestWorkItem::process() {
    [[maybe_unused]] auto admissionToken = std::move(_admissionToken);
    if (!_func) {
        AUTIL_LOG(WARN, "process http request but route callback is null");
        return;
    }
    if (!_request) {
        AUTIL_LOG(WARN, "process http request but request is null, cannot call back");
        return;
    }
    auto writer = std::make_unique<HttpResponseWriter>(_conn);
    _func(std::move(writer), *_request);
}

}  // namespace http_server
