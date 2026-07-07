#include "http_server/HttpRequestWorkItem.h"

#include "autil/TimeUtility.h"
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
    auto    writer = std::make_unique<HttpResponseWriter>(_conn);
    int64_t start  = autil::TimeUtility::monotonicTimeUs();
    _func(std::move(writer), *_request);
    int64_t cost = autil::TimeUtility::monotonicTimeUs() - start;
    if (cost >= 2000000) {
        AUTIL_LOG(WARN,
                  "[HttpRequestThreadPool] slow request: %s %s, cost_us: %ld",
                  _request->GetMethod().c_str(),
                  _request->GetEndpoint().c_str(),
                  cost);
    }
}

}  // namespace http_server