#include "kmonitor/client/net/HttpClientAsync.h"

#include <utility>

BEGIN_KMONITOR_NAMESPACE(kmonitor);
AUTIL_LOG_SETUP(kmonitor, HttpClientAsync);

HttpClientAsync::HttpClientAsync(int threadNum, int queueSize, int timeoutMs) : threadPool_(threadNum, queueSize) {
    client_.init(timeoutMs);
    threadPool_.start("prometheus_sink");
}

HttpClientAsync::~HttpClientAsync() { threadPool_.stop(); }

void HttpClientAsync::asyncPost(const std::string &addr, const std::string &&path, std::string &&data) {
    auto req = [&, path = std::move(path), data = std::move(data)]() {
        std::string result;
        if (!client_.post(addr, path, data, result)) {
            AUTIL_LOG(ERROR, "post addr:%s path:%s  failed", addr.c_str(), path.c_str());
        }
    };
    RequestWorkItem *workItem = new RequestWorkItem(std::move(req));
    if (threadPool_.pushWorkItem(workItem) != autil::ThreadPool::ERROR_NONE) {
        AUTIL_LOG(ERROR, "push work item failed");
        delete workItem;
    }
}

END_KMONITOR_NAMESPACE(kmonitor);
