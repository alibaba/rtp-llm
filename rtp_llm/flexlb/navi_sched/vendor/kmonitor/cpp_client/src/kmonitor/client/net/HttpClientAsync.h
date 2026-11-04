#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tools/HttpClient.h"
#include "autil/Log.h"
#include "autil/ThreadPool.h"
#include "autil/WorkItem.h"
#include "kmonitor/client/common/Common.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

constexpr int HTTP_CLIENT_DEFAULT_THREAD_NUM = 4;
constexpr int HTTP_CLIENT_DEFAULT_QUEUE_SIZE = 32;
constexpr int HTTP_CLIENT_DEFAULT_TIMEOUT_MS = 3000;

class RequestWorkItem : public autil::WorkItem {
public:
    RequestWorkItem(std::function<void()> &&func) : func_(std::forward<std::function<void()>>(func)) {}
    virtual ~RequestWorkItem() override = default;
    void process() override {
        if (func_) {
            func_();
        }
    }

private:
    std::function<void()> func_;
};

class HttpClientAsync {
public:
    HttpClientAsync(int threadNum = HTTP_CLIENT_DEFAULT_THREAD_NUM,
                    int queueSize = HTTP_CLIENT_DEFAULT_QUEUE_SIZE,
                    int timeoutMs = HTTP_CLIENT_DEFAULT_TIMEOUT_MS);
    ~HttpClientAsync();
    void asyncPost(const std::string &addr, const std::string &&path, std::string &&data);

private:
    autil::ThreadPool threadPool_;
    anet::tools::HttpClient client_;
    AUTIL_LOG_DECLARE();
};

END_KMONITOR_NAMESPACE(kmonitor);
