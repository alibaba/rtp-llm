#pragma once

#include "autil/Log.h"
#include "autil/TimeUtility.h"

namespace http_arpc {

class KMonitorHTTPRPCServerMetricReporter;

// 保存一次 HTTP RPC 调用过程中的统计信息，用于记录服务端处理HTTP请求的各个阶段
class HTTPRPCStats {
public:
    HTTPRPCStats(std::shared_ptr<KMonitorHTTPRPCServerMetricReporter> metricReporter)
        : _metricReporter(metricReporter) {}

    ~HTTPRPCStats();

public:
    void markRequestBegin() { _beginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    void markRequestDone(int httpStatus) {
        _httpStatus = httpStatus;
        _doneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    }

    int64_t totalCostUs() const { return nonZeroSubtract(_doneTimeUs, _beginTimeUs); }

    void markRequestParseBegin() { _requestParseBeginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    void markRequestParseDone() { _requestParseDoneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    int64_t requestParseCostUs() const { return nonZeroSubtract(_requestParseDoneTimeUs, _requestParseBeginTimeUs); }

    void markRequestPendingBegin() { _requestPendingBeginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    void markRequestPendingDone() { _requestPendingDoneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    int64_t requestPendingCostUs() const {
        return nonZeroSubtract(_requestPendingDoneTimeUs, _requestPendingBeginTimeUs);
    }

    void markRequestCallBegin() { _requestCallBeginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    void markRequestCallDone() { _requestCallDoneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    int64_t requestCallCostUs() const { return nonZeroSubtract(_requestCallDoneTimeUs, _requestCallBeginTimeUs); }

    void markResponsePackBegin() { _responsePackBeginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    void markResponsePackDone() { _responsePackDoneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    int64_t responsePackCostUs() const { return nonZeroSubtract(_responsePackDoneTimeUs, _responsePackBeginTimeUs); }

    void markResponseSendBegin() { _responseSendBeginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    void markResponseSendDone() { _responseSendDoneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds(); }

    int64_t responseSendCostUs() const { return nonZeroSubtract(_responseSendDoneTimeUs, _responseSendBeginTimeUs); }

    // 记录response的大小
    inline void markResponseSize(size_t size) { _responseSize = size; }

    size_t responseSize() const { return _responseSize; }

    inline void markThreadPoolItemCount(size_t itemCount) { _threadPoolItemCount = itemCount; }

    size_t threadPoolItemCount() const { return _threadPoolItemCount; }

public:
    void setRequestUri(const std::string &uri) { _requestUri = uri; }

    const std::string &requestUri() const { return _requestUri; }

    void setHttpMethod(const std::string &method) { _httpMethod = method; }

    const std::string &httpMethod() const { return _httpMethod; }

    int httpStatus() const { return _httpStatus; }

    bool success() const { return _httpStatus >= 200 && _httpStatus < 300; }

private:
    // 从接收到HTTP请求，到返回HTTP响应之间的完整耗时
    int64_t _beginTimeUs{0};
    int64_t _doneTimeUs{0};

    // 将收到的HTTP请求解析为内部格式
    int64_t _requestParseBeginTimeUs{0};
    int64_t _requestParseDoneTimeUs{0};

    // HTTP请求在线程池中等待处理的时间
    int64_t _requestPendingBeginTimeUs{0};
    int64_t _requestPendingDoneTimeUs{0};

    // 调用用户提供的方法
    int64_t _requestCallBeginTimeUs{0};
    int64_t _requestCallDoneTimeUs{0};

    // 将用户返回的响应编码为HTTP格式
    int64_t _responsePackBeginTimeUs{0};
    int64_t _responsePackDoneTimeUs{0};

    // 将HTTP响应发送给客户端
    int64_t _responseSendBeginTimeUs{0};
    int64_t _responseSendDoneTimeUs{0};

    // HTTP响应大小
    size_t _responseSize{0};

    // thread pool in queue item count
    size_t _threadPoolItemCount{0};

    std::string _requestUri;
    std::string _httpMethod;
    int _httpStatus{200};

private:
    int64_t nonZeroSubtract(int64_t lhs, int64_t rhs) const {
        return (lhs > 0 && rhs > 0 && lhs > rhs) ? lhs - rhs : 0L;
    }

private:
    std::weak_ptr<KMonitorHTTPRPCServerMetricReporter> _metricReporter;

    AUTIL_LOG_DECLARE();
};

} // namespace http_arpc
