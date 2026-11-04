#pragma once

#include <list>

#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCControllerBase.h"

namespace http_arpc {

class HTTPRPCController : public HTTPRPCControllerBase {
public:
    HTTPRPCController();
    ~HTTPRPCController();

public:
    /* override */ void Reset();
    /* override */ bool Failed() const;
    /* override */ std::string ErrorText() const;
    /* override */ void StartCancel();
    /* override */ void SetFailed(const std::string &reason);
    /* override */ bool IsCanceled() const;
    /* override */ void NotifyOnCancel(RPCClosure *callback);

    /* override */ void SetErrorCode(int32_t errorCode);
    /* override */ int32_t GetErrorCode();
    /* override */ void SetQueueTime(int64_t beginTime);
    /* override */ int64_t GetQueueTime() const;
    /* override */ void SetAddr(const std::string &addr);
    /* override */ const std::string &GetAddr() const;

private:
    std::string _reason;
    std::list<RPCClosure *> _cancelList;
    int32_t _errorCode;
    bool _failed;
    bool _canceled;
    int64_t _queueTime;
    std::string _addr;
};

} // namespace http_arpc

