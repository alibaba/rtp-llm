#pragma once

#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"

namespace http_arpc {

class HTTPRPCControllerBase : public RPCController {
public:
    virtual void SetErrorCode(int32_t errorCode) = 0;
    virtual int32_t GetErrorCode() = 0;
    virtual void SetQueueTime(int64_t beginTime) = 0;
    virtual int64_t GetQueueTime() const = 0;
    virtual void SetAddr(const std::string &addr) = 0;
    virtual const std::string &GetAddr() const = 0;
};

} // namespace http_arpc

