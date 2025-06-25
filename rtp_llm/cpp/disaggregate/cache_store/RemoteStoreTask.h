#pragma once

#include <string>
#include <unistd.h>
#include <map>
#include <functional>
#include <memory>

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MessagerRequest.h"

namespace rtp_llm {

class RemoteStoreTask {
public:
    typedef std::function<bool()> CheckCancelFunc;

public:
    RemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>& request, CheckCancelFunc check_cancel_func):
        request_(request), check_cancel_func_(check_cancel_func){};

public:
    virtual void     waitDone()      = 0;
    virtual bool     success() const = 0;
    const ErrorInfo& getErrorInfo() const {
        return error_info_;
    }
    std::string getRequestId() const {
        return request_ != nullptr ? request_->request_id : "";
    }

protected:
    std::shared_ptr<RemoteStoreRequest> request_;
    CheckCancelFunc                     check_cancel_func_;
    ErrorInfo                           error_info_;
};

}  // namespace rtp_llm
