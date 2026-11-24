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
        request_(request), check_cancel_func_(check_cancel_func) {};

public:
    virtual void waitDone()      = 0;
    virtual bool success() const = 0;
    using DoneCallback           = std::function<void(const ErrorInfo&)>;
    using DrainedCallback        = std::function<void()>;

    // Optional one-shot hooks. Late registration is notified immediately; false
    // means unsupported, empty callback, or a hook that was already registered.
    virtual bool setDoneCallback(DoneCallback callback) {
        return false;
    }
    // Fires once terminal and every admitted RPC has completed its callback
    // work, even after early failure. Safe point to release registered buffers.
    virtual bool setDrainedCallback(DrainedCallback callback) {
        return false;
    }
    // Stops admission immediately; dispatched RPCs must still drain. Idempotent.
    virtual bool cancel(const ErrorInfo& error) {
        return false;
    }

    // Valid after terminal completion (waitDone or the done callback).
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
