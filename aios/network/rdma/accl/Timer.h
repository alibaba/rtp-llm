#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "autil/Lock.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class TimerManager;

class Timer {
public:
    using Callback = std::function<void()>;
    Timer(int64_t expiredTimeUs, Callback&& callback);
    ~Timer();

public:
    // user thread may call stop to stop callback
    void stop();
    // timer thread may call callback to notify timeout
    void callback();

    double getExpiredTimeUs() const {
        return _expiredTimeUs;
    }

private:
    autil::ThreadMutex _mutex;
    bool               _stoped;
    int64_t            _expiredTimeUs;
    // callback should call at most once
    // after stop called, callback should be clear
    Callback _callback;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);