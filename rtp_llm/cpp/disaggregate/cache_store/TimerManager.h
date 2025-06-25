#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/Timer.h"
#include "autil/Lock.h"
#include "autil/LoopThread.h"

namespace rtp_llm {

class TimerManager {
public:
    TimerManager(const std::string& threadName = "TimerManager", int64_t timeoutIntervalUs = 2500);
    ~TimerManager();

public:
    std::shared_ptr<Timer> addTimer(int64_t timeoutMs, Timer::Callback&& callback);
    size_t                 getActiveTimerCount() const;

private:
    void timeoutProc();

private:
    mutable autil::ReadWriteLock              _rwlock;
    std::map<int64_t, std::shared_ptr<Timer>> _timeoutCheckTimers;

    autil::LoopThreadPtr _timerThread;
};

}  // namespace rtp_llm