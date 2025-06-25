#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

TimerManager::TimerManager(const std::string& threadName, int64_t timeoutIntevalUs) {
    _timerThread =
        autil::LoopThread::createLoopThread(std::bind(&TimerManager::timeoutProc, this), timeoutIntevalUs, threadName);
}

TimerManager::~TimerManager() {}

std::shared_ptr<Timer> TimerManager::addTimer(int64_t timeoutMs, Timer::Callback&& callback) {
    auto expiredTimeUs = timeoutMs * 1000 + currentTimeUs();
    auto timer         = std::make_shared<Timer>(expiredTimeUs, std::move(callback));

    {
        autil::ScopedWriteLock lock(_rwlock);
        _timeoutCheckTimers[expiredTimeUs] = timer;
    }
    return timer;
}

size_t TimerManager::getActiveTimerCount() const {
    autil::ScopedReadLock lock(_rwlock);
    return _timeoutCheckTimers.size();
}

void TimerManager::timeoutProc() {
    auto current_time_us = currentTimeUs();

    std::map<int64_t, std::shared_ptr<Timer>> timeoutTimers;
    {
        // extract all timeout timer
        autil::ScopedWriteLock lock(_rwlock);
        auto                   upperIter = _timeoutCheckTimers.upper_bound(current_time_us);
        timeoutTimers.insert(_timeoutCheckTimers.begin(), upperIter);
        _timeoutCheckTimers.erase(_timeoutCheckTimers.begin(), upperIter);
    }

    // callback may be set to thread pool for nonblock call
    for (auto& iter : timeoutTimers) {
        iter.second->callback();
    }
}

}  // namespace rtp_llm