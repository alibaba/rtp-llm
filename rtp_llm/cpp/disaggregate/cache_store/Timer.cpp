#include "rtp_llm/cpp/disaggregate/cache_store/Timer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"

namespace rtp_llm {

Timer::Timer(int64_t expiredTimeUs, Callback&& callback):
    _stoped(false), _expiredTimeUs(expiredTimeUs), _callback(callback) {}

Timer::~Timer() {}

void Timer::stop() {
    if (_stoped) {
        return;
    }

    autil::ScopedLock lock(_mutex);
    _stoped   = true;
    _callback = nullptr;
}

void Timer::callback() {
    Callback callback;
    {
        autil::ScopedLock lock(_mutex);
        if (_stoped || !_callback) {
            return;
        }
        _stoped   = true;
        callback  = std::move(_callback);
        _callback = nullptr;
    }
    if (callback) {
        callback();
    }
}
}  // namespace rtp_llm
