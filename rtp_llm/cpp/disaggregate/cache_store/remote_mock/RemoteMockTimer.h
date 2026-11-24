#pragma once
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace rtp_llm {
class RemoteMockTimer {
public:
    using Clock    = std::chrono::steady_clock;
    using Now      = std::function<Clock::time_point()>;
    using Callback = std::function<void(bool cancelled)>;
    // A fake clock uses threaded=false and pump(). Only one pump caller at a time.
    explicit RemoteMockTimer(size_t capacity = 128, Now now = Clock::now, bool threaded = true);
    ~RemoteMockTimer();
    RemoteMockTimer(const RemoteMockTimer&)            = delete;
    RemoteMockTimer& operator=(const RemoteMockTimer&) = delete;
    // False means not admitted, and the callback is not invoked. Callbacks must be short,
    // must not destroy the timer, and may schedule or stop it. Exceptions are contained.
    bool schedule(Clock::time_point deadline, Callback callback);
    void pump();
    // Cancels queued callbacks exactly once outside the mutex. An already extracted
    // callback can finish normally. External stop joins the worker before returning.
    void stop();

private:
    struct Entry {
        Clock::time_point deadline;
        uint64_t          sequence;
        mutable Callback  callback;
    };
    struct Later {
        bool operator()(const Entry& a, const Entry& b) const {
            return a.deadline != b.deadline ? a.deadline > b.deadline : a.sequence > b.sequence;
        }
    };
    static void                                           invoke(Callback& callback, bool cancelled) noexcept;
    void                                                  run();
    const size_t                                          capacity_;
    const Now                                             now_;
    std::mutex                                            mutex_;
    std::mutex                                            join_mutex_;
    std::condition_variable                               condition_;
    std::priority_queue<Entry, std::vector<Entry>, Later> queue_;
    uint64_t                                              sequence_ = 0;
    bool                                                  stopped_  = false;
    std::thread                                           thread_;
    std::thread::id                                       worker_id_;
};
}  // namespace rtp_llm
