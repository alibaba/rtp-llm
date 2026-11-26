#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace navi_sched {

// Local adapter state only. Covers payload preparation, queueing, and Python
// handoff until its measured-work admission checkpoint has been captured.
class ForwardAdmissionClock {
    using Key = std::pair<int64_t, uint64_t>;
    struct Shared {
        explicit Shared(std::function<int64_t()> source) : now(std::move(source)) {}
        std::mutex mutex;
        std::set<Key> retained;
        uint64_t nextId = 0;
        std::function<int64_t()> now;
    };

public:
    class Ticket {
        friend class ForwardAdmissionClock;
    public:
        Ticket(const Ticket &) = delete;
        Ticket &operator=(const Ticket &) = delete;
        ~Ticket() { release(); }

        int64_t readyTimeNs() const { return _key.first; }

        void release()
        {
            if (!_registered || _released.exchange(true, std::memory_order_acq_rel)) {
                return;
            }
            std::lock_guard<std::mutex> lock(_shared->mutex);
            _shared->retained.erase(_key);
        }

    private:
        explicit Ticket(std::shared_ptr<Shared> state) : _shared(std::move(state)) {}
        std::shared_ptr<Shared> _shared;
        Key _key{0, 0};
        bool _registered = false;
        std::atomic<bool> _released{false};
    };

    explicit ForwardAdmissionClock(std::function<int64_t()> now = steadyNowNs)
        : _shared(std::make_shared<Shared>(std::move(now)))
    {
    }

    std::shared_ptr<Ticket> open()
    {
        auto ticket = std::shared_ptr<Ticket>(new Ticket(_shared));
        std::lock_guard<std::mutex> lock(_shared->mutex);
        // Sample while holding the same mutex as watermark(). A new callback
        // cannot publish a timestamp earlier than an already returned watermark.
        const int64_t ready = _shared->now();
        if (ready < 0 || _shared->nextId == UINT64_MAX) {
            throw std::overflow_error("invalid forward admission clock");
        }
        ticket->_key = {ready, ++_shared->nextId};
        _shared->retained.emplace(ticket->_key);
        ticket->_registered = true;
        return ticket;
    }

    int64_t watermark() const
    {
        std::lock_guard<std::mutex> lock(_shared->mutex);
        return _shared->retained.empty() ? _shared->now()
                                         : _shared->retained.begin()->first;
    }

    size_t retainedCount() const
    {
        std::lock_guard<std::mutex> lock(_shared->mutex);
        return _shared->retained.size();
    }

private:
    static int64_t steadyNowNs()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    std::shared_ptr<Shared> _shared;
};

} // namespace navi_sched
