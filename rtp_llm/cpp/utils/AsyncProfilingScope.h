#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

struct AsyncProfileEvent {
    std::string name;
    int64_t     start_us;
    int64_t     end_us;
};

// The owner allocates a bounded buffer once. Workers publish POD records without
// allocations, session mutexes or Kineto TLS; only the owner formats names.
class AsyncProfileSession {
public:
    static int64_t nowUs() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
    static int64_t steadyNowUs() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    struct Record {
        const char* stage{"unknown"};
        const char* path{"disk_to_device"};
        const char* literal_name{nullptr};
        int64_t     request_id{-1};
        size_t      begin{0};
        size_t      end{0};
        int64_t     rank{-1};
        uint64_t    transfer_id{0};
        int64_t     start_us{0};
    };
    struct Entry {
        Record               record;
        std::atomic<int64_t> end_us{0};
        std::atomic<bool>    ready{false};
    };

    explicit AsyncProfileSession(int64_t start_us = nowUs(), size_t limit = 100000):
        start_us_(start_us),
        steady_start_us_(steadyNowUs()),
        limit_(limit),
        generation_(nextGeneration().fetch_add(1, std::memory_order_relaxed)),
        entries_(new Entry[limit]) {}

    static std::shared_ptr<AsyncProfileSession> start(size_t limit = 100000) noexcept {
        try {
            auto                        session = std::make_shared<AsyncProfileSession>(nowUs(), limit);
            std::lock_guard<std::mutex> lock(activationMutex());
            std::atomic_store(&active(), session);
            activeGeneration().store(session->generation_, std::memory_order_release);
            return session;
        } catch (...) {
            return {};
        }
    }
    static bool enabled() noexcept {
        return activeGeneration().load(std::memory_order_acquire) != 0;
    }
    static std::shared_ptr<AsyncProfileSession> current() noexcept {
        // Avoid libstdc++'s shared_ptr atomic lock while profiling is off.
        return enabled() ? std::atomic_load(&active()) : std::shared_ptr<AsyncProfileSession>{};
    }
    int64_t toWallUs(int64_t steady_us) const noexcept {
        return start_us_ + (steady_us - steady_start_us_);
    }

    Entry* add(Record record) noexcept {
        if (closed_.load(std::memory_order_acquire))
            return nullptr;
        if (saturated_.load(std::memory_order_relaxed)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        auto index = size_.load(std::memory_order_relaxed);
        for (size_t attempt = 0; index < limit_ && attempt < 16; ++attempt) {
            if (size_.compare_exchange_weak(index, index + 1, std::memory_order_acq_rel)) {
                auto& entry  = entries_[index];
                entry.record = record;
                entry.ready.store(true, std::memory_order_release);
                return &entry;
            }
        }
        if (index >= limit_)
            saturated_.store(true, std::memory_order_relaxed);
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // As with stage/path, this borrowed name must outlive the session.
    Entry* add(const char* literal_name, int64_t start_us) noexcept {
        Record r;
        r.literal_name = literal_name;
        r.start_us     = start_us;
        return add(r);
    }
    std::vector<AsyncProfileEvent> finish(int64_t stop_us = nowUs()) {
        if (closed_.exchange(true, std::memory_order_acq_rel))
            return {};
        {
            std::lock_guard<std::mutex> lock(activationMutex());
            auto                        expected = std::atomic_load(&active());
            if (expected.get() == this
                && std::atomic_compare_exchange_strong(&active(), &expected, std::shared_ptr<AsyncProfileSession>{})) {
                auto generation = generation_;
                activeGeneration().compare_exchange_strong(generation, 0, std::memory_order_acq_rel);
            }
        }
        const auto                     count = std::min(size_.load(std::memory_order_acquire), limit_);
        std::vector<AsyncProfileEvent> events;
        events.reserve(count + 1);
        auto dropped = dropped_.load(std::memory_order_relaxed);
        for (size_t index = 0; index < count; ++index) {
            const auto& entry = entries_[index];
            // A concurrent unpublished reservation is skipped, never waited on.
            if (!entry.ready.load(std::memory_order_acquire)) {
                ++dropped;
                continue;
            }
            const auto end         = entry.end_us.load(std::memory_order_acquire);
            const auto begin       = std::max(start_us_, entry.record.start_us);
            const auto clipped_end = end == 0 ? stop_us : std::min(end, stop_us);
            if (clipped_end >= begin) {
                auto name = formatName(entry.record);
                if (end == 0 || end > stop_us)
                    name += " incomplete=1";
                events.push_back({std::move(name), begin, clipped_end});
            }
        }
        if (dropped)
            events.push_back({"cache.transfer.profiler.dropped(count=" + std::to_string(dropped) + ")",
                              start_us_,
                              std::max(start_us_, stop_us)});
        return events;
    }

private:
    static std::string formatName(const Record& r) {
        if (r.literal_name)
            return r.literal_name;
        return std::string("cache.transfer.") + r.path + "." + r.stage + "(request_id=" + std::to_string(r.request_id)
               + ",batch=" + std::to_string(r.begin) + ":" + std::to_string(r.end) + ",rank=" + std::to_string(r.rank)
               + ",transfer_id=" + std::to_string(r.transfer_id) + ")";
    }
    static std::shared_ptr<AsyncProfileSession>& active() {
        static std::shared_ptr<AsyncProfileSession> s;
        return s;
    }
    static std::mutex& activationMutex() {
        static std::mutex mutex;
        return mutex;
    }
    static std::atomic<uint64_t>& activeGeneration() {
        static std::atomic<uint64_t> g{0};
        return g;
    }
    static std::atomic<uint64_t>& nextGeneration() {
        static std::atomic<uint64_t> g{1};
        return g;
    }
    const int64_t            start_us_, steady_start_us_;
    const size_t             limit_;
    const uint64_t           generation_;
    std::unique_ptr<Entry[]> entries_;
    std::atomic<size_t>      size_{0};
    std::atomic<uint64_t>    dropped_{0};
    std::atomic<bool>        saturated_{false}, closed_{false};
};

class AsyncProfileSpan {
public:
    AsyncProfileSpan() = default;
    AsyncProfileSpan(const char* stage,
                     int64_t     request_id,
                     size_t      begin,
                     size_t      end,
                     int64_t     start_us    = 0,
                     int64_t     rank        = -1,
                     uint64_t    transfer_id = 0,
                     const char* path        = "disk_to_device") noexcept:
        record_{stage, path, nullptr, request_id, begin, end, rank, transfer_id, start_us},
        steady_start_us_(start_us == 0 ? AsyncProfileSession::steadyNowUs() : 0),
        captured_(false) {
        capture();
    }
    static AsyncProfileSpan fromSteady(const char* stage,
                                       int64_t     request_id,
                                       size_t      begin,
                                       size_t      end,
                                       int64_t     start,
                                       int64_t     rank,
                                       uint64_t    id,
                                       const char* path) noexcept {
        AsyncProfileSpan span;
        span.record_          = {stage, path, nullptr, request_id, begin, end, rank, id, 0};
        span.steady_start_us_ = start;
        span.captured_        = false;
        span.capture();
        return span;
    }
    void finish(int64_t end_us = 0) const noexcept {
        if (entry_)
            finishEntry(entry_, end_us == 0 ? owner_->toWallUs(AsyncProfileSession::steadyNowUs()) : end_us);
        else if (!captured_ && AsyncProfileSession::enabled()) {
            if (auto session = AsyncProfileSession::current()) {
                auto r = record_;
                if (r.start_us == 0)
                    r.start_us = session->toWallUs(steady_start_us_);
                finishEntry(session->add(r),
                            end_us == 0 ? session->toWallUs(AsyncProfileSession::steadyNowUs()) : end_us);
            }
        }
    }
    void finishAtSteady(int64_t end_us) const noexcept {
        if (entry_)
            finishEntry(entry_, owner_->toWallUs(end_us));
        else if (!captured_ && AsyncProfileSession::enabled()) {
            if (auto session = AsyncProfileSession::current()) {
                auto r = record_;
                if (r.start_us == 0)
                    r.start_us = session->toWallUs(steady_start_us_);
                finishEntry(session->add(r), session->toWallUs(end_us));
            }
        }
    }

private:
    void capture() noexcept {
        if (auto session = AsyncProfileSession::current()) {
            captured_ = true;
            if (record_.start_us == 0)
                record_.start_us = session->toWallUs(steady_start_us_);
            entry_ = session->add(record_);
            if (entry_)
                owner_ = std::move(session);
        }
    }
    static void finishEntry(AsyncProfileSession::Entry* entry, int64_t end_us) noexcept {
        if (entry) {
            int64_t expected = 0;
            entry->end_us.compare_exchange_strong(expected, end_us, std::memory_order_release);
        }
    }
    AsyncProfileSession::Record          record_;
    int64_t                              steady_start_us_{0};
    bool                                 captured_{true};
    std::shared_ptr<AsyncProfileSession> owner_;
    AsyncProfileSession::Entry*          entry_{nullptr};
};
}  // namespace rtp_llm
