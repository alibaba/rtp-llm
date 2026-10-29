#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageCompletionAdmission.h"

namespace rtp_llm::kvcm {

class DecodePublicationAccounting: public std::enable_shared_from_this<DecodePublicationAccounting> {
public:
    using Clock = std::chrono::steady_clock;
    struct Snapshot {
        size_t  bytes{0}, jobs{0}, queued{0}, running{0}, quarantined{0};
        size_t  byte_limit{0}, job_limit{0};
        int64_t oldest_age_us{0};
    };
    // Observers must not own this accounting object. Reporting exceptions are
    // isolated from ownership, DMA completion and the inference outcome.
    using SnapshotObserver = std::function<void(const Snapshot&)>;
    using EventObserver    = std::function<void(const char*, std::optional<int64_t>)>;

private:
    enum class State {
        RESERVED,
        QUEUED,
        RUNNING,
        QUARANTINED
    };
    struct Entry {
        size_t                           bytes;
        Clock::time_point                admitted_at;
        std::optional<Clock::time_point> eos_at;
        State                            state{State::RESERVED};
        bool                             outcome_recorded{false};
    };

public:
    class Lease final: public StorageCompletionAdmission {
        friend class DecodePublicationAccounting;
        std::shared_ptr<DecodePublicationAccounting> owner_;
        uint64_t                                     id_;
        bool                                         armed_{false};

    public:
        Lease(std::shared_ptr<DecodePublicationAccounting> owner, uint64_t id): owner_(std::move(owner)), id_(id) {}
        ~Lease() override {
            if (armed_)
                owner_->release(id_);
        }
        void onQueued() noexcept override {
            owner_->transition(id_, State::QUEUED);
        }
        void onRunning() noexcept override {
            owner_->transition(id_, State::RUNNING);
        }
        void onQuarantined() noexcept override {
            owner_->transition(id_, State::QUARANTINED);
            owner_->recordOutcome(id_, "completion_unknown", false);
        }
        void recordOutcome(const char* outcome, bool acknowledged = false) noexcept {
            owner_->recordOutcome(id_, outcome, acknowledged);
        }
    };

    DecodePublicationAccounting(size_t           byte_limit,
                                size_t           job_limit,
                                SnapshotObserver snapshots = {},
                                EventObserver    events    = {}):
        byte_limit_(byte_limit), job_limit_(job_limit), snapshots_(std::move(snapshots)), events_(std::move(events)) {}
    ~DecodePublicationAccounting() {
        stopSampling();
    }

    std::shared_ptr<Lease>
    tryAcquire(size_t bytes, std::optional<Clock::time_point> eos_at, Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!accepting_ || bytes > byte_limit_ || bytes_ > byte_limit_ - bytes || entries_.size() >= job_limit_
            || next_id_ == std::numeric_limits<uint64_t>::max())
            return {};
        // An unarmed lease cannot recursively lock this mutex if allocation of
        // its map entry throws. Charge only after all allocations succeeded.
        auto lease = std::make_shared<Lease>(shared_from_this(), next_id_);
        entries_.emplace(next_id_, Entry{bytes, now, eos_at});
        ++next_id_;
        bytes_ += bytes;
        lease->armed_ = true;
        return lease;
    }

    void stopAdmission() noexcept {
        std::lock_guard<std::mutex> guard(mutex_);
        accepting_ = false;
    }

    Snapshot snapshot(Clock::time_point now = Clock::now()) const {
        std::lock_guard<std::mutex> guard(mutex_);
        Snapshot                    result;
        result.bytes      = bytes_;
        result.jobs       = entries_.size();
        result.byte_limit = byte_limit_;
        result.job_limit  = job_limit_;
        for (const auto& [id, entry] : entries_) {
            (void)id;
            switch (entry.state) {
                case State::RESERVED:
                case State::QUEUED:
                    ++result.queued;
                    break;
                case State::RUNNING:
                    ++result.running;
                    break;
                case State::QUARANTINED:
                    ++result.quarantined;
                    break;
            }
            result.oldest_age_us = std::max<int64_t>(
                result.oldest_age_us,
                std::chrono::duration_cast<std::chrono::microseconds>(now - entry.admitted_at).count());
        }
        return result;
    }

    void recordBudgetSkip() noexcept {
        if (events_) {
            try {
                events_("budget_skipped", std::nullopt);
            } catch (...) {}
        }
    }

    void reportSnapshot() noexcept {
        if (snapshots_) {
            try {
                snapshots_(snapshot());
            } catch (...) {}
        }
    }

    // One bounded CPU sampler only when an enabled D backend has a reporter.
    // Idle/stuck task age therefore remains observable without inference steps.
    void startSampling() {
        if (!snapshots_ || sampler_.joinable())
            return;
        sampler_ = std::thread([this] {
            pthread_setname_np(pthread_self(), "kvcm_dpub_stats");
            std::unique_lock<std::mutex> lock(sampler_mutex_);
            while (!stopping_) {
                lock.unlock();
                reportSnapshot();
                lock.lock();
                sampler_cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stopping_; });
            }
        });
    }
    void stopSampling() noexcept {
        {
            std::lock_guard<std::mutex> guard(sampler_mutex_);
            stopping_ = true;
        }
        sampler_cv_.notify_all();
        if (sampler_.joinable())
            sampler_.join();
    }

private:
    void transition(uint64_t id, State next) noexcept {
        std::lock_guard<std::mutex> guard(mutex_);
        auto                        it = entries_.find(id);
        if (it != entries_.end() && static_cast<int>(next) > static_cast<int>(it->second.state))
            it->second.state = next;
    }
    void recordOutcome(uint64_t id, const char* outcome, bool acknowledged) noexcept {
        std::optional<int64_t> latency;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto                        it = entries_.find(id);
            if (it == entries_.end() || it->second.outcome_recorded)
                return;
            it->second.outcome_recorded = true;
            if (acknowledged && it->second.eos_at)
                latency = std::max<int64_t>(
                    0,
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - *it->second.eos_at).count());
        }
        if (events_) {
            try {
                events_(outcome, latency);
            } catch (...) {}
        }
    }
    void release(uint64_t id) noexcept {
        // No earlier explicit acknowledgement/outcome means failure/cancellation
        // before successful publication. Record exactly once per admission.
        recordOutcome(id, "failed", false);
        std::lock_guard<std::mutex> guard(mutex_);
        const auto                  it = entries_.find(id);
        if (it != entries_.end()) {
            bytes_ -= it->second.bytes;
            entries_.erase(it);
        }
    }

    mutable std::mutex        mutex_;
    size_t                    byte_limit_, job_limit_, bytes_{0};
    uint64_t                  next_id_{0};
    std::map<uint64_t, Entry> entries_;
    SnapshotObserver          snapshots_;
    EventObserver             events_;
    std::mutex                sampler_mutex_;
    std::condition_variable   sampler_cv_;
    bool                      stopping_{false};
    bool                      accepting_{true};
    std::thread               sampler_;
};

}  // namespace rtp_llm::kvcm
