#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <list>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Local completion evidence keyed by the caller's unique execution trace.
// Completed entries remain queryable until explicitly forgotten or this registry is destroyed.
// Missing executions remain barred until their sole late arrival is rejected,
// or their retention expires. Execution IDs are never resent
// to the same worker within the worker group's lifetime.
// Ordinary transfer reuses its numeric trace ID; KVCM can reuse its existing string
// identity in a separate registry instance. No per-task identity is generated here.
// The worker group shares one lifetime; records are not reused across instances.
class TransferTaskRegistry {
public:
    enum class State { UNKNOWN, RUNNING, BUFFER_SAFE, REJECTED };
    // FAILED is reserved for routing or protocol failures outside this registry.
    enum class ForgetResult { FAILED, DONE, RUNNING, INVALID };

    using Clock = std::chrono::steady_clock;
    using Now = std::function<Clock::time_point()>;

    explicit TransferTaskRegistry(Now now = [] { return Clock::now(); }): epoch_(makeEpoch()), now_(std::move(now)) {
        logEvent("create", "", State::UNKNOWN, "created");
    }

    ~TransferTaskRegistry() {
        logEvent("destroy", "", State::UNKNOWN, "destroyed");
    }

    const std::string& epoch() const noexcept {
        return epoch_;
    }

    // Includes pending/completed executions and retained missing records.
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.size();
    }

    // Insert RUNNING without replacing an existing execution's evidence.
    // False reports an empty or already registered trace to the caller.
    bool begin(const std::string& trace_id) {
        if (trace_id.empty()) {
            logEvent("begin", trace_id, State::UNKNOWN, "invalid");
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto inserted = records_.emplace(trace_id, State::RUNNING);
        if (!inserted.second && inserted.first->second == State::REJECTED) {
            // The fence has intercepted its sole possible late execution. Consume
            // it atomically, but reject this arrival without admitting any I/O.
            const auto missing = missing_.find(trace_id);
            missing_queue_.erase(missing->second.position);
            missing_.erase(missing);
            records_.erase(inserted.first);
            logged_query_states_.erase(trace_id);
            logEvent("begin", trace_id, State::REJECTED, "late_execution_rejected_and_removed");
            return false;
        }
        logEvent("begin", trace_id, inserted.first->second, inserted.second ? "inserted" : "duplicate");
        return inserted.second;
    }

    bool begin(uint64_t trace_id) {
        return trace_id != 0 && begin(std::to_string(trace_id));
    }

    // buffer_safe is an explicit I/O completion proof, not business success.
    // False preserves the last known state. A true proof is monotonic and can
    // describe a safely drained failure. The return value only reports ID presence.
    bool finish(const std::string& trace_id, bool buffer_safe) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(trace_id);
        if (it == records_.end()) {
            logEvent("finish", trace_id, State::UNKNOWN, "missing");
            return false;
        }
        if (it->second == State::REJECTED) {
            // A late completion must not turn a missing record into a safety proof.
            touchMissing(trace_id, "finish");
            logEvent("finish", trace_id, it->second, "rejection_retained");
            return true;
        }
        if (buffer_safe) {
            it->second = State::BUFFER_SAFE;
        }
        logEvent("finish", trace_id, it->second, buffer_safe ? "safe_proof" : "no_safe_proof");
        return true;
    }

    bool finish(uint64_t trace_id, bool buffer_safe) {
        return trace_id != 0 && finish(std::to_string(trace_id), buffer_safe);
    }

    // Admitted executions require safety proof before deletion. Absent records
    // are already forgotten; a later finish never recreates them. Missing fences
    // survive client cleanup: DONE lets the caller retire its cleanup record,
    // while the server still rejects a late execution until arrival or expiry.
    ForgetResult forget(const std::string& trace_id) {
        if (trace_id.empty()) {
            logEvent("forget", trace_id, State::UNKNOWN, "invalid");
            return ForgetResult::INVALID;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(trace_id);
        if (it == records_.end()) {
            logEvent("forget", trace_id, State::UNKNOWN, "already_missing");
            return ForgetResult::DONE;
        }
        if (it->second == State::REJECTED) {
            logEvent("forget", trace_id, State::REJECTED, "fence_retained_cleanup_acknowledged");
            return ForgetResult::DONE;
        }
        if (it->second != State::BUFFER_SAFE) {
            logEvent("forget", trace_id, it->second, "retained");
            return ForgetResult::RUNNING;
        }
        records_.erase(it);
        logEvent("forget", trace_id, State::BUFFER_SAFE, "removed");
        return ForgetResult::DONE;
    }

    ForgetResult forget(uint64_t trace_id) {
        return trace_id == 0 ? ForgetResult::INVALID : forget(std::to_string(trace_id));
    }

    // Recovery queries close admission for missing identities atomically with
    // begin(). Either execution registered first and remains RUNNING, or this
    // query installs a retention fence before reporting that no I/O can start.
    // Ordinary diagnostic queryLocal() remains read-only.
    std::vector<State> queryAndBlockMissing(const std::vector<std::string>& trace_ids) {
        std::vector<State> states(trace_ids.size(), State::UNKNOWN);
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < trace_ids.size(); ++i) {
            if (trace_ids[i].empty()) {
                logQuery(trace_ids[i], states[i]);
                continue;
            }
            const auto inserted = records_.emplace(trace_ids[i], State::REJECTED);
            states[i] = inserted.first->second;
            if (states[i] == State::REJECTED)
                touchMissing(trace_ids[i], "query");
            if (inserted.second)
                logEvent("reject", trace_ids[i], states[i], "missing_admission_closed");
            logQuery(trace_ids[i], states[i]);
        }
        return states;
    }

    std::vector<State> queryAndBlockMissing(const std::vector<uint64_t>& trace_ids) {
        std::vector<std::string> keys;
        keys.reserve(trace_ids.size());
        for (const auto trace_id : trace_ids)
            keys.push_back(trace_id == 0 ? std::string{} : std::to_string(trace_id));
        return queryAndBlockMissing(keys);
    }

    // The worker group has one shared lifetime. Read-only queries of this
    // instance preserve request order and never create missing records.
    std::vector<State> queryLocal(const std::vector<std::string>& trace_ids) const {
        std::vector<State> states(trace_ids.size(), State::UNKNOWN);
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < trace_ids.size(); ++i) {
            const auto it = records_.find(trace_ids[i]);
            if (it != records_.end()) {
                states[i] = it->second;
            }
            logQuery(trace_ids[i], states[i]);
        }
        return states;
    }

    std::vector<State> queryLocal(const std::vector<uint64_t>& trace_ids) const {
        std::vector<std::string> keys;
        keys.reserve(trace_ids.size());
        for (const auto trace_id : trace_ids) {
            // Numeric zero is never an identity, even if a string caller has
            // registered the literal "0". Empty string records cannot exist.
            keys.push_back(trace_id == 0 ? std::string{} : std::to_string(trace_id));
        }
        return queryLocal(keys);
    }

    // Retain the explicit epoch check for callers that already hold one.
    std::vector<State> query(const std::string& expected_epoch, const std::vector<std::string>& trace_ids) const {
        return expected_epoch == epoch_ ? queryLocal(trace_ids) : std::vector<State>(trace_ids.size(), State::UNKNOWN);
    }

    std::vector<State> query(const std::string& expected_epoch, const std::vector<uint64_t>& trace_ids) const {
        return expected_epoch == epoch_ ? queryLocal(trace_ids) : std::vector<State>(trace_ids.size(), State::UNKNOWN);
    }

    // Called once per minute by cache maintenance. Only visit missing records,
    // rotating survivors so every record gets checked without a full-table scan.
    // A busy request path takes priority; defer this batch rather than wait.
    size_t cleanupExpiredMissing(size_t budget = 128) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock())
            return 0;
        const auto now = now_();
        const auto count = std::min(budget, missing_queue_.size());
        size_t removed = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto first = missing_queue_.begin();
            const auto missing = missing_.find(*first);
            if (missing->second.expire_time <= now) {
                records_.erase(*first);
                logged_query_states_.erase(*first);
                missing_.erase(missing);
                missing_queue_.erase(first);
                ++removed;
            } else {
                missing_queue_.splice(missing_queue_.end(), missing_queue_, first);
            }
        }
        return removed;
    }

private:
    struct MissingRecord {
        uint64_t count{0};
        Clock::time_point expire_time{};
        Clock::time_point last_warning_time{};
        bool warned{false};
        std::list<std::string>::iterator position;
    };

    // Caller holds mutex_. Diagnostic reads do not extend retention.
    void touchMissing(const std::string& trace_id, const char* source) {
        const auto now = now_();
        auto inserted = missing_.try_emplace(trace_id);
        auto& record = inserted.first->second;
        if (inserted.second) {
            missing_queue_.push_back(trace_id);
            record.position = std::prev(missing_queue_.end());
            record.expire_time = now;
        }
        if (record.count != std::numeric_limits<uint64_t>::max())
            ++record.count;
        const auto base = std::max(record.expire_time, now);
        record.expire_time = base > Clock::time_point::max() - std::chrono::hours(1) ?
                                 Clock::time_point::max() : base + std::chrono::hours(1);
        if (record.expire_time - now > std::chrono::hours(24)
            && (!record.warned || now - record.last_warning_time >= std::chrono::hours(1))) {
            record.warned = true;
            record.last_warning_time = now;
            try {
                RTP_LLM_LOG_WARNING(
                    "transfer missing repeatedly accessed trace=%s registry=%s source=%s count=%lu remaining_seconds=%ld",
                    trace_id.c_str(), epoch_.c_str(), source, static_cast<unsigned long>(record.count),
                    static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(record.expire_time - now).count()));
            } catch (...) {
                // Diagnostics must not affect admission or completion evidence.
            }
        }
    }

    // Log under the registry lock so begin/finish/forget/query ordering is
    // unambiguous. Diagnostics must never affect completion evidence.
    void logEvent(const char* event, const std::string& trace_id, State state, const char* result) const noexcept {
        try {
            const char* state_name = state == State::BUFFER_SAFE ? "BUFFER_SAFE" :
                                     state == State::RUNNING ? "RUNNING" :
                                     state == State::REJECTED ? "REJECTED" : "UNKNOWN";
            RTP_LLM_LOG_INFO("transfer_lifecycle event=registry_%s trace=%s registry=%s state=%s result=%s",
                              event, trace_id.c_str(), epoch_.c_str(), state_name, result);
        } catch (...) {
            // In particular, logging must not interrupt the safety callback.
        }
    }

    void logQuery(const std::string& trace_id, State state) const noexcept {
        try {
            if (!Logger::getEngineLogger().isLevelEnabled(alog::LOG_LEVEL_INFO)) {
                return;
            }
            const auto previous = logged_query_states_.find(trace_id);
            if (previous != logged_query_states_.end() && previous->second == state) {
                return;
            }
            // Emit the first observation and state changes, not every recovery
            // poll. Bound diagnostic memory even for queries of unknown traces.
            if (previous == logged_query_states_.end() && logged_query_states_.size() >= 4096) {
                logged_query_states_.clear();
            }
            logged_query_states_[trace_id] = state;
            logEvent("query", trace_id, state, state == State::UNKNOWN ? "missing" : "found");
        } catch (...) {
        }
    }

    static std::string makeEpoch() {
        std::random_device random;
        std::string        epoch = "transfer-registry-";
        for (int i = 0; i < 4; ++i) {
            epoch += std::to_string(random()) + "-";
        }
        return epoch;
    }

    const std::string                     epoch_;
    const Now                            now_;
    mutable std::mutex                    mutex_;
    std::list<std::string>                missing_queue_;
    std::unordered_map<std::string, MissingRecord> missing_;
    std::unordered_map<std::string, State> records_;
    mutable std::unordered_map<std::string, State> logged_query_states_;
};

}  // namespace rtp_llm
