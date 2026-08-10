#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// AutoTPM Cancel: cancel-intent map, request_id -> {reason, arrival_time}.
// Writer: the Prefill Cancel RPC handler records the intent and acks ACCEPTED
// immediately ("intent registered"); the master's release decision relies on
// WorkerStatus finished records, never on the ack.
// Consumers (cooperative checkpoints on existing code paths, no extra thread):
//   R1 enqueue checkpoint: reject a matching request before stream creation;
//   R2 schedule checkpoint: FIFOScheduler::schedule() stops matching streams;
//   R3 TTL sweep: the schedule loop drops unconsumed entries after kTtlMs.
// Consumption must go through tryConsume() (atomic match + erase) — a bare
// match() + erase() pair is a TOCTOU: a duplicate cancel can overwrite the
// entry between the two calls, and erase() would silently drop the rewritten
// intent. R1 consumes first and then acts (its reject has no failure path);
// R2 confirms via read-only match(), stops the stream (idempotent), and only
// then tryConsume()s, so a failed stop can never lose an intent.
class CancelIntentMap {
public:
    struct Entry {
        ErrorCode terminal_code   = ErrorCode::CANCELLED;
        int64_t   arrival_time_ms = 0;
    };

    static constexpr size_t  kMaxEntries = 10000;
    static constexpr int64_t kTtlMs      = 30 * 1000;

    // Duplicate cancel overwrites; when at capacity, evict the oldest entry.
    void registerCancel(int64_t request_id, ErrorCode terminal_code, int64_t now_ms) {
        std::lock_guard<std::mutex> lock(mu_);
        if (entries_.size() >= kMaxEntries && entries_.find(request_id) == entries_.end()) {
            auto oldest = entries_.begin();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (it->second.arrival_time_ms < oldest->second.arrival_time_ms) {
                    oldest = it;
                }
            }
            entries_.erase(oldest);
        }
        entries_[request_id] = Entry{terminal_code, now_ms};
    }

    // Returns the entry for the request id, if any. Read-only: consumers act
    // on the hit and then remove the entry via tryConsume(), never via a bare
    // erase().
    std::optional<Entry> match(int64_t request_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = entries_.find(request_id);
        if (it == entries_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    // Atomic match + erase under one lock. On a hit the entry is removed and
    // returned; otherwise nullopt and the map is untouched. This closes the
    // match()/erase() TOCTOU window: a duplicate cancel that overwrites the
    // entry between a stand-alone match() and the removal is consumed here as
    // one atomic step, never dropped by a blind erase().
    std::optional<Entry> tryConsume(int64_t request_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = entries_.find(request_id);
        if (it == entries_.end()) {
            return std::nullopt;
        }
        Entry entry = it->second;
        entries_.erase(it);
        return entry;
    }

    void erase(int64_t request_id) {
        std::lock_guard<std::mutex> lock(mu_);
        entries_.erase(request_id);
    }

    // R3: drop unconsumed entries older than kTtlMs.
    void sweepExpired(int64_t now_ms) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (now_ms - it->second.arrival_time_ms > kTtlMs) {
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_.size();
    }

private:
    mutable std::mutex                 mu_;
    std::unordered_map<int64_t, Entry> entries_;
};

}  // namespace rtp_llm
