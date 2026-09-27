#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>
namespace rtp_llm {

// Aggregates in-flight counters from every layer (frontend, rpc, scheduler,
// cache loading, connector, p2p, cache store) and decides whether the engine is
// fully drained. Counter sources are injected as named providers instead of hard
// dependencies on concrete classes, so unit tests mock them and integration
// wires the real getters
// (e.g. KVCacheConnectorCoordinator::inflightTransferCount,
// NormalCacheStore::activeTransferCount).
//
// Drain policies:
//   - wait: waitDrained(timeout_ms) polls until all counters reach zero or
//     the timeout expires. On timeout it returns false and the caller stays in
//     DRAINING without releasing GPU resources.
//   - cancel: an injected callback is invoked first, then drain is awaited.
//     The operation owner defines which work may be cancelled.
//
// Owned by SchedulerBase. Business lifecycle adapters install their policies.
class DrainManager {
public:
    using CounterFn = std::function<size_t()>;
    using CancelFn  = std::function<void()>;

    // Classification of a counter source, used to aggregate the two values
    // surfaced through SleepStatus (active_request_count /
    // active_cache_transfer_count). Both kinds participate in drained().
    enum class CounterKind {
        REQUEST,         // frontend_active / rpc_onflight / scheduler onflight streams ...
        CACHE_TRANSFER,  // loading_cache / connector inflight / p2p inflight / cache store transfers ...
    };

    DrainManager()          = default;
    virtual ~DrainManager() = default;

    DrainManager(const DrainManager&)            = delete;
    DrainManager& operator=(const DrainManager&) = delete;

    // Register (or replace) a named in-flight counter provider. Null providers
    // are rejected. Safe to call concurrently with drained()/waitDrained().
    void registerCounter(const std::string& name, CounterFn fn, CounterKind kind = CounterKind::REQUEST);

    // Inject cancellation policy. This manager does not decide whether
    // streaming or non-streaming requests are eligible for cancellation.
    void setCancelCallback(CancelFn fn);

    // True iff every registered counter currently reads zero.
    bool drained() const;

    // Wait drain: poll until drained or timeout. timeout_ms <= 0 performs a
    // single immediate check. Returns false on timeout (caller keeps DRAINING).
    bool waitDrained(int64_t timeout_ms);

    // Optionally request cancellation, then await actual cleanup completion.
    bool drain(int64_t timeout_ms, bool cancel = false);

    // Invoke the injected cancel callback (if any). Called outside the internal
    // lock so the callback may freely query this DrainManager.
    void forceCancel();

    // Aggregates for status() reporting.
    int64_t activeRequestCount() const;
    int64_t activeCacheTransferCount() const;

    // Named counters, not a deduplicated number of requests.
    std::string pendingCountersDebugString() const;

    // Wake up waitDrained() pollers early, e.g. when a counter source knows it
    // just dropped to zero. Purely an optimization; polling still converges.
    void notifyDrainProgress();

    // Shrink the poll interval in unit tests to keep them fast.
    void setPollIntervalMs(int64_t interval_ms);

private:
    struct CounterEntry {
        CounterFn   fn;
        CounterKind kind;
    };

    // Snapshot providers under lock, evaluate them outside the lock so a
    // provider may itself take locks without deadlocking registerCounter().
    std::vector<std::pair<std::string, CounterEntry>> snapshotCounters() const;

    int64_t sumByKind(CounterKind kind) const;

    mutable std::mutex                  mutex_;  // guards counters_ / cancel_callback_ / poll_interval_ms_
    std::map<std::string, CounterEntry> counters_;
    CancelFn                            cancel_callback_;
    int64_t                             poll_interval_ms_ = 10;

    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
};

}  // namespace rtp_llm
