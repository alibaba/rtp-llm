#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

namespace rtp_llm {

// M3 DrainManager (design doc §2 M3): aggregates in-flight counters from every
// layer (frontend, rpc, scheduler, cache loading, connector, p2p, cache store)
// and decides whether the engine is fully drained. Counter sources are injected
// as named providers instead of hard dependencies on concrete classes, so unit
// tests mock them and integration wires the real getters
// (e.g. KVCacheConnectorCoordinator::inflightTransferCount,
// NormalCacheStore::activeTransferCount).
//
// Drain policies:
//   - wait: waitDrained(timeout_ms) polls until all counters reach zero or
//     the timeout expires. On timeout it returns false and the caller (M1)
//     stays in DRAINING without releasing GPU resources.
//   - abort: an injected cancel callback is invoked first (the callback owner
//     is responsible for cancelling non-streaming requests and exempting
//     streaming ones), then drain is awaited as usual.
//
// Acts as the SleepHooks drain provider for SleepLifecycleController (M1)
// via installHooks(). Thread-safe.
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

    // Remove a counter provider; no-op when the name is unknown.
    void unregisterCounter(const std::string& name);

    // Inject the abort callback. The provider must cancel non-streaming
    // requests only; streaming exemption is its responsibility. DrainManager
    // just invokes it and keeps waiting for the counters to reach zero.
    void setCancelCallback(CancelFn fn);

    // True iff every registered counter currently reads zero.
    bool drained() const;

    // Wait drain: poll until drained or timeout. timeout_ms <= 0 performs a
    // single immediate check. Returns false on timeout (caller keeps DRAINING).
    bool waitDrained(int64_t timeout_ms);

    // SleepHooks::drain entry: applies abort policy (cancel callback) when
    // requested, then waits for drain up to opt.timeout_ms.
    bool drain(const SleepOptions& opt);

    // Invoke the injected cancel callback (if any). Called outside the internal
    // lock so the callback may freely query this DrainManager.
    void forceCancel();

    // Aggregates for M1 status() reporting.
    int64_t activeRequestCount() const;
    int64_t activeCacheTransferCount() const;

    // Wire this manager into M1's SleepHooks (drain + the two count hooks).
    // The DrainManager must outlive the controller that holds the hooks.
    void installHooks(SleepHooks& hooks);

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

    // Human-readable "name=value" list of non-zero counters (for logging).
    std::string pendingCountersDebugString() const;

    mutable std::mutex                  mutex_;  // guards counters_ / cancel_callback_ / poll_interval_ms_
    std::map<std::string, CounterEntry> counters_;
    CancelFn                            cancel_callback_;
    int64_t                             poll_interval_ms_ = 10;

    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
};

}  // namespace rtp_llm
