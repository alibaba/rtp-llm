#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace rtp_llm {

// Freeze lifecycle state machine (design doc M1).
//   RUNNING -> DRAINING -> FREEZING -> FROZEN -> RESUMING -> RUNNING (+ ERROR)
enum class FreezeState {
    RUNNING,
    DRAINING,
    FREEZING,
    FROZEN,
    RESUMING,
    ERROR,
};

std::string freezeStateToString(FreezeState state);

// Tracks where the KV physical memory currently is. Mirrors M5 backing states.
enum class KvMemoryState {
    ACTIVE,
    PAUSING,
    PAUSED,
    RESUMING,
};

std::string kvMemoryStateToString(KvMemoryState state);

// Options passed in via FreezeServing RPC (proto FreezeRequestPB, design doc M2).
struct FreezeOptions {
    std::string mode;  // "graceful" (default) | "force"
    int64_t     drain_timeout_ms = 0;
    bool        force            = false;
    std::string reason;
    bool        prepare_only = false;  // DRAINING + drained, no GPU release
    bool        commit_only  = false;  // DRAINING -> FREEZING -> FROZEN
};

// Snapshot returned by status() / GetFreezeStatus RPC (proto FreezeStatusResponsePB).
struct FreezeStatus {
    FreezeState state        = FreezeState::RUNNING;
    int64_t     freeze_epoch = 0;
    std::string kv_memory_state;
    bool        device_kv_cache_valid       = true;
    int64_t     active_request_count        = 0;
    int64_t     active_cache_transfer_count = 0;
    std::string gpu_resource_state;
    std::string last_error;
};

// Lightweight result type so the core state machine stays free of grpc/absl deps
// and is independently unit-testable. The RPC layer maps this to grpc::Status.
struct FreezeResult {
    bool        ok = true;
    std::string message;

    static FreezeResult success() {
        return FreezeResult{true, ""};
    }
    static FreezeResult error(const std::string& msg) {
        return FreezeResult{false, msg};
    }
};

// Injection points filled in by downstream modules (M3 drain, M5 KV saver,
// M6 weights saver, M7 MR/engine quiesce). All default to no-op success so the
// T1 skeleton drives the full state machine without those modules present.
//
// TODO(M3/M5/M6/M7): wire real implementations via setHooks().
struct FreezeHooks {
    // M3 DrainManager: block until drained (or timeout). Return true when drained.
    std::function<bool(const FreezeOptions&)> drain;
    // M5 KVCachePhysicalMemoryController::pause_physical_memory().
    std::function<bool(const FreezeOptions&)> pauseKvMemory;
    // M6 WeightMemorySaver: tms.pause("weights").
    std::function<bool(const FreezeOptions&)> pauseWeights;
    // M7: stop scheduler loop (no destruct), CUDA sync, dereg MR.
    std::function<bool(const FreezeOptions&)> deregMrAndQuiesceEngine;

    // M7: reg MR + refresh rkey/epoch + resume scheduler loop.
    std::function<bool()> regMrAndResumeEngine;
    // M5: KVCachePhysicalMemoryController::resume_physical_memory().
    std::function<bool()> resumeKvMemory;
    // M6: tms.resume("weights").
    std::function<bool()> resumeWeights;
    // M5: BlockPool::resetMetadata() + BlockCache::clear() + prefix generation++.
    std::function<bool()> resetKvMetadata;
    // M7/M8: warmup + health self-check before going back online.
    std::function<bool()> warmupAndHealthCheck;

    // M3: live counters surfaced through status().
    std::function<int64_t()> activeRequestCount;
    std::function<int64_t()> activeCacheTransferCount;
};

// Thread-safe freeze/resume lifecycle state machine. Owns the authoritative
// FreezeState, freeze_epoch, kv_memory_state, device_kv_cache_valid and
// last_error. State transitions are serialized through transition_mutex_;
// admit() / freezeEpoch() read the atomic state without locking.
class FreezeLifecycleController {
public:
    FreezeLifecycleController()          = default;
    virtual ~FreezeLifecycleController() = default;

    FreezeLifecycleController(const FreezeLifecycleController&)            = delete;
    FreezeLifecycleController& operator=(const FreezeLifecycleController&) = delete;

    // Inject downstream module callbacks. Any hook left empty keeps its no-op
    // default behavior. Must be called before freeze()/resume() are triggered.
    void setHooks(const FreezeHooks& hooks);

    // Trigger freeze: RUNNING -> DRAINING -> FREEZING -> FROZEN. Idempotent when
    // already draining/freezing/frozen. Illegal from RESUMING.
    //
    // prepare_only is used by the instance-level all-rank coordinator: it closes
    // admission and waits for local drain, but deliberately leaves the rank in
    // DRAINING so no rank releases GPU memory until every rank has prepared.
    // commit_only then performs the release from DRAINING.
    FreezeResult freeze(const FreezeOptions& opt);

    // Trigger resume: FROZEN -> RESUMING -> RUNNING. Idempotent when already
    // RUNNING. On failure returns ERROR with admission closed; recovery is an
    // explicit retry or operator action. Also allowed from ERROR as a recovery
    // attempt.
    FreezeResult resume();

    // Snapshot for GetFreezeStatus.
    FreezeStatus status() const;

    // AdmissionGate (M4) hook: true only when fully RUNNING.
    bool admit() const;

    int64_t freezeEpoch() const;

    FreezeState state() const;

private:
    // Pure transition legality check against the design doc state diagram.
    static bool isLegalTransition(FreezeState from, FreezeState to);

    // Atomically move state_ from expected_from to to if the transition is legal.
    // Caller must hold transition_mutex_. Returns false (and sets last_error) on
    // illegal transition.
    bool transitionLocked(FreezeState expected_from, FreezeState to);

    void setLastError(const std::string& msg);

    std::atomic<FreezeState> state_{FreezeState::RUNNING};
    std::atomic<int64_t>     freeze_epoch_{0};
    std::mutex               transition_mutex_;  // serializes freeze/resume + idempotency

    std::atomic<KvMemoryState> kv_memory_state_{KvMemoryState::ACTIVE};
    std::atomic<bool>          device_kv_cache_valid_{true};

    mutable std::mutex status_mutex_;  // guards non-atomic status fields
    std::string        last_error_;

    FreezeHooks hooks_;
};

}  // namespace rtp_llm
