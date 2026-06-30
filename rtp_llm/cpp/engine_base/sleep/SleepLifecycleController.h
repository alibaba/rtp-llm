#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// Sleep lifecycle state machine (design doc M1).
//   RUNNING -> DRAINING -> SUSPENDING -> SLEEPING -> WAKING_UP -> RUNNING
// ERROR is a terminal state reachable from SUSPENDING, DRAINING, or WAKING_UP
// on hook failure. The process must be restarted by the control plane.
enum class SleepState {
    RUNNING,
    DRAINING,
    SUSPENDING,
    SLEEPING,
    WAKING_UP,
    ERROR,
};

std::string sleepStateToString(SleepState state);

// Tracks where the KV physical memory currently is. Mirrors M5 backing states.
enum class KvMemoryState {
    ACTIVE,
    PAUSING,
    PAUSED,
    WAKING_UP,
};

std::string kvMemoryStateToString(KvMemoryState state);

// Options passed in via SleepServing RPC (proto SleepRequestPB, design doc M2).
struct SleepOptions {
    // vLLM-compatible level. level=0 is defined as state-preserving sleep
    // (restore weights/device KV/cuda graph on wake_up), but is not implemented
    // in the current MVP. Only level=1 is advertised in supported_levels.
    int32_t                  level      = 1;
    std::string              mode       = "wait";  // "wait" (default) | "abort"; "keep" is unsupported.
    int64_t                  timeout_ms = 0;
    std::string              reason;
    std::vector<std::string> tags;
    bool                     prepare_only = false;  // DRAINING + drained, no GPU release
    bool                     commit_only  = false;  // DRAINING -> SUSPENDING -> SLEEPING
};

// Options passed in via WakeUpServing RPC.
struct WakeUpOptions {
    bool prepare_only = false;  // restore/register resources, keep admission closed
    bool commit_only  = false;  // restart engine and reopen admission after every rank prepared
};

// Snapshot returned by status() / GetSleepStatus RPC (proto SleepStatusResponsePB).
struct SleepStatus {
    bool                     sleep_mode_enabled = false;
    bool                     effective          = false;
    std::vector<int32_t>     supported_levels;
    std::vector<std::string> supported_modes;
    std::string              disabled_reason;
    SleepState               state       = SleepState::RUNNING;
    int64_t                  sleep_epoch = 0;
    std::string              kv_memory_state;
    // True means device KV memory is backed and usable. It does not promise
    // that pre-sleep KV contents survived; those are discarded on sleep.
    bool        device_kv_cache_valid       = true;
    int64_t     active_request_count        = 0;
    int64_t     active_cache_transfer_count = 0;
    std::string gpu_resource_state;
    std::string last_error;
};

// Lightweight result type so the core state machine stays free of grpc/absl deps
// and is independently unit-testable. The RPC layer maps this to grpc::Status.
struct SleepResult {
    enum class Code {
        OK,
        DISABLED,
        UNIMPLEMENTED,
        INVALID_ARGUMENT,
        FAILED_PRECONDITION,
    };

    bool        ok   = true;
    Code        code = Code::OK;
    std::string message;

    static SleepResult success() {
        return SleepResult{true, Code::OK, ""};
    }
    static SleepResult disabled(const std::string& msg) {
        return SleepResult{false, Code::DISABLED, msg};
    }
    static SleepResult unimplemented(const std::string& msg) {
        return SleepResult{false, Code::UNIMPLEMENTED, msg};
    }
    static SleepResult invalidArgument(const std::string& msg) {
        return SleepResult{false, Code::INVALID_ARGUMENT, msg};
    }
    static SleepResult failedPrecondition(const std::string& msg) {
        return SleepResult{false, Code::FAILED_PRECONDITION, msg};
    }
};

// Injection points filled in by downstream modules (M3 drain, M5 KV memory,
// M6/M7 restorable GPU memory, M7 MR/engine quiesce). Hooks left empty are
// treated as no-op success so the core state machine remains unit-testable.
struct SleepHooks {
    // M3 DrainManager: block until drained (or timeout). Return true when drained.
    std::function<bool(const SleepOptions&)> drain;
    // M7: stop scheduler loop at a collective-safe point. No memory/MR release here.
    std::function<bool(const SleepOptions&)> quiesceEngine;
    // M7: after every rank is quiesced, CUDA sync and dereg MR before memory release.
    std::function<bool(const SleepOptions&)> synchronizeAndDeregisterMr;
    // M5: release KV physical pages while keeping VA reserved. KV content is discarded.
    std::function<bool(const SleepOptions&)> releaseKvMemoryBacking;
    // M6/M7: release CPU-backed long-lived allocations, currently weights + cuda_graph tags.
    std::function<bool(const SleepOptions&)> releaseRestorableGpuMemory;

    // M5: re-map KV physical pages at the same VA and reset KV/prefix-cache metadata.
    std::function<bool()> restoreKvMemoryBackingAndResetMetadata;
    // M6/M7: restore CPU-backed long-lived allocations, currently weights + cuda_graph tags.
    std::function<bool()> restoreRestorableGpuMemory;
    // M7: reg MR + refresh rkey/epoch, while the engine loop is still quiesced.
    std::function<bool()> registerMr;
    // M7: restart scheduler loop without resource work.
    std::function<bool()> restartEngine;
    // M7: abort a prepared sleep from DRAINING and resume the engine loop.
    std::function<bool()> cancelQuiesceAndRestartEngine;
    // M7/M8: warmup + health self-check before going back online.
    std::function<bool()> warmupAndHealthCheck;

    // M3: live counters surfaced through status().
    std::function<int64_t()> activeRequestCount;
    std::function<int64_t()> activeCacheTransferCount;
};

// Thread-safe sleep/wake_up lifecycle state machine. Owns the authoritative
// SleepState, sleep_epoch, kv_memory_state, device_kv_cache_valid and
// last_error. State transitions are serialized through transition_mutex_;
// admit() / sleepEpoch() read the atomic state without locking.
class SleepLifecycleController {
public:
    explicit SleepLifecycleController(bool enabled = false): enabled_(enabled) {}
    virtual ~SleepLifecycleController() = default;

    SleepLifecycleController(const SleepLifecycleController&)            = delete;
    SleepLifecycleController& operator=(const SleepLifecycleController&) = delete;

    // Inject downstream module callbacks. Any hook left empty keeps its no-op
    // default behavior. Must be called before sleep()/wakeUp() are triggered.
    void setHooks(const SleepHooks& hooks);

    // Runtime feature gate. Server startup config keeps this disabled by
    // default; tests may enable it explicitly.
    void setEnabled(bool enabled);
    bool enabled() const;

    // Runtime capability gate. enable_sleep_mode may be set while a required
    // implementation detail (for example the torch_memory_saver preload shim)
    // is unavailable; in that case status().effective is false so the control
    // plane can fall back to normal offline.
    void setRuntimeSupport(bool supported, const std::string& disabled_reason = "");
    bool runtimeSupported() const;
    bool effective() const;

    // Trigger sleep: RUNNING -> DRAINING -> SUSPENDING -> SLEEPING. Idempotent when
    // already draining/suspending/sleeping. Illegal from WAKING_UP.
    //
    // prepare_only is used by the instance-level all-rank coordinator: it closes
    // admission and waits for local drain, but deliberately leaves the rank in
    // DRAINING so no rank releases GPU memory until every rank has prepared.
    // commit_only then performs the release from DRAINING.
    SleepResult sleep(const SleepOptions& opt);

    // Trigger wake_up: SLEEPING -> WAKING_UP -> RUNNING. Idempotent when already
    // RUNNING. On failure transitions to ERROR (terminal); the control plane
    // must restart the process.
    SleepResult wakeUp(const WakeUpOptions& opt = WakeUpOptions{});

    // Snapshot for GetSleepStatus.
    SleepStatus status() const;

    // AdmissionGate (M4) hook: true only when fully RUNNING.
    bool admit() const;

    int64_t sleepEpoch() const;

    SleepState state() const;

private:
    // Pure transition legality check against the design doc state diagram.
    static bool isLegalTransition(SleepState from, SleepState to);

    // Atomically move state_ from expected_from to to if the transition is legal.
    // Caller must hold transition_mutex_. Returns false (and sets last_error) on
    // illegal transition.
    bool transitionLocked(SleepState expected_from, SleepState to);

    void        setLastError(const std::string& msg);
    std::string disabledReason() const;

    std::atomic<SleepState> state_{SleepState::RUNNING};
    std::atomic<int64_t>    sleep_epoch_{0};
    std::atomic<bool>       enabled_{false};
    std::atomic<bool>       runtime_supported_{true};
    // Lock ordering: transition_mutex_ -> status_mutex_. Never acquire in reverse.
    std::mutex transition_mutex_;  // serializes sleep/wake_up + idempotency

    std::atomic<KvMemoryState> kv_memory_state_{KvMemoryState::ACTIVE};
    std::atomic<bool>          device_kv_cache_valid_{true};
    std::atomic<bool>          engine_quiesced_{false};

    mutable std::mutex status_mutex_;  // guards last_error_ and runtime_disabled_reason_
    std::string        last_error_;
    std::string        runtime_disabled_reason_;

    SleepHooks hooks_;
};

}  // namespace rtp_llm
