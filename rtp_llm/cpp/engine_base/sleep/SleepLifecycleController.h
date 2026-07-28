#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// Sleep lifecycle state machine.
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

class SleepLifecycleController;

// Move-only proof that an inference request was admitted while the controller
// was RUNNING. Destruction releases exactly one active admission.
class AdmissionLease {
public:
    AdmissionLease() = default;
    ~AdmissionLease();

    AdmissionLease(const AdmissionLease&)            = delete;
    AdmissionLease& operator=(const AdmissionLease&) = delete;

    AdmissionLease(AdmissionLease&& other) noexcept;
    AdmissionLease& operator=(AdmissionLease&& other) noexcept;

    explicit operator bool() const {
        return controller_ != nullptr;
    }

private:
    explicit AdmissionLease(SleepLifecycleController* controller): controller_(controller) {}
    void release();

    SleepLifecycleController* controller_ = nullptr;

    friend class SleepLifecycleController;
};

struct ControllerAdmissionResult {
    AdmissionLease lease;
    SleepState     state       = SleepState::RUNNING;
    int64_t        sleep_epoch = 0;

    bool admitted() const {
        return static_cast<bool>(lease);
    }
};

// Tracks where the KV physical memory currently is.
enum class KvMemoryState {
    ACTIVE,
    PAUSING,
    PAUSED,
    WAKING_UP,
    FAILED,  // a release/restore hook failed mid-transition; KV backing is in an indeterminate state
             // and the controller has gone to ERROR (terminal, needs restart). Set instead of leaving
             // a stale PAUSING/WAKING_UP half-state so status() reflects reality.
};

std::string kvMemoryStateToString(KvMemoryState state);

// Options passed in via SleepServing RPC (proto SleepRequestPB).
struct SleepOptions {
    // vLLM-compatible level. level=0 is defined as state-preserving sleep
    // (restore weights/device KV/cuda graph on wake_up), but is not implemented
    // in the current MVP. The startup-configured level (1, 2, or 3) is the
    // process's only advertised non-zero level.
    int32_t                  level      = 1;
    std::string              mode       = "wait";  // "wait" (default) | "abort"; "keep" is unsupported.
    int64_t                  timeout_ms = 0;
    std::string              reason;
    std::vector<std::string> tags;
    bool prepare_only = false;  // DRAINING + drained; engine running and transfer gate open
    bool commit_only  = false;  // freeze/re-drain/quiesce, then DRAINING -> SUSPENDING -> SLEEPING
};

using DrainCancellationPredicate = std::function<bool()>;

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

// Injection points filled in by downstream modules (drain, KV memory,
// restorable GPU memory, MR/engine quiesce). Hooks left empty are treated as
// no-op success so the core state machine remains unit-testable.
struct SleepHooks {
    // Close the external CacheStore/P2P admission gate after every rank has
    // completed prepare's admitted-work drain. Existing transfers may finish,
    // but no new transfer may race MR deregistration or KV backing release.
    std::function<bool(const SleepOptions&)> freezeExternalTransfers;
    // Arm the engine's collective sleep-quiesce consensus during commit, after
    // the instance-level coordinator has observed prepare success on every rank.
    // At that point no admitted forward remains, so every rank can enter the
    // collective pause without stopping work that drain still depends on.
    // No-op for single-rank. For Level 3, false or an exception is terminal.
    std::function<bool(const SleepOptions&)> armEngineQuiesce;
    // Block until drained (or timeout/cancellation). Long waits must poll the
    // predicate so wake_up or a newer sleep retry can supersede this attempt.
    std::function<bool(const SleepOptions&, const DrainCancellationPredicate&)> drain;
    // Stop scheduler loop at a collective-safe point. No memory/MR release here.
    std::function<bool(const SleepOptions&)> quiesceEngine;
    // Level-3 only: stop RDMA listeners/connections and release transport-owned
    // CUDA/verbs state before any MR or memory backing is removed.
    std::function<bool(const SleepOptions&)> teardownRdmaTransports;
    // After every rank is quiesced and transports are stopped, CUDA sync and
    // deregister KV MRs before memory release.
    std::function<bool(const SleepOptions&)> synchronizeAndDeregisterMr;
    // Release KV physical pages while keeping VA reserved. KV content is discarded.
    std::function<bool(const SleepOptions&)> releaseKvMemoryBacking;
    // Release CPU-backed long-lived allocations, currently weights + cuda_graph tags.
    std::function<bool(const SleepOptions&)> releaseRestorableGpuMemory;

    // Re-map KV physical pages at the same VA and reset KV/prefix-cache metadata.
    std::function<bool()> restoreKvMemoryBackingAndResetMetadata;
    // Restore CPU-backed long-lived allocations, currently weights + cuda_graph tags.
    std::function<bool()> restoreRestorableGpuMemory;
    // Reg MR + refresh rkey/epoch, while the engine loop is still quiesced.
    std::function<bool()> registerMr;
    // Restart scheduler loop without resource work.
    std::function<bool()> restartEngine;
    // Abort a sleep that reached engine quiesce while still in DRAINING.
    // A prepare-only rollback does not call this because prepare leaves the
    // engine running.
    std::function<bool()> cancelQuiesceAndRestartEngine;
    // Warmup + health self-check before going back online.
    std::function<bool()> warmupAndHealthCheck;
    // Reopen external transfer admission only after every restored resource and
    // health check is ready. Also used to roll back a failed commit after the
    // engine has restarted. A failed/throwing call must leave the gate closed.
    std::function<bool()> resumeExternalTransfers;

    // Level-3 only: tear down CUDA-backed cross-process resources before the
    // process is checkpointed, then rebuild them after restore. Graphs that
    // embed collective kernels are recaptured in wake commit after every rank
    // has completed wake prepare.
    // The coordinator returns the all-rank AND for (phase, epoch). Controllers
    // call every ready gate even when preceding local work failed; CUDA
    // collective work runs only after a successful global ready gate. A done
    // gate then propagates each rank's local result to every peer.
    std::function<bool(const std::string&, int64_t, bool)> coordinateResourcePhase;
    std::function<bool(const SleepOptions&)>               teardownCollectives;
    std::function<bool()>                                  rebuildCollectives;
    // Recreate transport-owned mempools/listeners/QPs after CUDA restore and
    // memory restoration, but before KV MRs are registered and advertised.
    std::function<bool()> rebuildRdmaTransports;
    std::function<bool()> recaptureCollectiveGraphs;

    // Live counters surfaced through status().
    std::function<int64_t()> activeRequestCount;
    std::function<int64_t()> activeCacheTransferCount;
};

// Thread-safe sleep/wake_up lifecycle state machine. Owns the authoritative
// SleepState, sleep_epoch, kv_memory_state, device_kv_cache_valid and
// last_error. State transitions are serialized through transition_mutex_.
// Admission acquisition and RUNNING-boundary transitions are linearized by
// admission_mutex_; long-running lifecycle hooks never hold that mutex.
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

    // Startup-selected sleep level for this process (see RuntimeConfig
    // sleep_mode_level). Must be called before sleep()/wakeUp(). Levels 2 and 3
    // discard weights: the weights VMM region was opened without host backup,
    // so wake reloads from the original checkpoint. Level 3 additionally tears
    // down CUDA-backed cross-process resources before an external CUDA process
    // checkpoint. The value is fixed at model-load time and must be 1, 2 or 3.
    void    setConfiguredLevel(int32_t level);
    int32_t configuredLevel() const;
    bool    discardWeights() const;
    // Level captured on the RUNNING->DRAINING transition of the active sleep,
    // read by the wake_up restore hook to decide whether to reload weights.
    int32_t activeSleepLevel() const;

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
    // admission and waits for local drain, but deliberately leaves both the
    // engine and external transfer gate running/open in DRAINING. Once every
    // rank has prepared, commit_only freezes transfers, re-drains the gate
    // boundary, quiesces the engine, and performs the release.
    SleepResult sleep(const SleepOptions& opt);

    // Trigger wake_up: SLEEPING -> WAKING_UP -> RUNNING. Idempotent when already
    // RUNNING. On failure transitions to ERROR (terminal); the control plane
    // must restart the process.
    SleepResult wakeUp(const WakeUpOptions& opt = WakeUpOptions{});

    // Snapshot for GetSleepStatus.
    SleepStatus status() const;

    // AdmissionGate hook: true only when fully RUNNING.
    bool admit() const;

    // Atomically check RUNNING and, if admitted, increment the controller-owned
    // active admission tracker. The returned lease releases the tracker once.
    ControllerAdmissionResult acquireAdmission();
    int64_t                   activeAdmissionCount() const;

    int64_t sleepEpoch() const;

    SleepState state() const;

private:
    // Pure transition legality check against the state diagram.
    static bool isLegalTransition(SleepState from, SleepState to);

    // Atomically move state_ from expected_from to to if the transition is legal.
    // Caller must hold transition_mutex_. Returns false (and sets last_error) on
    // illegal transition.
    bool transitionLocked(SleepState expected_from, SleepState to);

    // Caller holds transition_mutex_. Invalidates the previous token, then
    // joins its hook through drain_cv_ before returning the new generation.
    uint64_t supersedeDrainAndWaitLocked();
    // Called by the hook owner before it tries to reacquire transition_mutex_.
    void finishDrainAttempt();

    void releaseAdmission();
    void setLastError(const std::string& msg);
    // Read last_error_ under status_mutex_ only. Error paths use this instead of
    // status().last_error so they do not fire the activeRequestCount /
    // activeCacheTransferCount engine hooks as a side effect (those reach into
    // engine internals and could throw while transition_mutex_ is held).
    std::string lastError() const;
    std::string disabledReason() const;

    std::atomic<SleepState> state_{SleepState::RUNNING};
    std::atomic<int64_t>    sleep_epoch_{0};
    std::atomic<bool>       enabled_{false};
    std::atomic<bool>       runtime_supported_{true};
    // Startup-fixed sleep level for this process (1 = host backup, 2 = discard
    // weights, 3 = discard weights plus CUDA process checkpoint).
    std::atomic<int32_t> configured_level_{1};
    // Level of the in-progress/last sleep, captured at RUNNING->DRAINING.
    std::atomic<int32_t> active_sleep_level_{0};
    // Monotonic token for the drain attempt allowed to continue into quiesce.
    // wake_up(DRAINING) and every newer sleep phase/retry invalidate the previous
    // token, then join its promptly-cancelled drain hook.
    std::atomic<uint64_t> drain_generation_{0};
    // Lock ordering: transition_mutex_ -> drain_mutex_ / admission_mutex_ ->
    // hooks_mutex_ -> status_mutex_. A drain hook clears drain_active_ without
    // transition_mutex_, releases drain_mutex_, then reacquires transition_mutex_.
    // Never acquire in reverse.
    std::mutex              transition_mutex_;  // serializes sleep/wake_up + idempotency
    std::mutex              drain_mutex_;
    std::condition_variable drain_cv_;
    bool                    drain_active_ = false;
    mutable std::mutex      admission_mutex_;
    int64_t                 active_admissions_ = 0;

    std::atomic<KvMemoryState> kv_memory_state_{KvMemoryState::ACTIVE};
    std::atomic<bool>          device_kv_cache_valid_{true};
    // Phase markers let prepare rollback avoid touching an engine and transfer
    // gate that prepare deliberately left running/open. They also make a
    // commit-only request fail closed unless admitted work was actually drained.
    std::atomic<bool>          admitted_work_drained_{false};
    std::atomic<bool>          external_transfers_frozen_{false};
    std::atomic<bool>          engine_quiesce_armed_{false};
    std::atomic<bool>          engine_quiesced_{false};

    mutable std::mutex status_mutex_;  // guards last_error_ and runtime_disabled_reason_
    std::string        last_error_;
    std::string        runtime_disabled_reason_;

    // Guards hooks_. Transition paths read hooks_ under transition_mutex_ (which
    // is mutually exclusive with setHooks), so they do not take this lock; it
    // exists so status() can read the live-counter hooks off the transition path
    // without racing a concurrent setHooks (std::function assignment is not
    // atomic).
    mutable std::mutex hooks_mutex_;
    SleepHooks         hooks_;

    friend class AdmissionLease;
};

}  // namespace rtp_llm
