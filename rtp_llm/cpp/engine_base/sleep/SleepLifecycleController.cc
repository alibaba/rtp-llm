#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include <exception>
#include <utility>

namespace rtp_llm {

namespace {

template<typename Hook, typename... Args>
bool invokeHookNoThrow(const char* name, Hook& hook, Args&&... args) {
    try {
        return hook(std::forward<Args>(args)...);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("sleep lifecycle hook %s threw exception: %s", name, e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("sleep lifecycle hook %s threw unknown exception", name);
    }
    return false;
}

}  // namespace

AdmissionLease::~AdmissionLease() {
    release();
}

AdmissionLease::AdmissionLease(AdmissionLease&& other) noexcept: controller_(other.controller_) {
    other.controller_ = nullptr;
}

AdmissionLease& AdmissionLease::operator=(AdmissionLease&& other) noexcept {
    if (this != &other) {
        release();
        controller_       = other.controller_;
        other.controller_ = nullptr;
    }
    return *this;
}

void AdmissionLease::release() {
    if (controller_ == nullptr) {
        return;
    }
    auto* controller = controller_;
    controller_      = nullptr;
    controller->releaseAdmission();
}

std::string sleepStateToString(SleepState state) {
    switch (state) {
        case SleepState::RUNNING:
            return "RUNNING";
        case SleepState::DRAINING:
            return "DRAINING";
        case SleepState::SUSPENDING:
            return "SUSPENDING";
        case SleepState::SLEEPING:
            return "SLEEPING";
        case SleepState::WAKING_UP:
            return "WAKING_UP";
        case SleepState::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

std::string kvMemoryStateToString(KvMemoryState state) {
    switch (state) {
        case KvMemoryState::ACTIVE:
            return "ACTIVE";
        case KvMemoryState::PAUSING:
            return "PAUSING";
        case KvMemoryState::PAUSED:
            return "PAUSED";
        case KvMemoryState::WAKING_UP:
            return "WAKING_UP";
        default:
            return "UNKNOWN";
    }
}

void SleepLifecycleController::setHooks(const SleepHooks& hooks) {
    // transition_mutex_ excludes concurrent transitions (which read hooks_);
    // hooks_mutex_ additionally excludes status()'s off-transition counter reads.
    // Order: transition_mutex_ -> hooks_mutex_.
    std::lock_guard<std::mutex> transition_lock(transition_mutex_);
    std::lock_guard<std::mutex> hooks_lock(hooks_mutex_);
    hooks_ = hooks;
}

bool SleepLifecycleController::isLegalTransition(SleepState from, SleepState to) {
    switch (from) {
        case SleepState::RUNNING:
            return to == SleepState::DRAINING || to == SleepState::ERROR;
        case SleepState::DRAINING:
            // sleep cancelled before release -> RUNNING; drained -> SUSPENDING.
            return to == SleepState::SUSPENDING || to == SleepState::RUNNING || to == SleepState::ERROR;
        case SleepState::SUSPENDING:
            return to == SleepState::SLEEPING || to == SleepState::ERROR;
        case SleepState::SLEEPING:
            return to == SleepState::WAKING_UP;
        case SleepState::WAKING_UP:
            // rebuild ok -> RUNNING; rebuild failed -> ERROR for explicit
            // recovery. Do not run implicit resource rollback here.
            return to == SleepState::RUNNING || to == SleepState::ERROR;
        case SleepState::ERROR:
            // Terminal state. Process must be restarted by the control plane.
            return false;
        default:
            return false;
    }
}

bool SleepLifecycleController::transitionLocked(SleepState expected_from, SleepState to) {
    std::lock_guard<std::mutex> admission_lock(admission_mutex_);
    const SleepState            current = state_.load(std::memory_order_acquire);
    if (current != expected_from || !isLegalTransition(expected_from, to)) {
        setLastError("illegal transition: " + sleepStateToString(current) + " -> " + sleepStateToString(to));
        return false;
    }
    if (expected_from == SleepState::RUNNING && to == SleepState::DRAINING) {
        sleep_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }
    state_.store(to, std::memory_order_release);
    RTP_LLM_LOG_INFO("sleep state transition: %s -> %s (epoch=%ld)",
                     sleepStateToString(expected_from).c_str(),
                     sleepStateToString(to).c_str(),
                     sleep_epoch_.load());
    return true;
}

void SleepLifecycleController::setLastError(const std::string& msg) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    last_error_ = msg;
    if (!msg.empty()) {
        RTP_LLM_LOG_WARNING("sleep lifecycle: %s", msg.c_str());
    }
}

void SleepLifecycleController::setEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_release);
}

bool SleepLifecycleController::enabled() const {
    return enabled_.load(std::memory_order_acquire);
}

void SleepLifecycleController::setConfiguredLevel(int32_t level) {
    // torch_memory_saver fixes the weights backup mode at model-load time: level
    // 2 discards weights (region opened without host cpu_backup); any other value
    // keeps host backup and is treated as level 1.
    configured_level_.store(level == 2 ? 2 : 1, std::memory_order_release);
}

int32_t SleepLifecycleController::configuredLevel() const {
    return configured_level_.load(std::memory_order_acquire);
}

bool SleepLifecycleController::discardWeights() const {
    return configuredLevel() == 2;
}

int32_t SleepLifecycleController::activeSleepLevel() const {
    return active_sleep_level_.load(std::memory_order_acquire);
}

void SleepLifecycleController::setRuntimeSupport(bool supported, const std::string& disabled_reason) {
    runtime_supported_.store(supported, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        runtime_disabled_reason_ = supported ? "" : disabled_reason;
    }
    if (!supported) {
        RTP_LLM_LOG_WARNING("sleep mode runtime support unavailable: %s", disabled_reason.c_str());
    }
}

bool SleepLifecycleController::runtimeSupported() const {
    return runtime_supported_.load(std::memory_order_acquire);
}

bool SleepLifecycleController::effective() const {
    return enabled() && runtimeSupported();
}

std::string SleepLifecycleController::disabledReason() const {
    if (!enabled()) {
        return "sleep mode is disabled";
    }
    if (!runtimeSupported()) {
        std::lock_guard<std::mutex> lock(status_mutex_);
        return runtime_disabled_reason_.empty() ? "sleep mode runtime support is unavailable" :
                                                  runtime_disabled_reason_;
    }
    return "";
}

SleepResult SleepLifecycleController::sleep(const SleepOptions& opt) {
    std::lock_guard<std::mutex> lock(transition_mutex_);

    if (!effective()) {
        return SleepResult::disabled(disabledReason());
    }
    if (opt.prepare_only && opt.commit_only) {
        return SleepResult::invalidArgument("sleep rejected: prepare_only and commit_only cannot both be true");
    }
    // torch_memory_saver binds the weights region's cpu_backup at model-load
    // time, so this process supports exactly one non-zero level, selected at
    // startup: 2 (discard weights) when sleep_mode_level=2, otherwise 1 (host
    // backup). A request must match it.
    const int32_t configured_level = configuredLevel();
    if (opt.level == 0) {
        return SleepResult::unimplemented(
            "sleep rejected: level=0 state-preserving sleep is defined but not implemented; supported_levels=["
            + std::to_string(configured_level) + "]");
    }
    if (opt.level != configured_level) {
        return SleepResult::invalidArgument(
            "sleep rejected: level=" + std::to_string(opt.level)
            + " does not match this process's startup sleep_mode_level=" + std::to_string(configured_level)
            + " (torch_memory_saver fixes the weights backup mode at load time); supported_levels=["
            + std::to_string(configured_level) + "]");
    }
    if (opt.mode != "wait" && opt.mode != "abort") {
        return SleepResult::invalidArgument("sleep rejected: mode must be \"wait\" or \"abort\"");
    }
    if (opt.timeout_ms < 0) {
        return SleepResult::invalidArgument("sleep rejected: timeout_ms must be non-negative");
    }

    const SleepState current = state_.load(std::memory_order_acquire);
    // Idempotency: already sleeping or inside the release section. DRAINING is
    // intentionally retriable so a timeout can later progress, or be escalated
    // with mode=abort.
    if (current == SleepState::SUSPENDING || current == SleepState::SLEEPING) {
        return SleepResult::success();
    }
    if (opt.commit_only && current != SleepState::DRAINING) {
        return SleepResult::failedPrecondition("sleep commit rejected in state " + sleepStateToString(current));
    }
    if (current != SleepState::RUNNING && current != SleepState::DRAINING) {
        return SleepResult::failedPrecondition("sleep rejected in state " + sleepStateToString(current));
    }

    setLastError("");

    if (current == SleepState::RUNNING) {
        engine_quiesced_.store(false, std::memory_order_release);
        // Record the level of this sleep so the wake_up restore hook knows
        // whether to reload discarded weights (level 2) or not (level 1).
        active_sleep_level_.store(opt.level, std::memory_order_release);
        if (!transitionLocked(SleepState::RUNNING, SleepState::DRAINING)) {
            return SleepResult::failedPrecondition(status().last_error);
        }
    }

    // --- DRAINING: wait for in-flight requests and cache transfers. ---
    if (!opt.commit_only && hooks_.drain) {
        // Route through invokeHookNoThrow like every other hook: a throwing
        // drain must not escape the transition while transition_mutex_ is held
        // (it would leave the controller wedged in DRAINING with a poisoned
        // mutex). An exception is treated as "not drained".
        if (!invokeHookNoThrow("drain", hooks_.drain, opt)) {
            // Per design: graceful drain timeout keeps DRAINING and does NOT
            // release GPU. The controller stays in DRAINING; control plane can
            // retry sleep (idempotent) or escalate with mode=abort.
            setLastError("drain not finished (timeout or aborted), staying in DRAINING");
            return SleepResult::failedPrecondition("drain not finished, state=DRAINING");
        }
    }

    if (!opt.commit_only && !engine_quiesced_.load(std::memory_order_acquire)) {
        if (hooks_.quiesceEngine) {
            if (!invokeHookNoThrow("quiesceEngine", hooks_.quiesceEngine, opt)) {
                setLastError("quiesceEngine failed, staying in DRAINING");
                return SleepResult::failedPrecondition(status().last_error);
            }
        }
        engine_quiesced_.store(true, std::memory_order_release);
    }

    if (opt.prepare_only) {
        RTP_LLM_LOG_INFO("sleep prepare completed, staying in DRAINING (epoch=%ld)", sleep_epoch_.load());
        return SleepResult::success();
    }

    if (!engine_quiesced_.load(std::memory_order_acquire)) {
        setLastError("sleep commit rejected: engine is not quiesced");
        return SleepResult::failedPrecondition(status().last_error);
    }

    if (!transitionLocked(SleepState::DRAINING, SleepState::SUSPENDING)) {
        return SleepResult::failedPrecondition(status().last_error);
    }

    // --- SUSPENDING: dereg MR, release memory backing. ---
    // Ordering: engine already quiesced in prepare; CUDA sync + dereg MR happen
    // before pausing KV physical memory; CPU-backed persistent allocations are
    // released last.
    bool ok = true;
    if (ok && hooks_.synchronizeAndDeregisterMr) {
        ok = invokeHookNoThrow("synchronizeAndDeregisterMr", hooks_.synchronizeAndDeregisterMr, opt);
        if (!ok) {
            setLastError("synchronizeAndDeregisterMr failed");
        }
    }
    if (ok && hooks_.releaseKvMemoryBacking) {
        kv_memory_state_.store(KvMemoryState::PAUSING, std::memory_order_release);
        ok = invokeHookNoThrow("releaseKvMemoryBacking", hooks_.releaseKvMemoryBacking, opt);
        if (ok) {
            kv_memory_state_.store(KvMemoryState::PAUSED, std::memory_order_release);
            device_kv_cache_valid_.store(false, std::memory_order_release);
        } else {
            setLastError("releaseKvMemoryBacking failed");
        }
    }
    if (ok && hooks_.releaseRestorableGpuMemory) {
        ok = invokeHookNoThrow("releaseRestorableGpuMemory", hooks_.releaseRestorableGpuMemory, opt);
        if (!ok) {
            setLastError("releaseRestorableGpuMemory failed");
        }
    }

    if (!ok) {
        transitionLocked(SleepState::SUSPENDING, SleepState::ERROR);
        return SleepResult::failedPrecondition(status().last_error);
    }

    if (!transitionLocked(SleepState::SUSPENDING, SleepState::SLEEPING)) {
        return SleepResult::failedPrecondition(status().last_error);
    }
    return SleepResult::success();
}

SleepResult SleepLifecycleController::wakeUp(const WakeUpOptions& opt) {
    std::lock_guard<std::mutex> lock(transition_mutex_);

    if (!effective()) {
        return SleepResult::disabled(disabledReason());
    }
    if (opt.prepare_only && opt.commit_only) {
        return SleepResult::invalidArgument("wake_up rejected: prepare_only and commit_only cannot both be true");
    }

    const SleepState current = state_.load(std::memory_order_acquire);
    // Idempotency: already running.
    if (current == SleepState::RUNNING) {
        return SleepResult::success();
    }
    // Instance-level coordinator uses wake_up as the abort path for a prepared
    // sleep that never committed. No GPU resource was released in DRAINING.
    if (current == SleepState::DRAINING) {
        setLastError("");
        if (hooks_.cancelQuiesceAndRestartEngine) {
            if (!invokeHookNoThrow("cancelQuiesceAndRestartEngine", hooks_.cancelQuiesceAndRestartEngine)) {
                setLastError("cancelQuiesceAndRestartEngine failed");
                transitionLocked(SleepState::DRAINING, SleepState::ERROR);
                return SleepResult::failedPrecondition(status().last_error);
            }
        } else if (hooks_.restartEngine) {
            if (!invokeHookNoThrow("restartEngine", hooks_.restartEngine)) {
                setLastError("restartEngine failed");
                transitionLocked(SleepState::DRAINING, SleepState::ERROR);
                return SleepResult::failedPrecondition(status().last_error);
            }
        }
        engine_quiesced_.store(false, std::memory_order_release);
        if (!transitionLocked(SleepState::DRAINING, SleepState::RUNNING)) {
            return SleepResult::failedPrecondition(status().last_error);
        }
        return SleepResult::success();
    }
    if (opt.commit_only && current != SleepState::WAKING_UP && current != SleepState::RUNNING) {
        return SleepResult::failedPrecondition("wake_up commit rejected in state " + sleepStateToString(current));
    }
    if (opt.prepare_only && current == SleepState::WAKING_UP) {
        return SleepResult::success();
    }
    if (current != SleepState::SLEEPING && current != SleepState::WAKING_UP) {
        return SleepResult::failedPrecondition("wake_up rejected in state " + sleepStateToString(current));
    }

    setLastError("");
    if (current != SleepState::WAKING_UP && !transitionLocked(current, SleepState::WAKING_UP)) {
        return SleepResult::failedPrecondition(status().last_error);
    }

    // --- WAKING_UP: restore memory backing, reset metadata, reg MR, warmup. ---
    bool ok = true;
    // Restore weights (level-2 streams them back in place from the model loader)
    // BEFORE re-backing the KV cache. The KV cache is sized to consume nearly all
    // GPU memory left free after weights at cold start, so remapping the KV
    // physical pages first leaves no headroom for the loader's transient buffers
    // (raw checkpoint reads, dequant / TP-split / MoE-fusion intermediates) during
    // the level-2 reload -> OOM. Weights-then-KV mirrors the cold-start order
    // (weights load, then KV is sized from what remains). The two hooks are
    // independent: the reload only copies into the weight tensors and cuda_graph
    // resume only remaps graph-private pages; neither touches KV content.
    if (!opt.commit_only && ok && hooks_.restoreRestorableGpuMemory) {
        ok = invokeHookNoThrow("restoreRestorableGpuMemory", hooks_.restoreRestorableGpuMemory);
        if (!ok) {
            setLastError("restoreRestorableGpuMemory failed");
        }
    }
    if (!opt.commit_only && ok && hooks_.restoreKvMemoryBackingAndResetMetadata) {
        kv_memory_state_.store(KvMemoryState::WAKING_UP, std::memory_order_release);
        ok = invokeHookNoThrow("restoreKvMemoryBackingAndResetMetadata", hooks_.restoreKvMemoryBackingAndResetMetadata);
        if (!ok) {
            setLastError("restoreKvMemoryBackingAndResetMetadata failed");
        }
    }
    if (!opt.commit_only && ok) {
        kv_memory_state_.store(KvMemoryState::ACTIVE, std::memory_order_release);
    }
    if (!opt.commit_only && ok && hooks_.registerMr) {
        ok = invokeHookNoThrow("registerMr", hooks_.registerMr);
        if (!ok) {
            setLastError("registerMr failed");
        }
    }

    if (!ok) {
        // Admission remains closed in ERROR. Control plane only observes wake_up
        // failure; recovery is an explicit retry or operator action.
        transitionLocked(SleepState::WAKING_UP, SleepState::ERROR);
        return SleepResult::failedPrecondition(status().last_error);
    }

    if (opt.prepare_only) {
        RTP_LLM_LOG_INFO("wake_up prepare completed, staying in WAKING_UP (epoch=%ld)", sleep_epoch_.load());
        return SleepResult::success();
    }

    if (ok && hooks_.restartEngine) {
        ok = invokeHookNoThrow("restartEngine", hooks_.restartEngine);
        if (!ok) {
            setLastError("restartEngine failed");
        }
    }
    if (ok && hooks_.warmupAndHealthCheck) {
        ok = invokeHookNoThrow("warmupAndHealthCheck", hooks_.warmupAndHealthCheck);
        if (!ok) {
            setLastError("warmupAndHealthCheck failed");
        }
    }

    if (!ok) {
        // Admission remains closed in ERROR. Control plane only observes wake_up
        // failure; recovery is an explicit retry or operator action.
        transitionLocked(SleepState::WAKING_UP, SleepState::ERROR);
        return SleepResult::failedPrecondition(status().last_error);
    }

    device_kv_cache_valid_.store(true, std::memory_order_release);
    engine_quiesced_.store(false, std::memory_order_release);
    if (!transitionLocked(SleepState::WAKING_UP, SleepState::RUNNING)) {
        return SleepResult::failedPrecondition(status().last_error);
    }
    return SleepResult::success();
}

SleepStatus SleepLifecycleController::status() const {
    SleepStatus s;
    s.sleep_mode_enabled = enabled();
    s.effective          = effective();
    // This process supports exactly one non-zero level, fixed at startup by
    // sleep_mode_level (2 = discard weights, else 1); see sleep() gate.
    const int32_t configured_level = configuredLevel();
    s.supported_levels             = s.effective ? std::vector<int32_t>{configured_level} : std::vector<int32_t>{};
    s.supported_modes       = s.effective ? std::vector<std::string>{"wait", "abort"} : std::vector<std::string>{};
    s.disabled_reason       = s.effective ? "" : disabledReason();
    s.state                 = state_.load(std::memory_order_acquire);
    s.sleep_epoch           = sleep_epoch_.load(std::memory_order_acquire);
    s.kv_memory_state       = kvMemoryStateToString(kv_memory_state_.load(std::memory_order_acquire));
    s.device_kv_cache_valid = device_kv_cache_valid_.load(std::memory_order_acquire);
    // Copy the live-counter hooks under hooks_mutex_, then invoke the copies with
    // the lock released (the hooks reach into engine counters and must not run
    // under a controller mutex). Off the transition path, so we must not touch
    // transition_mutex_ here (sleep()/wakeUp() call status() while holding it).
    std::function<int64_t()> active_request_count_fn;
    std::function<int64_t()> active_cache_transfer_count_fn;
    {
        std::lock_guard<std::mutex> hooks_lock(hooks_mutex_);
        active_request_count_fn        = hooks_.activeRequestCount;
        active_cache_transfer_count_fn = hooks_.activeCacheTransferCount;
    }
    if (active_request_count_fn) {
        s.active_request_count = active_request_count_fn();
    }
    if (active_cache_transfer_count_fn) {
        s.active_cache_transfer_count = active_cache_transfer_count_fn();
    }
    if (s.state == SleepState::SLEEPING) {
        s.gpu_resource_state = "RELEASED";
    } else if (s.state == SleepState::SUSPENDING) {
        s.gpu_resource_state = "RELEASING";
    } else if (s.state == SleepState::WAKING_UP) {
        s.gpu_resource_state = "RESTORING";
    } else if (s.state == SleepState::ERROR) {
        s.gpu_resource_state = "UNKNOWN";
    } else {
        s.gpu_resource_state = "ACTIVE";
    }
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        s.last_error = last_error_;
    }
    return s;
}

bool SleepLifecycleController::admit() const {
    return state_.load(std::memory_order_acquire) == SleepState::RUNNING;
}

ControllerAdmissionResult SleepLifecycleController::acquireAdmission() {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    ControllerAdmissionResult   result;
    result.state       = state_.load(std::memory_order_acquire);
    result.sleep_epoch = sleep_epoch_.load(std::memory_order_acquire);
    if (result.state == SleepState::RUNNING) {
        ++active_admissions_;
        result.lease = AdmissionLease(this);
    }
    return result;
}

void SleepLifecycleController::releaseAdmission() {
    {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        if (active_admissions_ <= 0) {
            RTP_LLM_LOG_ERROR("sleep lifecycle admission lease released with no active admission");
            return;
        }
        --active_admissions_;
    }
}

int64_t SleepLifecycleController::activeAdmissionCount() const {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    return active_admissions_;
}

int64_t SleepLifecycleController::sleepEpoch() const {
    return sleep_epoch_.load(std::memory_order_acquire);
}

SleepState SleepLifecycleController::state() const {
    return state_.load(std::memory_order_acquire);
}

}  // namespace rtp_llm
