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
    std::lock_guard<std::mutex> lock(transition_mutex_);
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
    const SleepState current = state_.load(std::memory_order_acquire);
    if (current != expected_from || !isLegalTransition(expected_from, to)) {
        setLastError("illegal transition: " + sleepStateToString(current) + " -> " + sleepStateToString(to));
        return false;
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
    if (opt.level == 0) {
        return SleepResult::unimplemented(
            "sleep rejected: level=0 state-preserving sleep is defined but not implemented; supported_levels=[1]");
    }
    if (opt.level != 1) {
        return SleepResult::invalidArgument("sleep rejected: unknown level=" + std::to_string(opt.level)
                                            + "; supported_levels=[1], defined_unimplemented_levels=[0]");
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
        sleep_epoch_.fetch_add(1, std::memory_order_acq_rel);
        engine_quiesced_.store(false, std::memory_order_release);
        if (!transitionLocked(SleepState::RUNNING, SleepState::DRAINING)) {
            return SleepResult::failedPrecondition(status().last_error);
        }
    }

    // --- DRAINING: wait for in-flight requests and cache transfers (M3). ---
    if (!opt.commit_only && hooks_.drain) {
        if (!hooks_.drain(opt)) {
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

    // --- SUSPENDING: dereg MR, release memory backing (M7/M5/M6). ---
    // Order per design doc: engine already quiesced in prepare; CUDA sync + dereg MR
    // happen before pausing KV physical memory; CPU-backed persistent allocations
    // are released last.
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

    // --- WAKING_UP: restore memory backing, reset metadata, reg MR, warmup (M5/M6/M7). ---
    bool ok = true;
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
    if (!opt.commit_only && ok && hooks_.restoreRestorableGpuMemory) {
        ok = invokeHookNoThrow("restoreRestorableGpuMemory", hooks_.restoreRestorableGpuMemory);
        if (!ok) {
            setLastError("restoreRestorableGpuMemory failed");
        }
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
    s.sleep_mode_enabled    = enabled();
    s.effective             = effective();
    s.supported_levels      = s.effective ? std::vector<int32_t>{1} : std::vector<int32_t>{};
    s.supported_modes       = s.effective ? std::vector<std::string>{"wait", "abort"} : std::vector<std::string>{};
    s.disabled_reason       = s.effective ? "" : disabledReason();
    s.state                 = state_.load(std::memory_order_acquire);
    s.sleep_epoch           = sleep_epoch_.load(std::memory_order_acquire);
    s.kv_memory_state       = kvMemoryStateToString(kv_memory_state_.load(std::memory_order_acquire));
    s.device_kv_cache_valid = device_kv_cache_valid_.load(std::memory_order_acquire);
    if (hooks_.activeRequestCount) {
        s.active_request_count = hooks_.activeRequestCount();
    }
    if (hooks_.activeCacheTransferCount) {
        s.active_cache_transfer_count = hooks_.activeCacheTransferCount();
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

int64_t SleepLifecycleController::sleepEpoch() const {
    return sleep_epoch_.load(std::memory_order_acquire);
}

SleepState SleepLifecycleController::state() const {
    return state_.load(std::memory_order_acquire);
}

}  // namespace rtp_llm
