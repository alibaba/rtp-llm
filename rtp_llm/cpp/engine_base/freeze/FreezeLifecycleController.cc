#include "rtp_llm/cpp/engine_base/freeze/FreezeLifecycleController.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::string freezeStateToString(FreezeState state) {
    switch (state) {
        case FreezeState::RUNNING:
            return "RUNNING";
        case FreezeState::DRAINING:
            return "DRAINING";
        case FreezeState::FREEZING:
            return "FREEZING";
        case FreezeState::FROZEN:
            return "FROZEN";
        case FreezeState::RESUMING:
            return "RESUMING";
        case FreezeState::ERROR:
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
        case KvMemoryState::RESUMING:
            return "RESUMING";
        default:
            return "UNKNOWN";
    }
}

void FreezeLifecycleController::setHooks(const FreezeHooks& hooks) {
    std::lock_guard<std::mutex> lock(transition_mutex_);
    hooks_ = hooks;
}

bool FreezeLifecycleController::isLegalTransition(FreezeState from, FreezeState to) {
    switch (from) {
        case FreezeState::RUNNING:
            return to == FreezeState::DRAINING || to == FreezeState::ERROR;
        case FreezeState::DRAINING:
            // freeze cancelled before release -> RUNNING; drained -> FREEZING.
            return to == FreezeState::FREEZING || to == FreezeState::RUNNING || to == FreezeState::ERROR;
        case FreezeState::FREEZING:
            return to == FreezeState::FROZEN || to == FreezeState::ERROR;
        case FreezeState::FROZEN:
            return to == FreezeState::RESUMING;
        case FreezeState::RESUMING:
            // rebuild ok -> RUNNING; rebuild failed -> FROZEN (never half-available).
            return to == FreezeState::RUNNING || to == FreezeState::FROZEN;
        case FreezeState::ERROR:
            // explicit recovery attempt only.
            return to == FreezeState::RESUMING;
        default:
            return false;
    }
}

bool FreezeLifecycleController::transitionLocked(FreezeState expected_from, FreezeState to) {
    const FreezeState current = state_.load(std::memory_order_acquire);
    if (current != expected_from || !isLegalTransition(expected_from, to)) {
        setLastError("illegal transition: " + freezeStateToString(current) + " -> " + freezeStateToString(to));
        return false;
    }
    state_.store(to, std::memory_order_release);
    RTP_LLM_LOG_INFO("freeze state transition: %s -> %s (epoch=%ld)",
                     freezeStateToString(expected_from).c_str(),
                     freezeStateToString(to).c_str(),
                     freeze_epoch_.load());
    return true;
}

void FreezeLifecycleController::setLastError(const std::string& msg) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    last_error_ = msg;
    if (!msg.empty()) {
        RTP_LLM_LOG_WARNING("freeze lifecycle: %s", msg.c_str());
    }
}

FreezeResult FreezeLifecycleController::freeze(const FreezeOptions& opt) {
    std::lock_guard<std::mutex> lock(transition_mutex_);

    if (opt.prepare_only && opt.commit_only) {
        return FreezeResult::error("freeze rejected: prepare_only and commit_only cannot both be true");
    }

    const FreezeState current = state_.load(std::memory_order_acquire);
    // Idempotency: already frozen or inside the release section. DRAINING is
    // intentionally retriable so a timeout can later progress, or be escalated
    // with force=true.
    if (current == FreezeState::FREEZING || current == FreezeState::FROZEN) {
        return FreezeResult::success();
    }
    if (opt.commit_only && current != FreezeState::DRAINING) {
        return FreezeResult::error("freeze commit rejected in state " + freezeStateToString(current));
    }
    if (current != FreezeState::RUNNING && current != FreezeState::DRAINING) {
        return FreezeResult::error("freeze rejected in state " + freezeStateToString(current));
    }

    setLastError("");

    if (current == FreezeState::RUNNING) {
        freeze_epoch_.fetch_add(1, std::memory_order_acq_rel);
        if (!transitionLocked(FreezeState::RUNNING, FreezeState::DRAINING)) {
            return FreezeResult::error(status().last_error);
        }
    }

    // --- DRAINING: wait for in-flight requests and cache transfers (M3). ---
    if (hooks_.drain) {
        if (!hooks_.drain(opt)) {
            // Per design: graceful drain timeout keeps DRAINING and does NOT
            // release GPU. The controller stays in DRAINING; control plane can
            // retry freeze (idempotent) or escalate with force.
            setLastError("drain not finished (timeout or aborted), staying in DRAINING");
            return FreezeResult::error("drain not finished, state=DRAINING");
        }
    }

    if (opt.prepare_only) {
        RTP_LLM_LOG_INFO("freeze prepare completed, staying in DRAINING (epoch=%ld)", freeze_epoch_.load());
        return FreezeResult::success();
    }

    if (!transitionLocked(FreezeState::DRAINING, FreezeState::FREEZING)) {
        return FreezeResult::error(status().last_error);
    }

    // --- FREEZING: quiesce engine, dereg MR, pause memory (M7/M5/M6). ---
    // Order per design doc: stop loop + CUDA sync + dereg MR BEFORE pausing
    // KV physical memory; weights pause last (cpu-backup).
    bool ok = true;
    if (ok && hooks_.deregMrAndQuiesceEngine) {
        ok = hooks_.deregMrAndQuiesceEngine(opt);
        if (!ok) {
            setLastError("deregMrAndQuiesceEngine failed");
        }
    }
    if (ok && hooks_.pauseKvMemory) {
        kv_memory_state_.store(KvMemoryState::PAUSING, std::memory_order_release);
        ok = hooks_.pauseKvMemory(opt);
        if (!ok) {
            setLastError("pauseKvMemory failed");
        }
    }
    if (ok) {
        kv_memory_state_.store(KvMemoryState::PAUSED, std::memory_order_release);
        device_kv_cache_valid_.store(false, std::memory_order_release);
    }
    if (ok && hooks_.pauseWeights) {
        ok = hooks_.pauseWeights(opt);
        if (!ok) {
            setLastError("pauseWeights failed");
        }
    }

    if (!ok) {
        transitionLocked(FreezeState::FREEZING, FreezeState::ERROR);
        return FreezeResult::error(status().last_error);
    }

    if (!transitionLocked(FreezeState::FREEZING, FreezeState::FROZEN)) {
        return FreezeResult::error(status().last_error);
    }
    return FreezeResult::success();
}

FreezeResult FreezeLifecycleController::resume() {
    std::lock_guard<std::mutex> lock(transition_mutex_);

    const FreezeState current = state_.load(std::memory_order_acquire);
    // Idempotency: already running.
    if (current == FreezeState::RUNNING) {
        return FreezeResult::success();
    }
    // Instance-level coordinator uses resume as the abort path for a prepared
    // freeze that never committed. No GPU resource was released in DRAINING.
    if (current == FreezeState::DRAINING) {
        setLastError("");
        if (!transitionLocked(FreezeState::DRAINING, FreezeState::RUNNING)) {
            return FreezeResult::error(status().last_error);
        }
        return FreezeResult::success();
    }
    if (current != FreezeState::FROZEN && current != FreezeState::ERROR) {
        return FreezeResult::error("resume rejected in state " + freezeStateToString(current));
    }

    setLastError("");
    if (!transitionLocked(current, FreezeState::RESUMING)) {
        return FreezeResult::error(status().last_error);
    }

    // --- RESUMING: re-back memory, reset metadata, reg MR, warmup (M5/M6/M7). ---
    bool ok = true;
    if (ok && hooks_.resumeKvMemory) {
        kv_memory_state_.store(KvMemoryState::RESUMING, std::memory_order_release);
        ok = hooks_.resumeKvMemory();
        if (!ok) {
            setLastError("resumeKvMemory failed");
        }
    }
    if (ok) {
        kv_memory_state_.store(KvMemoryState::ACTIVE, std::memory_order_release);
    }
    if (ok && hooks_.resetKvMetadata) {
        ok = hooks_.resetKvMetadata();
        if (!ok) {
            setLastError("resetKvMetadata failed");
        }
    }
    if (ok && hooks_.resumeWeights) {
        ok = hooks_.resumeWeights();
        if (!ok) {
            setLastError("resumeWeights failed");
        }
    }
    if (ok && hooks_.regMrAndResumeEngine) {
        ok = hooks_.regMrAndResumeEngine();
        if (!ok) {
            setLastError("regMrAndResumeEngine failed");
        }
    }
    if (ok && hooks_.warmupAndHealthCheck) {
        ok = hooks_.warmupAndHealthCheck();
        if (!ok) {
            setLastError("warmupAndHealthCheck failed");
        }
    }

    if (!ok) {
        // Per design: resume failure keeps the instance FROZEN, never half-available.
        transitionLocked(FreezeState::RESUMING, FreezeState::FROZEN);
        return FreezeResult::error(status().last_error);
    }

    if (!transitionLocked(FreezeState::RESUMING, FreezeState::RUNNING)) {
        return FreezeResult::error(status().last_error);
    }
    return FreezeResult::success();
}

FreezeStatus FreezeLifecycleController::status() const {
    FreezeStatus s;
    s.state                 = state_.load(std::memory_order_acquire);
    s.freeze_epoch          = freeze_epoch_.load(std::memory_order_acquire);
    s.kv_memory_state       = kvMemoryStateToString(kv_memory_state_.load(std::memory_order_acquire));
    s.device_kv_cache_valid = device_kv_cache_valid_.load(std::memory_order_acquire);
    if (hooks_.activeRequestCount) {
        s.active_request_count = hooks_.activeRequestCount();
    }
    if (hooks_.activeCacheTransferCount) {
        s.active_cache_transfer_count = hooks_.activeCacheTransferCount();
    }
    s.gpu_resource_state = (s.state == FreezeState::FROZEN) ? "RELEASED" : "ACTIVE";
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        s.last_error = last_error_;
    }
    return s;
}

bool FreezeLifecycleController::admit() const {
    return state_.load(std::memory_order_acquire) == FreezeState::RUNNING;
}

int64_t FreezeLifecycleController::freezeEpoch() const {
    return freeze_epoch_.load(std::memory_order_acquire);
}

FreezeState FreezeLifecycleController::state() const {
    return state_.load(std::memory_order_acquire);
}

}  // namespace rtp_llm
