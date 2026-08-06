#pragma once

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <exception>
#include <functional>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::cuda_graph {

struct GraphLifecycleContext {
    uint64_t owner_token{0};
    uint64_t generation{0};
};

// Owns one ROCm graph-communicator lease. CUDA constructs the same object but
// receives an inactive context from the device shim.
class GraphOwnerLease {
public:
    using AcquireFn = std::function<GraphLifecycleContext(uintptr_t)>;
    using ReleaseFn = std::function<void(const GraphLifecycleContext&)>;

    GraphOwnerLease(AcquireFn acquire_fn, ReleaseFn release_fn):
        acquire_fn_(std::move(acquire_fn)), release_fn_(std::move(release_fn)) {}
    ~GraphOwnerLease() noexcept {
        reset();
    }

    GraphOwnerLease(const GraphOwnerLease&)            = delete;
    GraphOwnerLease& operator=(const GraphOwnerLease&) = delete;

    void acquire(uintptr_t owner_id) {
        RTP_LLM_CHECK_WITH_INFO(!acquired_, "graph owner lease was already acquired");
        context_  = acquire_fn_(owner_id);
        acquired_ = true;
        active_   = context_.owner_token != 0;
    }

    void reset() noexcept {
        if (!acquired_) {
            return;
        }
        if (active_) {
            try {
                release_fn_(context_);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("Failed to release ROCm graph owner lease token=%" PRIu64 " generation=%" PRIu64
                                  ": %s",
                                  context_.owner_token,
                                  context_.generation,
                                  e.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("Failed to release ROCm graph owner lease token=%" PRIu64 " generation=%" PRIu64
                                  ": unknown error",
                                  context_.owner_token,
                                  context_.generation);
            }
        }
        context_  = {};
        acquired_ = false;
        active_   = false;
    }

    const GraphLifecycleContext& context() const {
        return context_;
    }

    const GraphLifecycleContext* contextPtr() const {
        return &context_;
    }

private:
    GraphLifecycleContext context_{};
    bool                  acquired_{false};
    bool                  active_{false};
    AcquireFn             acquire_fn_;
    ReleaseFn             release_fn_;
};

enum class CaptureGuardOrder {
    BEFORE_CAPTURE_BEGIN,
    AFTER_CAPTURE_BEGIN,
};

template<typename BeginPlanning, typename Forward, typename Prepare, typename Cancel>
void runCapturePlanning(BeginPlanning&& begin_planning, Forward&& forward, Prepare&& prepare, Cancel&& cancel) {
    try {
        begin_planning();
        forward();
        // The second warm-up replaces the first pass's occurrence plan.
        begin_planning();
        forward();
        prepare();
    } catch (...) {
        cancel();
        throw;
    }
}

template<typename Enter, typename Begin, typename Capture, typename End, typename Exit>
void runCaptureTransaction(
    CaptureGuardOrder order, Enter&& enter, Begin&& begin, Capture&& capture, End&& end, Exit&& exit) {
    bool entered = false;
    bool started = false;
    try {
        if (order == CaptureGuardOrder::BEFORE_CAPTURE_BEGIN) {
            enter();
            entered = true;
        }
        begin();
        started = true;
        if (order == CaptureGuardOrder::AFTER_CAPTURE_BEGIN) {
            enter();
            entered = true;
        }
        capture();
        // capture_end may consume the active capture before throwing. Never
        // retry it from the unwind path once the call has begun.
        started = false;
        end();
    } catch (...) {
        if (started) {
            try {
                end();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("Graph capture rollback failed to end capture: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("Graph capture rollback failed to end capture: unknown error");
            }
        }
        if (entered) {
            try {
                exit();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("Graph capture rollback failed to exit capture mode: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("Graph capture rollback failed to exit capture mode: unknown error");
            }
        }
        throw;
    }
    if (entered) {
        exit();
    }
}

class ShutdownSynchronizationGate {
public:
    bool claim(bool synchronization_required) noexcept {
        if (!synchronization_required || claimed_.exchange(true)) {
            return false;
        }
        return true;
    }

private:
    std::atomic<bool> claimed_{false};
};

}  // namespace rtp_llm::cuda_graph
