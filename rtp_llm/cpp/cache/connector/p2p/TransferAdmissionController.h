#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace rtp_llm {

// Serializes transfer admission with transport lifecycle changes. A Lease owns
// the exact worker snapshot used by an operation and may be retained by the
// backend completion callback after the caller has timed out or been cancelled.
template<typename Worker>
class TransferAdmissionController {
private:
    struct State {
        mutable std::mutex      mutex;
        std::shared_ptr<Worker> worker;
        size_t                  inflight{0};
        bool                    admission_closed{false};
        bool                    transport_suspended{false};
        // Distinguishes an admission-only close (all levels) from a completed
        // transport rebuild that still needs its Level-3 resume callback.
        bool transport_needs_resume{false};
    };

public:
    class Lease {
    public:
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;

        ~Lease() {
            std::lock_guard<std::mutex> lock(state_->mutex);
            --state_->inflight;
        }

        const std::shared_ptr<Worker>& worker() const {
            return worker_;
        }

    private:
        friend class TransferAdmissionController;

        Lease(std::shared_ptr<State> state, std::shared_ptr<Worker> worker):
            state_(std::move(state)), worker_(std::move(worker)) {}

        std::shared_ptr<State>  state_;
        std::shared_ptr<Worker> worker_;
    };

    using LeasePtr = std::shared_ptr<Lease>;

    TransferAdmissionController(): state_(std::make_shared<State>()) {}

    explicit TransferAdmissionController(std::shared_ptr<Worker> worker): TransferAdmissionController() {
        state_->worker = std::move(worker);
    }

    bool installWorker(std::shared_ptr<Worker> worker) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!worker || state_->inflight != 0 || state_->transport_suspended) {
            return false;
        }
        state_->worker = std::move(worker);
        return true;
    }

    LeasePtr tryAcquire() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->admission_closed || state_->transport_suspended || !state_->worker) {
            return nullptr;
        }
        ++state_->inflight;
        return LeasePtr(new Lease(state_, state_->worker));
    }

    bool close() {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->admission_closed = true;
        return true;
    }

    template<typename StopTransport>
    bool teardown(StopTransport&& stop_transport) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->transport_suspended) {
            return true;
        }
        if (!state_->admission_closed || state_->inflight != 0 || !state_->worker) {
            return false;
        }
        if (!std::forward<StopTransport>(stop_transport)(*state_->worker)) {
            return false;
        }
        state_->transport_suspended = true;
        return true;
    }

    template<typename RebuildTransport>
    bool rebuild(RebuildTransport&& rebuild_transport) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->transport_suspended) {
            return true;
        }
        if (!state_->admission_closed || state_->inflight != 0 || !state_->worker) {
            return false;
        }
        if (!std::forward<RebuildTransport>(rebuild_transport)(*state_->worker)) {
            return false;
        }
        state_->transport_suspended    = false;
        state_->transport_needs_resume = true;
        return true;
    }

    template<typename ResumeTransport>
    bool resume(ResumeTransport&& resume_transport) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->transport_suspended || !state_->worker) {
            return false;
        }
        if (state_->transport_needs_resume) {
            if (!std::forward<ResumeTransport>(resume_transport)(*state_->worker)) {
                return false;
            }
            state_->transport_needs_resume = false;
        }
        state_->admission_closed = false;
        return true;
    }

    bool resume() {
        return resume([](Worker&) { return true; });
    }

    size_t inflightCount() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->inflight;
    }

private:
    std::shared_ptr<State> state_;
};

}  // namespace rtp_llm
