#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace rtp_llm {

// Construct only for an enabled publication backend. A lease charges retained
// native pages, not another pool. Unknown completion must retain the lease
// alongside its page/registration owners until buffer safety is established.
class DecodePublicationBudget {
    struct State {
        State(size_t bytes, size_t jobs): byte_limit(bytes), job_limit(jobs) {}
        std::mutex mutex;
        const size_t byte_limit, job_limit;
        size_t bytes{0}, jobs{0};
        bool accepting{true};
    };

public:
    class Lease {
        friend class DecodePublicationBudget;
        Lease(std::shared_ptr<State> state, size_t bytes): state_(std::move(state)), bytes_(bytes) {}
        std::shared_ptr<State> state_;
        size_t bytes_;
        bool armed_{false};

    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() {
            if (!armed_) {
                return;
            }
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->bytes -= bytes_;
            --state_->jobs;
        }
    };
    struct Snapshot {
        size_t bytes, jobs;
        bool accepting;
    };

    DecodePublicationBudget(size_t byte_limit, size_t job_limit) {
        if (!byte_limit || !job_limit) {
            throw std::invalid_argument("publication byte/job limits must be positive");
        }
        state_ = std::make_shared<State>(byte_limit, job_limit);
    }

    std::shared_ptr<Lease> tryAcquire(size_t bytes) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->accepting || !bytes || bytes > state_->byte_limit
            || state_->bytes > state_->byte_limit - bytes || state_->jobs >= state_->job_limit) {
            return {};
        }
        // Allocate the control block before charging. If either allocation
        // throws, destruction of the unarmed lease cannot re-lock this mutex.
        auto lease = std::shared_ptr<Lease>(new Lease(state_, bytes));
        state_->bytes += bytes;
        ++state_->jobs;
        lease->armed_ = true;
        return lease;
    }

    void stopAdmission() {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->accepting = false;
    }

    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return {state_->bytes, state_->jobs, state_->accepting};
    }

private:
    std::shared_ptr<State> state_;
};

}  // namespace rtp_llm
