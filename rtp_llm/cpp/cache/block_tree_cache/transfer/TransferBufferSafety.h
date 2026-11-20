#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace rtp_llm {

// A producer owns one token for its entire physical operation. Business
// completion, an unproven result and destruction never supply a safety proof.
// Install the token before submission; its callback must not throw.
class TransferBufferSafety {
public:
    explicit TransferBufferSafety(std::function<void()> safe_callback): safe_callback_(std::move(safe_callback)) {}

    void report(bool buffer_safe) noexcept {
        if (!buffer_safe || reported_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        auto callback = std::move(safe_callback_);
        if (callback) {
            callback();
        }
    }

private:
    std::atomic<bool>     reported_{false};
    std::function<void()> safe_callback_;
};

// One local execution may submit several physical batches. Even synchronously
// completed batches cannot publish safety while more batches can still start.
class TransferBufferSafetyTracker: public std::enable_shared_from_this<TransferBufferSafetyTracker> {
public:
    explicit TransferBufferSafetyTracker(std::function<void()> safe_callback):
        safe_callback_(std::move(safe_callback)) {}

    std::shared_ptr<TransferBufferSafety> addBatch() {
        // Complete all allocation before changing the number of required proofs.
        auto proof = std::make_shared<TransferBufferSafety>([self = shared_from_this()] { self->completeBatch(); });
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (submitting_finished_) {
                throw std::logic_error("cannot add a transfer batch after submission finished");
            }
            ++remaining_;
        }
        return proof;
    }

    void finishSubmitting() noexcept {
        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (submitting_finished_) {
                return;
            }
            submitting_finished_ = true;
            if (remaining_ == 0) {
                callback.swap(safe_callback_);
            }
        }
        if (callback) {
            callback();
        }
    }

private:
    void completeBatch() noexcept {
        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --remaining_;
            if (submitting_finished_ && remaining_ == 0) {
                callback.swap(safe_callback_);
            }
        }
        if (callback) {
            callback();
        }
    }

    std::mutex            mutex_;
    size_t                remaining_{0};
    bool                  submitting_finished_{false};
    std::function<void()> safe_callback_;
};

}  // namespace rtp_llm
