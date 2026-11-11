#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

#include <utility>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
TransferResult unknownCompletionWithoutAllocation() noexcept {
    auto result = TransferResult::executionException(false);
    result.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    result.copy_error = CacheCopyError::RPC_FAILED;
    return result;
}

[[noreturn]] void failCompletionDelivery() noexcept {
    std::fputs("transfer completion could not be delivered safely\n", stderr);
    std::fflush(stderr);
    std::abort();
}

void invokeCompletion(TransferAsyncContext::DoneCallback& callback, TransferResult result, bool isolate_known_errors) {
    const bool unknown = result.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN;
    try {
        callback(std::move(result));
    } catch (const std::bad_alloc&) {
        // A business consumer may not have established frozen custody yet.
        failCompletionDelivery();
    } catch (...) {
        if (unknown) {
            failCompletionDelivery();
        }
        if (!isolate_known_errors) {
            throw;
        }
        try {
            RTP_LLM_LOG_WARNING("transfer completion callback failed");
        } catch (...) {}
    }
}
}  // namespace

void TransferBatchAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return done_; });
}

void TransferBatchAsyncContext::onDone(DoneCallback callback) {
    if (!callback) {
        return;
    }
    bool           run_now = false;
    TransferResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            run_now = true;
            try {
                result = result_;
            } catch (...) {
                result = unknownCompletionWithoutAllocation();
            }
        } else {
            callbacks_.push_back(std::move(callback));
        }
    }
    if (run_now) {
        invokeCompletion(callback, std::move(result), false);
    }
}

bool TransferBatchAsyncContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool TransferBatchAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && result_.ok();
}

TransferResult TransferBatchAsyncContext::result() const {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        return result_;
    } catch (...) {
        return unknownCompletionWithoutAllocation();
    }
}

void TransferBatchAsyncContext::complete(TransferResult result) {
    // Producer evidence precedes diagnostic expansion and mandatory delivery.
    // A late safe result may advance proof without replacing the business result.
    if (completion_proof_) {
        completion_proof_->report(result.buffer_safe);
    }
    std::shared_ptr<void>     completion_guard;
    std::vector<DoneCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            return;
        }
        try {
            result.expandBlockErrors(block_count_);
        } catch (...) {
            // Empty block_errors intentionally encodes a uniform conservative
            // failure without requiring a second allocation under memory pressure.
            result = unknownCompletionWithoutAllocation();
        }
        result_          = std::move(result);
        done_            = true;
        completion_guard = std::move(completion_guard_);
        callbacks.swap(callbacks_);
    }
    done_cv_.notify_all();
    for (auto& callback : callbacks) {
        TransferResult delivery;
        try {
            delivery = result_;
        } catch (...) {
            delivery = unknownCompletionWithoutAllocation();
        }
        invokeCompletion(callback, std::move(delivery), true);
    }
}

}  // namespace rtp_llm
