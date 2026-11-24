#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"

namespace rtp_llm {

class TransferBatchAsyncContext final: public TransferAsyncContext {
public:
    explicit TransferBatchAsyncContext(size_t                                block_count      = 0,
                                       std::shared_ptr<void>                 completion_guard = nullptr,
                                       std::shared_ptr<TransferBufferSafety> proof            = nullptr):
        block_count_(block_count),
        completion_guard_(std::move(completion_guard)),
        completion_proof_(std::move(proof)) {}

    bool           tryAdoptPendingCallbacks(CallbackList& callbacks) noexcept override;
    void           waitDone() override;
    void           onDone(DoneCallback callback) override;
    bool           done() const override;
    bool           success() const override;
    TransferResult result() const override;

    void complete(TransferResult result);

private:
    friend class BlockTransferDispatcher;

    void protectPhysicalCallbacks() noexcept {
        physical_callback_count_ = callbacks_.size() + (first_callback_ ? 1 : 0);
    }
    void holdUntilCompletion(std::shared_ptr<void> guard) noexcept {
        completion_guard_ = std::move(guard);
    }

    mutable std::mutex                    mutex_;
    std::condition_variable               done_cv_;
    bool                                  done_{false};
    TransferResult                        result_;
    size_t                                block_count_{0};
    size_t                                physical_callback_count_{0};
    std::shared_ptr<void>                 completion_guard_;
    std::shared_ptr<TransferBufferSafety> completion_proof_;
    DoneCallback                          first_callback_;
    CallbackList                          callbacks_;
};

}  // namespace rtp_llm
