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
    explicit TransferBatchAsyncContext(size_t block_count = 0, std::shared_ptr<void> completion_guard = nullptr):
        block_count_(block_count), completion_guard_(std::move(completion_guard)) {}

    void           waitDone() override;
    void           onDone(DoneCallback callback) override;
    bool           done() const override;
    bool           success() const override;
    TransferResult result() const override;

    void complete(TransferResult result);

private:
    mutable std::mutex        mutex_;
    std::condition_variable   done_cv_;
    bool                      done_{false};
    TransferResult            result_;
    size_t                    block_count_{0};
    std::shared_ptr<void>     completion_guard_;
    std::vector<DoneCallback> callbacks_;
};

}  // namespace rtp_llm
