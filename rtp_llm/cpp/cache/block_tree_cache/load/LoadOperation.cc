#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"

#include <map>
#include <utility>

namespace rtp_llm {

LoadOperation::LoadOperation(TransferDescriptor descriptor, GroupSetPtr group_set, uint64_t workflow_id):
    descriptor_(std::move(descriptor)), group_set_(std::move(group_set)), workflow_id_(workflow_id) {}

void LoadOperation::unreferenceSource() {
    group_set_->unreferenceBlocks(descriptor_.source_tier, descriptor_.source_blocks, BlockTreeRefType::LOAD);
}

void LoadOperation::complete(bool success) {
    std::map<uint64_t, std::function<void(ErrorInfo)>> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) {
            return;
        }
        if (!success) {
            error_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load settlement failed");
        }
        done_ = true;
        callbacks.swap(callbacks_);
    }
    for (auto& [_, callback] : callbacks) {
        callback(error_);
    }
}

void LoadOperation::subscribe(uint64_t subscriber_id, std::function<void(ErrorInfo)> callback) {
    if (!callback) {
        return;
    }
    ErrorInfo error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!done_) {
            callbacks_.emplace(subscriber_id, std::move(callback));
            return;
        }
        error = error_;
    }
    callback(std::move(error));
}

void LoadOperation::unsubscribe(uint64_t subscriber_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.erase(subscriber_id);
}

}  // namespace rtp_llm
