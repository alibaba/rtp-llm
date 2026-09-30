#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class TransferStageState {
public:
    using DoneCallback = std::function<void(TransferResult)>;

    explicit TransferStageState(DoneCallback callback, size_t block_count = 0):
        callback_(std::move(callback)), result_(block_count) {}

    void addBatch() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (submitting_finished_ || completed_) {
            return;
        }
        ++remaining_;
    }

    void completeBatch(TransferResult batch_result, const std::vector<size_t>& block_indices = {}) {
        DoneCallback   callback;
        TransferResult result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completed_ || remaining_ <= (submitting_finished_ ? 0u : 1u)) {
                return;
            }
            for (size_t i = 0; i < block_indices.size(); ++i) {
                result_.block_errors[block_indices[i]] = batch_result.blockError(i);
            }
            result_.merge(std::move(batch_result));
            --remaining_;
            finishIfReadyLocked(callback, result);
        }
        if (callback) {
            callback(std::move(result));
        }
    }

    void finishSubmitting() {
        DoneCallback   callback;
        TransferResult result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (submitting_finished_ || completed_) {
                return;
            }
            submitting_finished_ = true;
            --remaining_;
            finishIfReadyLocked(callback, result);
        }
        if (callback) {
            callback(std::move(result));
        }
    }

private:
    void finishIfReadyLocked(DoneCallback& callback, TransferResult& result) {
        if (remaining_ != 0 || completed_) {
            return;
        }
        completed_ = true;
        result     = result_;
        callback   = std::move(callback_);
    }

    std::mutex     mutex_;
    DoneCallback   callback_;
    TransferResult result_;
    size_t         remaining_{1};
    bool           submitting_finished_{false};
    bool           completed_{false};
};

}  // namespace rtp_llm
