#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class TransferStageState {
public:
    using DoneCallback = std::function<void(ErrorInfo)>;

    explicit TransferStageState(DoneCallback callback): callback_(std::move(callback)) {}

    void addBatch() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (submitting_finished_ || completed_) {
            return;
        }
        ++remaining_;
    }

    void completeBatch(ErrorInfo error) {
        DoneCallback callback;
        ErrorInfo    result = ErrorInfo::OkStatus();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completed_ || remaining_ <= (submitting_finished_ ? 0u : 1u)) {
                return;
            }
            if (first_error_.ok() && !error.ok()) {
                first_error_ = std::move(error);
            }
            --remaining_;
            finishIfReadyLocked(callback, result);
        }
        if (callback) {
            callback(std::move(result));
        }
    }

    void finishSubmitting() {
        DoneCallback callback;
        ErrorInfo    result = ErrorInfo::OkStatus();
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
    void finishIfReadyLocked(DoneCallback& callback, ErrorInfo& result) {
        if (remaining_ != 0 || completed_) {
            return;
        }
        completed_ = true;
        result     = first_error_;
        callback   = std::move(callback_);
    }

    std::mutex   mutex_;
    DoneCallback callback_;
    ErrorInfo    first_error_{ErrorInfo::OkStatus()};
    size_t       remaining_{1};
    bool         submitting_finished_{false};
    bool         completed_{false};
};

}  // namespace rtp_llm
