#include "rtp_llm/cpp/cache/AsyncContext.h"

#include <mutex>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

// --------------------------------- CompletedAsyncContext ---------------------------------

CompletedAsyncContext::CompletedAsyncContext(ErrorInfo error_info): error_info_(std::move(error_info)) {}

void CompletedAsyncContext::waitDone() {}

void CompletedAsyncContext::onDone(DoneCallback callback) {
    if (callback) {
        callback(error_info_);
    }
}

bool CompletedAsyncContext::done() const {
    return true;
}

bool CompletedAsyncContext::success() const {
    return error_info_.ok();
}

ErrorInfo CompletedAsyncContext::errorInfo() const {
    return error_info_;
}

// --------------------------------- FusedAsyncContext ---------------------------------

FusedAsyncContext::FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts): contexts_(contexts) {}

void FusedAsyncContext::waitDone() {
    RTP_LLM_PROFILE_FUNCTION();
    for (size_t i = 0; i < contexts_.size(); i++) {
        if (contexts_[i]) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("wait_sub_context[%zu]", i);
            contexts_[i]->waitDone();
        }
    }
    RTP_LLM_LOG_DEBUG("fused async context wait done, success: %d", success());
}

void FusedAsyncContext::onDone(DoneCallback callback) {
    if (!callback) {
        return;
    }
    size_t remaining = 0;
    for (const auto& context : contexts_) {
        remaining += context != nullptr;
    }
    if (remaining == 0) {
        callback(ErrorInfo::OkStatus());
        return;
    }

    struct CallbackState {
        std::mutex   mutex;
        size_t       remaining{0};
        ErrorInfo    first_error{ErrorInfo::OkStatus()};
        DoneCallback callback;
    };
    auto state       = std::make_shared<CallbackState>();
    state->remaining = remaining;
    state->callback  = std::move(callback);
    for (const auto& context : contexts_) {
        if (!context) {
            continue;
        }
        context->onDone([state](ErrorInfo error) mutable {
            DoneCallback callback;
            ErrorInfo    result = ErrorInfo::OkStatus();
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!error.ok() && state->first_error.ok()) {
                    state->first_error = std::move(error);
                }
                if (--state->remaining == 0) {
                    result   = state->first_error;
                    callback = std::move(state->callback);
                }
            }
            if (callback) {
                callback(std::move(result));
            }
        });
    }
}

bool FusedAsyncContext::done() const {
    for (const auto& context : contexts_) {
        if (context && !context->done()) {
            return false;
        }
    }
    return true;
}

bool FusedAsyncContext::success() const {
    for (const auto& context : contexts_) {
        if (context && !context->success()) {
            RTP_LLM_LOG_DEBUG("fused async context success is false, context error info: %s",
                              context->errorInfo().ToString().c_str());
            return false;
        }
    }
    return true;
}

ErrorInfo FusedAsyncContext::errorInfo() const {
    for (const auto& context : contexts_) {
        if (context && !context->success()) {
            return context->errorInfo();
        }
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
