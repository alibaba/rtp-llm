#include "rtp_llm/cpp/cache/AsyncContext.h"

#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

// --------------------------------- CompletedAsyncContext ---------------------------------

CompletedAsyncContext::CompletedAsyncContext(ErrorInfo error_info): error_info_(std::move(error_info)) {}

void CompletedAsyncContext::waitDone() {}

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
