#include "rtp_llm/cpp/cache/AsyncContext.h"

#include <utility>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
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

// --------------------------------- FusedAsyncReadContext ---------------------------------

FusedAsyncReadContext::FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context,
                                             const std::shared_ptr<KVCacheResource>&   resource):
    fused_match_context_(fused_match_context), resource_(resource) {}

void FusedAsyncReadContext::waitDone() {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [&] { return done(); });
}

void FusedAsyncReadContext::notifyDone() {
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_cv_.notify_all();
}

void FusedAsyncReadContext::cancel() {
    cancelled_.store(true, std::memory_order_release);
    notifyDone();
}

bool FusedAsyncReadContext::cancelled() const {
    return cancelled_.load(std::memory_order_acquire);
}

bool FusedAsyncReadContext::done() const {
    if (cancelled()) {
        return true;
    }
    if (!fused_match_context_) {
        return true;
    }
    if (!fused_match_context_->done()) {
        return false;
    }
    if (!fused_match_context_->success()) {
        return true;
    }
    std::lock_guard<std::mutex> lock(read_ctx_mutex_);
    if (!read_ctx_set_.load()) {
        return false;
    }
    return !fused_read_context_ || fused_read_context_->done();
}

bool FusedAsyncReadContext::success() const {
    if (cancelled()) {
        return false;
    }
    if (done() && (fused_match_context_ && fused_match_context_->success())) {
        std::lock_guard<std::mutex> lk(read_ctx_mutex_);
        return !fused_read_context_ || fused_read_context_->success();
    }
    return false;
}

ErrorInfo FusedAsyncReadContext::errorInfo() const {
    if (cancelled()) {
        return ErrorInfo(ErrorCode::CANCELLED, "async read cancelled");
    }
    if (fused_match_context_ && !fused_match_context_->success()) {
        return fused_match_context_->errorInfo();
    }
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    if (fused_read_context_ && !fused_read_context_->success()) {
        return fused_read_context_->errorInfo();
    }
    return ErrorInfo::OkStatus();
}

void FusedAsyncReadContext::setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context) {
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    fused_read_context_ = fused_read_context;
    read_ctx_set_.store(true);
}

const std::shared_ptr<FusedAsyncContext> FusedAsyncReadContext::fusedReadContext() const {
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    return fused_read_context_;
}

const std::shared_ptr<FusedAsyncContext>& FusedAsyncReadContext::fusedMatchContext() const {
    return fused_match_context_;
}

const std::shared_ptr<KVCacheResource>& FusedAsyncReadContext::resource() const {
    return resource_;
}

}  // namespace rtp_llm
