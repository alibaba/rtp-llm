#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>

namespace rtp_llm {

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
                                             const std::shared_ptr<KVCacheResource>&   resource,
                                             const std::shared_ptr<Meta>&              meta,
                                             bool                                      has_async_cache_dependency):
    fused_match_context_(fused_match_context), resource_(resource), meta_(meta) {
    metrics_snapshot_.has_async_cache_dependency = has_async_cache_dependency;
}

void FusedAsyncReadContext::waitDone() {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [&] { return done(); });
}

void FusedAsyncReadContext::notifyDone() {
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_cv_.notify_all();
}

bool FusedAsyncReadContext::done() const {
    std::lock_guard<std::mutex> lock(read_ctx_mutex_);
    if (fused_match_context_) {
        if (!fused_match_context_->done()) {
            return false;
        }
        if (fused_match_context_->success()
            && (!read_ctx_set_.load() || (fused_read_context_ && !fused_read_context_->done()))) {
            return false;
        }
    }
    // The legacy context publishes completion when its aggregate done() first
    // succeeds, whether polled by the coordinator or the scheduler.
    if (!metrics_snapshot_.terminal_time_us) {
        metrics_snapshot_.success = fused_match_context_ && fused_match_context_->success()
                                    && (!fused_read_context_ || fused_read_context_->success());
        metrics_snapshot_.terminal_time_us = currentTimeUs();
    }
    return true;
}

std::optional<CacheLoadTerminalSnapshot> FusedAsyncReadContext::cacheLoadMetricsSnapshot() const {
    std::lock_guard<std::mutex> lock(read_ctx_mutex_);
    return metrics_snapshot_;
}

bool FusedAsyncReadContext::success() const {
    if (done() && (fused_match_context_ && fused_match_context_->success())) {
        std::lock_guard<std::mutex> lk(read_ctx_mutex_);
        return !fused_read_context_ || fused_read_context_->success();
    }
    return false;
}

ErrorInfo FusedAsyncReadContext::errorInfo() const {
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
    if (fused_read_context_) {
        const auto& contexts = fused_read_context_->contexts();
        metrics_snapshot_.has_async_cache_dependency |=
            std::any_of(contexts.begin(), contexts.end(), [](const auto& context) { return context != nullptr; });
    }
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

const std::shared_ptr<Meta>& FusedAsyncReadContext::meta() const {
    return meta_;
}

}  // namespace rtp_llm