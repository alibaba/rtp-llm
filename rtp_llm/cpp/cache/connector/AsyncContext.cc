#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

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

std::optional<int64_t> FusedAsyncContext::readyTimeUs() const {
    if (!done()) {
        return std::nullopt;
    }
    int64_t ready_time_us = 0;
    for (const auto& context : contexts_) {
        if (!context) {
            continue;
        }
        const auto child_ready_time_us = context->readyTimeUs();
        if (!child_ready_time_us) {
            return std::nullopt;
        }
        ready_time_us = std::max(ready_time_us, *child_ready_time_us);
    }
    return ready_time_us;
}

// --------------------------------- FusedAsyncReadContext ---------------------------------

FusedAsyncReadContext::FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context,
                                             const std::shared_ptr<KVCacheResource>&   resource,
                                             const std::shared_ptr<Meta>&              meta):
    fused_match_context_(fused_match_context), resource_(resource), meta_(meta) {}

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

std::optional<int64_t> FusedAsyncReadContext::readyTimeUs() const {
    if (!done() || !fused_match_context_) {
        return std::nullopt;
    }
    const auto match_ready_time_us = fused_match_context_->readyTimeUs();
    if (!match_ready_time_us) {
        return std::nullopt;
    }
    if (!fused_match_context_->success()) {
        return match_ready_time_us;
    }

    std::shared_ptr<FusedAsyncContext> read_context;
    int64_t                            read_context_set_time_us = 0;
    {
        std::lock_guard<std::mutex> lock(read_ctx_mutex_);
        if (!read_ctx_set_.load(std::memory_order_acquire) || read_context_set_time_us_ == 0) {
            return std::nullopt;
        }
        read_context             = fused_read_context_;
        read_context_set_time_us = read_context_set_time_us_;
    }

    int64_t ready_time_us = std::max(*match_ready_time_us, read_context_set_time_us);
    if (read_context) {
        const auto read_ready_time_us = read_context->readyTimeUs();
        if (!read_ready_time_us) {
            return std::nullopt;
        }
        ready_time_us = std::max(ready_time_us, *read_ready_time_us);
    }
    return ready_time_us;
}

void FusedAsyncReadContext::setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context) {
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    fused_read_context_        = fused_read_context;
    read_context_set_time_us_  = currentTimeUs();
    read_ctx_set_.store(true, std::memory_order_release);
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