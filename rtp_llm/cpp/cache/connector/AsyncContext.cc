#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

// --------------------------------- FusedAsyncContext ---------------------------------

FusedAsyncContext::FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts): contexts_(contexts) {}

void FusedAsyncContext::waitDone() {
    for (const auto& context : contexts_) {
        if (context) {
            context->waitDone();
        }
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
            return false;
        }
    }
    return true;
}

// --------------------------------- FusedAsyncReadContext ---------------------------------

FusedAsyncReadContext::FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context,
                                             const std::shared_ptr<KVCacheResource>&   resource,
                                             const std::shared_ptr<Meta>&              meta):
    fused_match_context_(fused_match_context), resource_(resource), meta_(meta) {}

FusedAsyncReadContext::~FusedAsyncReadContext() {
    setFusedReadContext(nullptr);
}

void FusedAsyncReadContext::waitDone() {
    if (!fused_match_context_) {
        return;
    }
    fused_match_context_->waitDone();
    // if match failed, there will be no read context
    if (!fused_match_context_->success()) {
        return;
    }

    std::shared_ptr<FusedAsyncContext> read_ctx;
    {
        std::unique_lock<std::mutex> lock(read_ctx_mutex_);
        read_ctx_cv_.wait(lock, [&] { return read_ctx_set_.load(); });
        read_ctx = fused_read_context_;
    }
    if (read_ctx) {
        read_ctx->waitDone();
    }
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

void FusedAsyncReadContext::setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context) {
    std::lock_guard<std::mutex> lk(read_ctx_mutex_);
    fused_read_context_ = fused_read_context;
    read_ctx_set_.store(true);
    read_ctx_cv_.notify_all();
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