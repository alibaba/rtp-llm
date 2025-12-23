#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

namespace rtp_llm {

// --------------------------------- FusedAsyncContext ---------------------------------

FusedAsyncContext::FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts): contexts_(contexts) {}

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
                                             const std::shared_ptr<KVCacheResource>&   resource):
    fused_match_context_(fused_match_context), resource_(resource) {}

FusedAsyncReadContext::~FusedAsyncReadContext() {}

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
    return fused_read_context_ && fused_read_context_->done();
}

bool FusedAsyncReadContext::success() const {
    return done() && (fused_match_context_ && fused_match_context_->success())
           && (!fused_read_context_ || fused_read_context_->success());
}

void FusedAsyncReadContext::setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context) {
    fused_read_context_ = fused_read_context;
}

}  // namespace rtp_llm