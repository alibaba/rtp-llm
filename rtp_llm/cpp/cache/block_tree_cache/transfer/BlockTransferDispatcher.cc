#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <memory>
#include <utility>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine):
    per_rank_engine_(std::move(per_rank_engine)), multi_rank_engine_(std::move(multi_rank_engine)) {}

bool BlockTransferDispatcher::executePerRank(const TransferDescriptor& descriptor) const {
    const std::shared_ptr<AsyncContext> context = per_rank_engine_->submit(descriptor);
    context->waitDone();
    if (!context->success()) {
        RTP_LLM_LOG_WARNING("per-rank block transfer failed: %s", context->errorInfo().ToString().c_str());
        return false;
    }
    return true;
}

std::shared_ptr<AsyncContext>
BlockTransferDispatcher::executePerRank(const std::vector<TransferDescriptor>& descriptors) const {
    if (descriptors.empty()) {
        return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    }
    return per_rank_engine_->submit(descriptors);
}

std::shared_ptr<AsyncContext>
BlockTransferDispatcher::executeMultiRank(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const {
    if (descriptors.empty()) {
        return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    }
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(descriptors, timeout_ms);
    }
    return executePerRank(descriptors);
}

bool BlockTransferDispatcher::hasMultiRankEngine() const {
    return multi_rank_engine_ != nullptr;
}

}  // namespace rtp_llm
