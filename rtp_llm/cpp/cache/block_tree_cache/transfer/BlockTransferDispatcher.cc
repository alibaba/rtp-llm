#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine):
    per_rank_engine_(std::move(per_rank_engine)), multi_rank_engine_(std::move(multi_rank_engine)) {}

TransferStatus BlockTransferDispatcher::executePerRank(const TransferDescriptor& descriptor) const {
    if (per_rank_engine_ == nullptr) {
        RTP_LLM_LOG_WARNING("BlockTransferDispatcher: per-rank engine is not initialized");
        return TransferStatus::INVALID_ARGS;
    }
    return per_rank_engine_->submit(descriptor).status();
}

bool BlockTransferDispatcher::executeMultiRank(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const {
    if (descriptors.empty()) {
        return true;
    }
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(descriptors, timeout_ms);
    }
    for (const TransferDescriptor& descriptor : descriptors) {
        if (executePerRank(descriptor) != TransferStatus::OK) {
            return false;
        }
    }
    return true;
}

bool BlockTransferDispatcher::hasMultiRankEngine() const {
    return multi_rank_engine_ != nullptr;
}

}  // namespace rtp_llm
