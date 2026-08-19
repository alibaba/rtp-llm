#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;

class BlockTransferDispatcher {
public:
    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine = nullptr);

    std::shared_ptr<AsyncContext> executePerRank(const std::vector<TransferDescriptor>& descriptors) const;
    std::shared_ptr<AsyncContext> executeMultiRank(const std::vector<TransferDescriptor>& descriptors,
                                                   int                                    timeout_ms) const;

private:
    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
