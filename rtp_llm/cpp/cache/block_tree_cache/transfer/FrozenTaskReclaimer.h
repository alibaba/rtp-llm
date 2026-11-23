#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

struct TransferTargetBlock {
    std::shared_ptr<IBlockPool> pool;
    BlockIdxType block;
};

class FrozenTaskReclaimer {
public:
    virtual ~FrozenTaskReclaimer() = default;
    virtual void freezeTargetBlocksWithTraceIDAndStillIOWorkerID(const TransferCompletionKey& trace,
                                                                 std::vector<TransferTargetBlock> target_blocks,
                                                                 const std::vector<size_t>& pending_workers) noexcept = 0;
    virtual bool markWorkerSafe(const TransferCompletionKey& trace, size_t worker) = 0;
};

}
