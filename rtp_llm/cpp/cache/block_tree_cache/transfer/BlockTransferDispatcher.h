#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;

class BlockTransferDispatcher {
public:
    using TransferDoneCallback = std::function<void(ErrorInfo)>;

    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine          = nullptr,
                            size_t                                        max_descriptors_per_batch = 64);

    std::shared_ptr<AsyncContext> executePerRank(const std::vector<TransferDescriptor>& descriptors) const;
    std::shared_ptr<AsyncContext> executeMultiRank(const std::vector<TransferDescriptor>& descriptors,
                                                   int                                    timeout_ms) const;

    // Synchronous compatibility path used by Evict: singleton descriptors, strictly serial.
    bool runTransfer(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const;

    // Callback-driven path used by Load/Store: stable grouping and bounded batches.
    void runTransfer(const std::vector<TransferDescriptor>& descriptors,
                     int                                    timeout_ms,
                     TransferDoneCallback                   callback) const;

    void cancelPendingStagingTransfers() const;
    void shutdown() const;

private:
    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
    size_t                                        max_descriptors_per_batch_{64};
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
