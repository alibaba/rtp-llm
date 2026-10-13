#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTreeTaskPool;
class PerRankBlockTransferEngine;

class TransferExecutor {
public:
    virtual ~TransferExecutor() = default;

protected:
    TransferExecutor(BlockTreeTaskPool&                             transfer_task_pool,
                     size_t                                         max_descriptors_per_batch,
                     std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter);

    virtual TransferStatus executeBatch(const std::vector<HostBufferView>&     hosts,
                                        const std::vector<TransferDescriptor>& descriptors,
                                        const std::vector<const GroupSet*>&    group_sets) = 0;

private:
    friend class PerRankBlockTransferEngine;

    std::shared_ptr<AsyncContext>
    execute(TransferTask task, std::vector<HostBufferView> hosts, std::vector<const GroupSet*> group_sets);

    BlockTreeTaskPool&                             transfer_task_pool_;
    size_t                                         max_descriptors_per_batch_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
};

}  // namespace rtp_llm
