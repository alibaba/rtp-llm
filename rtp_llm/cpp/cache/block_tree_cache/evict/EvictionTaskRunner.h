#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTreeTaskPool;
class EvictionTaskRunner {
public:
    EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                       const BlockTransferDispatcher*  transfer_dispatcher,
                       BlockTreeTaskPool*              task_pool,
                       BlockTreeCacheMetricsReporter&  metrics_reporter);

    bool submit(std::shared_ptr<const EvictionTransferTask> task, std::function<void(bool)> on_done) const;

private:
    void run(std::shared_ptr<const EvictionTransferTask> task, std::function<void(bool)> on_done) const;

    const std::vector<GroupSetPtr>& group_sets_;
    const BlockTransferDispatcher*  transfer_dispatcher_{nullptr};
    BlockTreeTaskPool*              task_pool_;
    BlockTreeCacheMetricsReporter&  metrics_reporter_;
};

}  // namespace rtp_llm
