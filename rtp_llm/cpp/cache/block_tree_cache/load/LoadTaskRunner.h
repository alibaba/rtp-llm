#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTransferDispatcher;
class BlockTreeTaskPool;

class LoadTaskRunner {
public:
    LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                   const BlockTransferDispatcher*  transfer_dispatcher,
                   BlockTreeTaskPool&              task_pool,
                   BlockTreeCacheMetricsReporter&  metrics_reporter);

    bool submit(std::shared_ptr<LoadTask> task) const;

private:
    void run(std::shared_ptr<LoadTask> task, std::function<void()> callback) const;

    const std::vector<GroupSetPtr>& group_sets_;
    const BlockTransferDispatcher*  transfer_dispatcher_;
    BlockTreeTaskPool&              task_pool_;
    BlockTreeCacheMetricsReporter&  metrics_reporter_;
};

}  // namespace rtp_llm
