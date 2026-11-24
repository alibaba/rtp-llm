#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTree;
class EvictionTaskRunner {
public:
    using EvictionDoneCallback = std::function<void(ErrorInfo)>;

    EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                       const BlockTransferDispatcher*  transfer_dispatcher,
                       BlockTree*                      pressure_tree = nullptr);

    void runTransfer(std::shared_ptr<const EvictionTransferTask> task,
                     BlockTreeCacheMetricsReporter&              metrics_reporter,
                     EvictionDoneCallback                        on_done) const;

private:
    BlockTree*                      pressure_tree_{nullptr};
    const std::vector<GroupSetPtr>& group_sets_;
    const BlockTransferDispatcher*  transfer_dispatcher_{nullptr};
};

}  // namespace rtp_llm
