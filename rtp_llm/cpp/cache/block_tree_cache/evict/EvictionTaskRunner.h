#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class EvictionTaskRunner {
public:
    using EvictionDoneCallback = std::function<void(EvictionTaskResult)>;

    EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                       const BlockTransferDispatcher*  transfer_dispatcher,
                       int                             memory_timeout_ms,
                       int                             disk_timeout_ms);

    void runTransfer(std::shared_ptr<const EvictionTask> task,
                     BlockTreeCacheMetricsReporter&     metrics_reporter,
                     EvictionDoneCallback               on_done) const;

private:
    static bool validateTransferDescriptors(const EvictionTask& task);
    static int selectTransferTimeoutMs(const EvictionTask& task, int memory_timeout_ms, int disk_timeout_ms);

    const std::vector<GroupSetPtr>& group_sets_;
    const BlockTransferDispatcher*  transfer_dispatcher_{nullptr};
    int                            memory_timeout_ms_{0};
    int                             disk_timeout_ms_{0};
};

}  // namespace rtp_llm
