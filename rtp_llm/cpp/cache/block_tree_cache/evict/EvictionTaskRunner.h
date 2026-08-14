#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class EvictionTaskRunner {
public:
    EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                       const BlockTransferDispatcher*  transfer_dispatcher,
                       int                             memory_timeout_ms,
                       int                             disk_timeout_ms);

    EvictionTaskResult runTransfer(const EvictionTask& task) const;

private:
    EvictionTaskResult runPerRankTransfer(const EvictionTask& task) const;
    static bool        buildTransferDescriptors(const EvictionTask& task, std::vector<TransferDescriptor>& descriptors);
    std::vector<std::vector<TransferDescriptor>>
               partitionTransferDescriptors(const std::vector<TransferDescriptor>& descriptors) const;
    static int selectTransferTimeoutMs(const EvictionTask& task, int memory_timeout_ms, int disk_timeout_ms);

    const std::vector<GroupSetPtr>& group_sets_;
    const BlockTransferDispatcher*  transfer_dispatcher_{nullptr};
    int                            memory_timeout_ms_{0};
    int                             disk_timeout_ms_{0};
};

}  // namespace rtp_llm
