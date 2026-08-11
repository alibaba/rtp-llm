#pragma once

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

enum class BlockIOStatus;

class HostDiskTransferExecutor {
public:
    TransferStatus execute(HostBufferView host, const TransferDescriptor& desc, const GroupSet& group_set) const;

    TransferStatus hostToDisk(const std::vector<HostBufferView>&     hosts,
                              const std::vector<TransferDescriptor>& descriptors,
                              const std::vector<const GroupSet*>&    group_sets) const;
    TransferStatus diskToHost(const std::vector<TransferDescriptor>& descriptors,
                              const std::vector<const GroupSet*>&    group_sets,
                              const std::vector<HostBufferView>&     hosts) const;

private:
    static TransferStatus blockIOStatusToTransferStatus(BlockIOStatus status);
    static const char*    blockIOStatusName(BlockIOStatus status);
};

}  // namespace rtp_llm
