#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

enum class BlockIOStatus;

// Consumes a validated HostBufferView; disk I/O always uses the padded stride.
class HostDiskTransferExecutor {
public:
    TransferStatus hostToDisk(HostBufferView host, const TransferDescriptor& desc, const GroupSet& group_set) const;
    TransferStatus diskToHost(const TransferDescriptor& desc, const GroupSet& group_set, HostBufferView host) const;

private:
    static TransferStatus blockIOStatusToTransferStatus(BlockIOStatus status);
    static const char*    blockIOStatusName(BlockIOStatus status);
};

}  // namespace rtp_llm
