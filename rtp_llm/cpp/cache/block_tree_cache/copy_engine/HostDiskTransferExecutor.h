#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

enum class BlockIOStatus;

// Internal (non-public API) executor for Host <-> Disk transfers. Only requires host/disk
// pools; never touches the device-host layout. Does block-level BlockTreeDiskBlockPool read/write and
// maps BlockIOStatus to CopyStatus.
class HostDiskTransferExecutor {
public:
    CopyStatus hostToDisk(const TransferDescriptor& desc, const ComponentGroup& group) const;
    CopyStatus diskToHost(const TransferDescriptor& desc, const ComponentGroup& group) const;

private:
    static CopyStatus  blockIOStatusToCopyStatus(BlockIOStatus status);
    static const char* blockIOStatusName(BlockIOStatus status);
};

}  // namespace rtp_llm
