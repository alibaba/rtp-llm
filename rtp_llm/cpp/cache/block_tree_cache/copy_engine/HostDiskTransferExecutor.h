#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngineLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

enum class BlockIOStatus;

// Internal (non-public API) executor for Host <-> Disk transfers. Only requires host/disk
// pools; never touches the device-host layout. Does block-level DiskBlockPool read/write and
// maps BlockIOStatus to CopyStatus.
class HostDiskTransferExecutor {
public:
    // Synchronous execution entry. Direction (H2Disk vs Disk2H) is taken from desc's tier pair.
    CopyStatus execute(const TransferDescriptor& desc, const ResolvedGroupLayout& layout) const;

private:
    CopyStatus hostToDisk(const TransferDescriptor& desc, const ResolvedGroupLayout& layout) const;
    CopyStatus diskToHost(const TransferDescriptor& desc, const ResolvedGroupLayout& layout) const;

    static CopyStatus  blockIOStatusToCopyStatus(BlockIOStatus status);
    static const char* blockIOStatusName(BlockIOStatus status);
};

}  // namespace rtp_llm
