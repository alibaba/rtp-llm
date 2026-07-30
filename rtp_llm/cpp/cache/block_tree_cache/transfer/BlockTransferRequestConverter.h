#pragma once

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class BlockTransferRequestConverter {
public:
    static bool appendTransfer(const TransferDescriptor&       descriptor,
                               const std::vector<GroupSetPtr>& group_sets,
                               MemoryOperationRequestPB&       request);

    // Returns an invalid descriptor (isValid() == false) when the request cannot be decoded.
    static TransferDescriptor
    decodeTransfer(const MemoryOperationRequestPB& request, int item_index, const std::vector<GroupSetPtr>& group_sets);

private:
    using CopyItem = MemoryOperationRequestPB::CopyItem;

    static bool            hasSourceMemory(const CopyItem& item);
    static bool            hasSourceDisk(const CopyItem& item);
    static bool            hasTargetDisk(const CopyItem& item);
    static const GroupSet* findGroupSet(const CopyItem& item, const std::vector<GroupSetPtr>& group_sets);
    static bool            validDeviceBlocks(const std::vector<BlockIdxType>& blocks, const GroupSet& group_set);
    static bool            validHostBlock(BlockIdxType block, const GroupSet& group_set);
    static bool            validDiskBlock(BlockIdxType block, const GroupSet& group_set);
    static bool            directionFor(const TransferDescriptor&                descriptor,
                                        const GroupSet&                          group_set,
                                        MemoryOperationRequestPB::CopyDirection& request_direction);
    static void setDeviceBlocks(const std::vector<BlockIdxType>& blocks, const GroupSet& group_set, CopyItem& item);
    static bool decodeDeviceBlocks(const CopyItem& item, const GroupSet& group_set, std::vector<BlockIdxType>& blocks);
    static TransferDescriptor
    decodeDeviceHostTransfer(const MemoryOperationRequestPB& request, const CopyItem& item, const GroupSet& group_set);
    static TransferDescriptor
    decodeHostDiskTransfer(const MemoryOperationRequestPB& request, const CopyItem& item, const GroupSet& group_set);
    static TransferDescriptor
    decodeDeviceDiskTransfer(const MemoryOperationRequestPB& request, const CopyItem& item, const GroupSet& group_set);
};

}  // namespace rtp_llm
