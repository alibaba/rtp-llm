#pragma once

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class BlockTreeTransferConverter {
public:
    static bool appendTransfer(const TransferDescriptor& descriptor, MemoryOperationRequestPB& request);

    static bool decodeTransfer(const MemoryOperationRequestPB& request, int item_index, TransferDescriptor& descriptor);

private:
    using CopyItem = MemoryOperationRequestPB::CopyItem;

    static bool validBlock(BlockIdxType block);
    static bool validDeviceBlocks(const std::vector<BlockIdxType>& blocks);
    static bool hasSourceMemory(const CopyItem& item);
    static bool hasSourceDisk(const CopyItem& item);
    static bool hasTargetDisk(const CopyItem& item);
    static bool validateCommonItem(const CopyItem& item);
    static bool directionFor(const TransferDescriptor&                descriptor,
                             MemoryOperationRequestPB::CopyDirection& request_direction);
    static void setDeviceBlocks(const std::vector<BlockIdxType>& blocks, CopyItem& item);
    static bool decodeDeviceHostTransfer(const MemoryOperationRequestPB& request,
                                         const CopyItem&                 item,
                                         TransferDescriptor&             descriptor);
    static bool decodeHostDiskTransfer(const MemoryOperationRequestPB& request,
                                       const CopyItem&                 item,
                                       TransferDescriptor&             descriptor);
};

}  // namespace rtp_llm
