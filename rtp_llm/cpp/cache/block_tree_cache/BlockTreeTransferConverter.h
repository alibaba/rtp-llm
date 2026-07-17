#pragma once

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class BlockTreeTransferConverter {
public:
    static bool appendTransfer(const TransferDescriptor&             descriptor,
                               const std::vector<ComponentGroupPtr>& component_groups,
                               MemoryOperationRequestPB&             request);

    static bool decodeTransfer(const MemoryOperationRequestPB&       request,
                               int                                   item_index,
                               const std::vector<ComponentGroupPtr>& component_groups,
                               TransferDescriptor&                   descriptor);

private:
    using CopyItem = MemoryOperationRequestPB::CopyItem;

    static bool                  hasSourceMemory(const CopyItem& item);
    static bool                  hasSourceDisk(const CopyItem& item);
    static bool                  hasTargetDisk(const CopyItem& item);
    static bool                  validateCommonItem(const CopyItem& item);
    static const ComponentGroup* findComponentGroup(int                                   component_group_id,
                                                    const std::vector<ComponentGroupPtr>& component_groups);
    static bool validDeviceBlocks(const std::vector<BlockIdxType>& blocks, const ComponentGroup& component_group);
    static bool validHostBlock(BlockIdxType block, const ComponentGroup& component_group);
    static bool validDiskBlock(BlockIdxType block, const ComponentGroup& component_group);
    static bool directionFor(const TransferDescriptor&                descriptor,
                             const ComponentGroup&                    component_group,
                             MemoryOperationRequestPB::CopyDirection& request_direction);
    static void setDeviceBlocks(const std::vector<BlockIdxType>& blocks, CopyItem& item);
    static bool decodeDeviceHostTransfer(const MemoryOperationRequestPB& request,
                                         const CopyItem&                 item,
                                         const ComponentGroup&           component_group,
                                         TransferDescriptor&             descriptor);
    static bool decodeHostDiskTransfer(const MemoryOperationRequestPB& request,
                                       const CopyItem&                 item,
                                       const ComponentGroup&           component_group,
                                       TransferDescriptor&             descriptor);
};

}  // namespace rtp_llm
