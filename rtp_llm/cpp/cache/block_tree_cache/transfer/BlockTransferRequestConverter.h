#pragma once

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BlockTransferRequestConverter {
public:
    static bool encodeTransfer(MemoryOperationRequestPB&       request,
                               const TransferTask&             task,
                               const std::vector<GroupSetPtr>& group_sets);

    static bool decodeTransfer(const MemoryOperationRequestPB&  request,
                               std::vector<TransferDescriptor>& descriptors,
                               const std::vector<GroupSetPtr>&  group_sets);

    static MemoryOperationResponsePB::CopyError encodeCopyError(const ErrorInfo&                       status,
                                                                const std::vector<TransferDescriptor>& descriptors);
    static CacheCopyError                       decodeCopyError(MemoryOperationResponsePB::CopyError error);

private:
    using CopyItem = MemoryOperationRequestPB::CopyItem;

    static bool directionFor(const TransferDescriptor&                descriptor,
                             MemoryOperationRequestPB::CopyDirection& request_direction);
    static bool decodeDeviceBlocks(const CopyItem& item, const GroupSet& group_set, std::vector<BlockIdxType>& blocks);
};

}  // namespace rtp_llm
