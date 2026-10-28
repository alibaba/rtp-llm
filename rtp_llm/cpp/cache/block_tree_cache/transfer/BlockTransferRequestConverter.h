#pragma once

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BlockTransferRequestConverter {
public:
    static TaskCompletion queryCompletion(const std::string& id,
        const std::vector<QueryTransferTasksResponsePB>& responses, size_t worker_count);
    static bool encodeTransfer(MemoryOperationRequestPB&       request,
                               const TransferTask&             task,
                               const std::vector<GroupSetPtr>& group_sets);

    static bool decodeTransfer(const MemoryOperationRequestPB&  request,
                               std::vector<TransferDescriptor>& descriptors,
                               const std::vector<GroupSetPtr>&  group_sets);

    static MemoryOperationResponsePB::Code encodeCode(CacheCopyError error);
    static CacheCopyError                  decodeCode(MemoryOperationResponsePB::Code error);

    static void           encodeTransferResult(MemoryOperationResponsePB& response, const TransferResult& result);
    static TransferResult decodeTransferResult(const MemoryOperationResponsePB& response, size_t block_count);

private:
    using CopyItem = MemoryOperationRequestPB::CopyItem;

    static bool directionFor(const TransferDescriptor&                descriptor,
                             MemoryOperationRequestPB::CopyDirection& request_direction);
    static bool decodeDeviceBlocks(const CopyItem& item, const GroupSet& group_set, std::vector<BlockIdxType>& blocks);
};

}  // namespace rtp_llm
