#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaTransferConnection.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaTransferConnection::RdmaTransferConnection(const std::shared_ptr<RdmaConnection>& rdma_connection):
    rdma_connection_(rdma_connection) {}

void RdmaTransferConnection::read(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                                  const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
                                  TransferConnection::ReadDoneCallback                 callback,
                                  uint32_t                                             timeout_ms) {
    if (local_blocks.size() != remote_blocks.size()) {
        RTP_LLM_LOG_WARNING("cache store service read block failed, local blocks size is %d, remote blocks size is %d",
                            local_blocks.size(),
                            remote_blocks.size());
        callback(false, CacheStoreErrorCode::LoadRdmaWriteFailed, local_blocks);
        return;
    }

    std::vector<std::shared_ptr<RdmaConnectionWriteBlock>> write_blocks;
    for (int i = 0; i < local_blocks.size(); i++) {
        auto& local_block  = local_blocks[i];
        auto& remote_block = remote_blocks[i];
        auto  write_block  = std::make_shared<RdmaConnectionWriteBlock>(
            local_block, remote_block, [local_block, remote_block, callback](bool success) {
                callback(success,
                         success ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadRdmaWriteFailed,
                         {local_block});
            });
        write_blocks.push_back(write_block);
    }
    rdma_connection_->writeBlocks(write_blocks, false);
}

}  // namespace rtp_llm