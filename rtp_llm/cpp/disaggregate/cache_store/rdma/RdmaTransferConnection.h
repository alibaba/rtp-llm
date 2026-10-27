#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/TransferConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {

class RdmaTransferConnection: public TransferConnection {
public:
    RdmaTransferConnection(const std::shared_ptr<RdmaConnection>& rdma_connection);

public:
    void read(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
              const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
              TransferConnection::ReadDoneCallback                 callback,
              uint32_t                                             timeout_ms) override;

private:
    std::shared_ptr<RdmaConnection> rdma_connection_;
};

}  // namespace rtp_llm