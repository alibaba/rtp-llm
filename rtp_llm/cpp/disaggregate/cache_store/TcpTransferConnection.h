#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/TransferConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "aios/network/arpc/arpc/ANetRPCChannel.h"

namespace rtp_llm {

class TcpTransferConnection: public TransferConnection {
public:
    TcpTransferConnection(const std::shared_ptr<arpc::RPCChannelBase>& channel, int device_id = -1);

public:
    void read(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
              const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
              TransferConnection::ReadDoneCallback                 callback,
              uint32_t                                             timeout_ms) override;

private:
    std::shared_ptr<arpc::RPCChannelBase> channel_;
    int                                   device_id_{-1};
};

}  // namespace rtp_llm
