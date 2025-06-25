#pragma once

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TransferConnection.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"

namespace rtp_llm {

class TcpBlockReadClosure: public RPCClosure {
public:
    TcpBlockReadClosure(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                        const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
                        TransferConnection::ReadDoneCallback                 callback,
                        BlockReadRequest*                                    request,
                        BlockReadResponse*                                   response,
                        arpc::ANetRPCController*                             controller);
    ~TcpBlockReadClosure();

public:
    void Run() override;

private:
    void end(bool success, CacheStoreErrorCode error_code);

private:
    std::vector<std::shared_ptr<BlockBuffer>>     local_blocks_;
    std::vector<std::shared_ptr<BlockBufferInfo>> remote_blocks_;
    TransferConnection::ReadDoneCallback          callback_;

    BlockReadRequest*        request_;
    BlockReadResponse*       response_;
    arpc::ANetRPCController* controller_;
    rtp_llm::DeviceBase*     device_{nullptr};
};
}  // namespace rtp_llm