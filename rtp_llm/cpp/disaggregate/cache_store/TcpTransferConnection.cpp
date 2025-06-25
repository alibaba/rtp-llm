#include "rtp_llm/cpp/disaggregate/cache_store/TcpTransferConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpBlockReadClosure.h"

namespace rtp_llm {

TcpTransferConnection::TcpTransferConnection(const std::shared_ptr<arpc::RPCChannelBase>& channel): channel_(channel) {}

void TcpTransferConnection::read(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                                 const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
                                 TransferConnection::ReadDoneCallback                 callback,
                                 uint32_t                                             timeout_ms) {
    auto request = new BlockReadRequest;
    for (auto& block : remote_blocks) {
        auto new_block = request->add_blocks();
        new_block->CopyFrom(*block);
    }

    auto                     response   = new BlockReadResponse;
    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(timeout_ms);
    auto closure = new TcpBlockReadClosure(local_blocks, remote_blocks, callback, request, response, controller);

    KvCacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel_.get()),
                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.blockRead(controller, request, response, closure);
}
}  // namespace rtp_llm