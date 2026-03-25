#pragma once

#include "rtp_llm/cpp/embedding_engine/arpc/ArpcServerWrapper.h"
#include "aios/network/arpc/arpc/ANetRPCServer.h"

namespace rtp_llm {
class AnetArpcServerWrapper: public ArpcServerWrapper {
public:
    AnetArpcServerWrapper(
        std::unique_ptr<::google::protobuf::Service> service, int threadNum, int queueNum, int ioThreadNum, int port):
        ArpcServerWrapper(std::move(service), threadNum, queueNum, ioThreadNum, port) {}
    virtual void start() override;
    virtual void stop() override;

private:
    std::unique_ptr<arpc::ANetRPCServer> arpc_server_;
    std::unique_ptr<anet::Transport>     arpc_server_transport_;
};

}  // namespace rtp_llm
