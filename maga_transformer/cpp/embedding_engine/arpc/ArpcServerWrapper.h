#pragma once

#include "aios/network/arpc/arpc/ANetRPCServer.h"

namespace rtp_llm {
class ArpcServerWrapper {
public:
    ArpcServerWrapper(std::unique_ptr<::google::protobuf::Service> service, int port): service_(std::move(service)), port_(port) {}
    void start();
    void stop();
private:
    std::unique_ptr<::google::protobuf::Service> service_;
    int port_;
    std::unique_ptr<arpc::ANetRPCServer> arpc_server_;
    std::unique_ptr<anet::Transport> arpc_server_transport_;
};

} // namespace rtp_llm