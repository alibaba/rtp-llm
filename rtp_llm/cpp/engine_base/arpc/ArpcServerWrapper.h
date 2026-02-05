#pragma once

#include "aios/network/arpc/arpc/ANetRPCServer.h"

namespace rtp_llm {
class ArpcServerWrapper {
public:
    ArpcServerWrapper(
        std::unique_ptr<::google::protobuf::Service> service, int threadNum, int queueNum, int ioThreadNum, int port):
        service_(std::move(service)),
        port_(port),
        threadNum_(threadNum),
        queueNum_(queueNum),
        ioThreadNum_(ioThreadNum) {}
    void start();
    void stop();

private:
    std::unique_ptr<::google::protobuf::Service> service_;
    int                                          port_;
    int                                          threadNum_;
    int                                          queueNum_;
    int                                          ioThreadNum_;
    std::unique_ptr<arpc::ANetRPCServer>         arpc_server_;
    std::unique_ptr<anet::Transport>             arpc_server_transport_;
};

}  // namespace rtp_llm