#pragma once

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCServer.h"

namespace rtp_llm {

class TcpServer {
public:
    TcpServer() = default;
    ~TcpServer();

public:
    bool init(uint32_t io_thread_count, uint32_t worker_thread_count, bool enable_metric);
    bool registerService(RPCService* rpc_service);
    bool start(uint32_t listen_port);

private:
    void stop();

private:
    std::unique_ptr<anet::Transport>       rpc_server_transport_;
    std::shared_ptr<arpc::ANetRPCServer>   rpc_server_;
    std::shared_ptr<autil::ThreadPoolBase> rpc_worker_threadpool_;
};

}  // namespace rtp_llm