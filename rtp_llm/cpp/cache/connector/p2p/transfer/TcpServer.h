#pragma once

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCServer.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

namespace transfer {
class TcpServer {
public:
    TcpServer() = default;
    ~TcpServer();

public:
    bool init(uint32_t io_thread_count, uint32_t worker_thread_count, uint32_t listen_port, bool enable_metric);
    bool registerService(RPCService* rpc_service);
    bool start();

    std::string getIP() const {
        return autil::NetUtil::getBindIp();
    }
    uint32_t getPort() const {
        return listen_port_;
    }

private:
    void stop();

private:
    uint32_t                               listen_port_;
    std::unique_ptr<anet::Transport>       rpc_server_transport_;
    std::shared_ptr<arpc::ANetRPCServer>   rpc_server_;
    std::shared_ptr<autil::ThreadPoolBase> rpc_worker_threadpool_;
};

}  // namespace transfer
}  // namespace rtp_llm
