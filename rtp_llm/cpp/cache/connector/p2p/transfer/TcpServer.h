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
    /// @brief 初始化 transport 和 RPC server（未启动监听）
    bool init(uint32_t io_thread_count,
              uint32_t worker_thread_count,
              uint32_t listen_port,
              bool     enable_metric,
              uint32_t anet_rpc_thread_num = 3,
              uint32_t anet_rpc_queue_num  = 100);
    /// @brief 注册 RPC 服务（须在 start() 之前调用）
    bool registerService(RPCService* rpc_service);
    /// @brief 启动监听并开始处理请求
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
    bool                                   rpc_worker_pool_started_{false};
    std::unique_ptr<anet::Transport>       rpc_server_transport_;
    std::shared_ptr<arpc::ANetRPCServer>   rpc_server_;
    std::shared_ptr<autil::ThreadPoolBase> rpc_worker_threadpool_;
};

}  // namespace transfer
}  // namespace rtp_llm
