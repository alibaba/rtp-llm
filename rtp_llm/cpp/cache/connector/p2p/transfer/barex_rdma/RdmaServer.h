#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xserver_transport.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include <memory>
#include <string>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief RDMA 服务端，监听指定端口接受 Decode 侧的 RDMA Read 请求
class RdmaServer: public IRdmaServer {
public:
    explicit RdmaServer(const std::shared_ptr<IRdmaMemoryManager>& memory_manager);
    ~RdmaServer();

public:
    /// @brief 初始化 XServerTransport 并开始监听指定端口
    bool init(uint32_t listen_port, int io_thread_count, int worker_thread_count);

    uint32_t getListenPort() const override {
        return listen_port_;
    }

private:
    std::shared_ptr<::accl::barex::XServerTransport> server_;
    std::shared_ptr<IRdmaMemoryManager>              memory_manager_;
    uint32_t                                         listen_port_    = 0;
    bool                                             initialized_    = false;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
