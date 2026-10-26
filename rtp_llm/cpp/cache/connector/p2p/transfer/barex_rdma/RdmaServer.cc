#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"

#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/metric/KMonitorAcclMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

// 连接回调类，用于处理服务器端连接
// 服务器端连接建立后，客户端可以通过 RDMA read 读取数据
class RdmaServerConnectionCallback: public ::accl::barex::XConnectionCallback {
public:
    RdmaServerConnectionCallback() {}

    void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& conn,
                    const ::accl::barex::XMessage&                     msgs) override {
        // 服务器端通常不需要处理 RPC 调用，RDMA read 是直接的内存访问
        RTP_LLM_LOG_INFO("rdma server recv call from %s", conn->GetPeerAddrAndPort().first.c_str());
    }

    void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& conn) override {
        RTP_LLM_LOG_INFO("rdma server connection success from %p %s:%u",
                         conn.get(),
                         conn->GetPeerAddrAndPort().first.c_str(),
                         conn->GetPeerAddrAndPort().second);
    }

    void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& conn) override {
        RTP_LLM_LOG_WARNING("rdma server connection failed from %p %s:%u",
                            conn.get(),
                            conn->GetPeerAddrAndPort().first.c_str(),
                            conn->GetPeerAddrAndPort().second);
    }
};

RdmaServer::RdmaServer(const std::shared_ptr<IRdmaMemoryManager>& memory_manager):
    memory_manager_(memory_manager) {}

RdmaServer::~RdmaServer() {
    // RdmaMempoolFactory 的 Shutdown 会在全局析构时调用，这里不需要手动调用
}

bool RdmaServer::init(uint32_t listen_port, int io_thread_count, int worker_thread_count) {
    if (initialized_) {
        RTP_LLM_LOG_WARNING("rdma server already initialized");
        return true;
    }

    if (!memory_manager_) {
        RTP_LLM_LOG_ERROR("rdma server init failed, memory manager is nullptr");
        return false;
    }

    auto concrete_mm = std::dynamic_pointer_cast<RdmaMemoryManager>(memory_manager_);
    if (!concrete_mm) {
        RTP_LLM_LOG_ERROR("rdma server init failed, memory manager is not RdmaMemoryManager");
        return false;
    }
    auto xmempool = concrete_mm->getXMempool();
    if (!xmempool) {
        RTP_LLM_LOG_ERROR("rdma server init failed, get xmempool failed");
        return false;
    }

    // 创建 metrics reporter
    auto metrics_reporter = std::make_shared<::accl::barex::KMonitorAcclMetricReporter>(true, kmonitor::FATAL);
    if (!metrics_reporter->init()) {
        RTP_LLM_LOG_ERROR("rdma server init failed, metrics reporter init failed");
        return false;
    }

    // 创建 XServerTransport
    ::accl::barex::XServerTransportConfig config(io_thread_count, worker_thread_count, xmempool, metrics_reporter);
    server_ = ::accl::barex::XServerTransport::NewInstance(config);
    if (!server_) {
        RTP_LLM_LOG_ERROR("rdma server init failed, accl server init failed");
        return false;
    }

    // 创建连接回调
    auto callback = std::make_shared<RdmaServerConnectionCallback>();

    // 监听端口
    if (server_->Listen(::accl::barex::TCP_V4, listen_port, callback) != ::accl::barex::BAREX_SUCCESS) {
        RTP_LLM_LOG_ERROR("rdma server listen on port %u failed", listen_port);
        return false;
    }

    listen_port_ = listen_port;
    initialized_ = true;
    RTP_LLM_LOG_INFO("rdma server init success, listen port %u, io thread count %d, worker thread count %d",
                     listen_port,
                     io_thread_count,
                     worker_thread_count);
    return true;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
