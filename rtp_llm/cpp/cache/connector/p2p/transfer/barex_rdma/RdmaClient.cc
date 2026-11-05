#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaHostConnectionPool.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"

#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"

#include "aios/network/accl-barex/metric/KMonitorAcclMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include <unistd.h>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

// 连接回调类，用于处理连接建立成功/失败
class RdmaClientConnectionCallback: public ::accl::barex::XConnectionCallback {
public:
    RdmaClientConnectionCallback(const std::shared_ptr<RdmaConnection>& rdma_connection):
        rdma_connection_(rdma_connection) {}

    void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& conn,
                    const ::accl::barex::XMessage&                     msgs) override {
        RTP_LLM_LOG_ERROR("recv call, should not happen, request from %s", conn->GetPeerAddrAndPort().first.c_str());
    }

    void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& conn) override {
        if (auto rdma_conn = rdma_connection_.lock()) {
            rdma_conn->onConnectSuccess(conn);
        } else {
            RTP_LLM_LOG_ERROR("rdma connection callback on connect success, but rdma connection is nullptr");
        }
    }

    void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& conn) override {
        if (auto rdma_conn = rdma_connection_.lock()) {
            rdma_conn->onConnectFailed(conn);
        } else {
            RTP_LLM_LOG_ERROR("rdma connection callback on connect failed, but rdma connection is nullptr");
        }
    }

private:
    std::weak_ptr<RdmaConnection> rdma_connection_;
};

RdmaClient::RdmaClient(const std::shared_ptr<IRdmaMemoryManager>&        memory_manager,
                       uint32_t                                          rdma_connections_per_host,
                       const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter):
    rdma_connections_per_host_(rdma_connections_per_host),
    metrics_reporter_(metrics_reporter),
    memory_manager_(memory_manager) {}

RdmaClient::~RdmaClient() {
    // 清理所有连接池
    {
        std::unique_lock<std::shared_mutex> lock(host_connections_mutex_);
        host_connection_pools_.clear();
    }
    // RdmaMempoolFactory 的 Shutdown 会在全局析构时调用，这里不需要手动调用
}

bool RdmaClient::init(int io_thread_count, int worker_thread_count) {
    if (initialized_) {
        RTP_LLM_LOG_WARNING("rdma client already initialized");
        return true;
    }

    if (!memory_manager_) {
        RTP_LLM_LOG_ERROR("rdma client init failed, memory manager is nullptr");
        return false;
    }

    auto concrete_mm = std::dynamic_pointer_cast<RdmaMemoryManager>(memory_manager_);
    if (!concrete_mm) {
        RTP_LLM_LOG_ERROR("rdma client init failed, memory manager is not RdmaMemoryManager");
        return false;
    }

    auto xmempool = concrete_mm->getXMempool();
    if (!xmempool) {
        RTP_LLM_LOG_ERROR("rdma client init failed, get xmempool failed");
        return false;
    }

    // 创建 metrics reporter
    auto metrics_reporter = std::make_shared<::accl::barex::KMonitorAcclMetricReporter>(true, kmonitor::FATAL);
    if (!metrics_reporter->init()) {
        RTP_LLM_LOG_ERROR("rdma client init failed, metrics reporter init failed");
        return false;
    }

    // 创建 XClientTransport
    ::accl::barex::XClientTransportConfig config(io_thread_count, worker_thread_count, xmempool, metrics_reporter);
    client_ = ::accl::barex::XClientTransport::NewInstance(config);
    if (!client_) {
        RTP_LLM_LOG_ERROR("rdma client init failed, accl client init failed");
        return false;
    }

    initialized_ = true;
    RTP_LLM_LOG_INFO(
        "rdma client init success, io thread count %d, worker thread count %d",
        io_thread_count,
        worker_thread_count);
    return true;
}

std::shared_ptr<IRdmaConnection> RdmaClient::getConnection(const std::string& ip, uint32_t port) {
    if (!initialized_) {
        RTP_LLM_LOG_ERROR("rdma client not initialized");
        return nullptr;
    }

    auto pool           = getOrCreateConnectionPool(ip, port);
    auto [conn, idx]    = pool->getNextConnection();
    if (conn) {
        return conn;
    }

    auto new_conn = createNewConnection(ip, port);
    if (!new_conn) {
        return nullptr;
    }
    return pool->installOrGetExistingAt(idx, new_conn);
}

std::shared_ptr<RdmaConnection> RdmaClient::createNewConnection(const std::string& ip, uint32_t port) {
    if (!memory_manager_) {
        RTP_LLM_LOG_WARNING("rdma client create connection failed, memory manager is nullptr");
        return nullptr;
    }

    auto rdma_connection = std::make_shared<RdmaConnection>(memory_manager_, ip, port, metrics_reporter_);

    auto callback = std::make_shared<RdmaClientConnectionCallback>(rdma_connection);

    client_->Connect(::accl::barex::TCP_V4, ip, port, callback);

    return rdma_connection;
}

std::shared_ptr<RdmaHostConnectionPool> RdmaClient::getOrCreateConnectionPool(const std::string& ip, uint32_t port) {
    std::string spec = ip + ":" + std::to_string(port);

    {
        std::shared_lock<std::shared_mutex> lock(host_connections_mutex_);
        auto                                it = host_connection_pools_.find(spec);
        if (it != host_connection_pools_.end()) {
            return it->second;
        }
    }

    std::unique_lock<std::shared_mutex> lock(host_connections_mutex_);
    auto                                it = host_connection_pools_.find(spec);
    if (it != host_connection_pools_.end()) {
        return it->second;
    }

    auto pool = std::make_shared<RdmaHostConnectionPool>(ip, port, rdma_connections_per_host_);
    host_connection_pools_.insert(std::make_pair(spec, pool));
    RTP_LLM_LOG_INFO("rdma client created connection pool to %s:%u, pool size: %u",
                     ip.c_str(), port, rdma_connections_per_host_);
    return pool;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
