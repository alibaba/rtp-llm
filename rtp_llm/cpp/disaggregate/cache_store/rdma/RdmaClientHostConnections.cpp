#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientConnectionCallback.h"

#include "autil/Scope.h"

#include <chrono>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
// Limit simultaneous handshake batches for each peer. Established QPs are usable
// by other callers while a batch is connecting.
constexpr size_t kMaxConnectingGroups = 64;
}  // namespace

RdmaClientHostConnections::RdmaClientHostConnections(const std::shared_ptr<MemoryUtil>& memory_util,
                                                     const std::string&                 host,
                                                     uint32_t                           port,
                                                     uint32_t                           qp_count_per_rdma_connection,
                                                     uint32_t                           connect_timeout_ms):
    memory_util_(memory_util),
    host_(host),
    port_(port),
    qp_count_per_rdma_connection_(qp_count_per_rdma_connection),
    connect_timeout_ms_(connect_timeout_ms) {}

RdmaClientHostConnections::~RdmaClientHostConnections() {
    for (auto& connection : newly_connections_) {
        connection->Close();
    }
    newly_connections_.clear();

    for (auto& connection : available_connections_) {
        connection->Close();
    }
    available_connections_.clear();

    if (busy_connections_.size() > 0) {
        RTP_LLM_LOG_WARNING("still has inprocessing rdma connection while rdma client dtor");
    }
    for (auto& connection : busy_connections_) {
        connection->Close();
    }
    busy_connections_.clear();
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::getConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client) {
    auto rdma_connection = acquire(client, true);
    if (rdma_connection == nullptr) {
        RTP_LLM_LOG_WARNING("new available connections to %s:%u failed", host_.c_str(), port_);
    }
    return rdma_connection;
}

std::shared_ptr<RdmaConnection> RdmaClientHostConnections::takeReadyConnectionsLocked(bool use_recycled, bool use_new) {
    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections;
    auto                                                     take = [&](auto& pool) {
        while (!pool.empty() && connections.size() < qp_count_per_rdma_connection_) {
            auto it = pool.begin();
            connections.push_back(*it);
            busy_connections_.insert(*it);
            pool.erase(it);
        }
    };
    if (use_recycled) {
        take(available_connections_);
    }
    if (use_new) {
        take(newly_connections_);
    }
    if (connections.empty()) {
        return nullptr;
    }
    return std::make_shared<RdmaConnection>(memory_util_, host_, port_, connections);
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::acquire(const std::shared_ptr<::accl::barex::XClientTransport>& client, bool use_recycled) {
    if (qp_count_per_rdma_connection_ == 0) {
        return nullptr;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(connect_timeout_ms_);
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         recycled    = [&] { return use_recycled && !available_connections_.empty(); };
    auto                         can_acquire = [&] { return recycled() || connecting_groups_ < kMaxConnectingGroups; };
    // The last caller can time out before its callbacks arrive. Even one late
    // QP can service a request, as can a partially recycled group.
    if (!pool_changed_.wait_until(lock, deadline, can_acquire)) {
        return takeReadyConnectionsLocked(use_recycled, true);
    }
    // Let a caller holding a connect slot consume its completed QPs and
    // release the slot. Otherwise queued callers can starve every creator.
    if (recycled() || (connecting_groups_ == 0 && !newly_connections_.empty())) {
        return takeReadyConnectionsLocked(use_recycled, connecting_groups_ == 0);
    }

    ++connecting_groups_;
    autil::ScopeGuard release_slot([&] {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        --connecting_groups_;
        lock.unlock();
        pool_changed_.notify_all();
    });
    lock.unlock();  // Connect may invoke its callback synchronously.
    auto callback = std::make_shared<RdmaClientConnectionCallback>(shared_from_this());
    for (uint32_t i = 0; i < qp_count_per_rdma_connection_; ++i) {
        client->Connect(::accl::barex::TCP_V4, host_, port_, callback);
    }
    lock.lock();
    pool_changed_.wait_until(
        lock, deadline, [&] { return recycled() || newly_connections_.size() >= qp_count_per_rdma_connection_; });
    return takeReadyConnectionsLocked(use_recycled, true);
}

void RdmaClientHostConnections::recycleConnection(const std::shared_ptr<RdmaConnection>& connection) {
    if (connection == nullptr) {
        return;
    }
    auto connections = connection->getAllConnections();
    doRecycleConnections(connections);
}

void RdmaClientHostConnections::doRecycleConnections(
    const std::vector<std::shared_ptr<::accl::barex::XConnection>>& connections) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& connection : connections) {
        if (connection == nullptr) {
            continue;
        }
        // connection may already broken and call failed callback
        if (busy_connections_.find(connection) != busy_connections_.end()) {
            busy_connections_.erase(connection);
            available_connections_.insert(connection);
        }
    }
    pool_changed_.notify_all();
}

void RdmaClientHostConnections::destroyConnection(const std::shared_ptr<RdmaConnection>& rdma_connection) {
    if (rdma_connection == nullptr) {
        return;
    }

    auto connections = rdma_connection->getAllConnections();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& connection : connections) {
            if (busy_connections_.find(connection) != busy_connections_.end()) {
                busy_connections_.erase(connection);
                RTP_LLM_LOG_INFO("rdma connection to %s:%u call disconnect", host_.c_str(), port_);
            }
        }
    }

    // trigger close
    for (auto& connection : connections) {
        connection->Close();
    }
}

void RdmaClientHostConnections::connectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection == nullptr) {
        return;
    }
    RTP_LLM_LOG_INFO("rdma connection to %s:%u connect success", host_.c_str(), port_);
    std::lock_guard<std::mutex> lock(mutex_);
    newly_connections_.insert(connection);
    pool_changed_.notify_all();
}

void RdmaClientHostConnections::connectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto iter = newly_connections_.find(connection);
    if (iter != newly_connections_.end()) {
        newly_connections_.erase(iter);
        RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
        return;
    }

    iter = available_connections_.find(connection);
    if (iter != available_connections_.end()) {
        available_connections_.erase(iter);
        RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
        return;
    }

    iter = busy_connections_.find(connection);
    if (iter != busy_connections_.end()) {
        busy_connections_.erase(iter);
        RTP_LLM_LOG_WARNING("rdma connection to %s:%u disconnect while inuse", host_.c_str(), port_);
        return;
    }
    RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
}

int RdmaClientHostConnections::getAvailableConnectionCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_connections_.size();
}

bool RdmaClientHostConnections::fillConnectionPool(const std::shared_ptr<::accl::barex::XClientTransport>& client,
                                                   int                                                     count) {
    for (int i = 0; i < count; i++) {
        // Each iteration must add idle QPs, even when the pool is already warm.
        auto rdma_connection = acquire(client, false);
        if (rdma_connection == nullptr) {
            RTP_LLM_LOG_WARNING("new available connections to %s:%u failed", host_.c_str(), port_);
            return false;
        }
        recycleConnection(rdma_connection);
    }
    RTP_LLM_LOG_INFO(
        "rdma client host connections to %s:%u fill connection pool success, count %d", host_.c_str(), port_, count);
    return true;
}

}  // namespace rtp_llm
