#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaHostConnectionPool.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaHostConnectionPool::RdmaHostConnectionPool(const std::string& ip, uint32_t port, uint32_t pool_size):
    ip_(ip), port_(port), pool_size_(pool_size), connections_(pool_size) {}

ConnectionSlot RdmaHostConnectionPool::getNextConnection() {
    size_t idx = round_robin_idx_.fetch_add(1, std::memory_order_relaxed) % pool_size_;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& conn = connections_[idx];
    if (conn && conn->getState() != RdmaConnectionState::FAILED) {
        return {conn, idx};
    }
    return {nullptr, idx};
}

std::shared_ptr<RdmaConnection> RdmaHostConnectionPool::installOrGetExistingAt(
    size_t idx, const std::shared_ptr<RdmaConnection>& new_conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (idx >= connections_.size()) {
        return nullptr;
    }
    auto& slot = connections_[idx];
    if (slot && slot->getState() != RdmaConnectionState::FAILED) {
        return slot;
    }
    slot = new_conn;
    RTP_LLM_LOG_DEBUG("rdma host connection pool set connection[%zu] to %s:%u", idx, ip_.c_str(), port_);
    return new_conn;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
