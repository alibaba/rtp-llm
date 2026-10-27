#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"

#include "aios/network/accl-barex/include/accl/barex/xchannel.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <algorithm>
#include <thread>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

namespace {
constexpr int kMaxFlushAttempts = 3;
}

RdmaConnection::RdmaConnection(const std::shared_ptr<IRdmaMemoryManager>&        memory_manager,
                               const std::string&                                ip,
                               uint32_t                                          port,
                               const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter):
    memory_manager_(memory_manager),
    ip_(ip),
    port_(port),
    metrics_reporter_(metrics_reporter) {}

RdmaConnection::~RdmaConnection() {
    auto pending = drainPendingRequests();
    failRequests(pending);
}

RdmaConnectionState RdmaConnection::getState() const {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    return state_;
}

void RdmaConnection::onConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    std::shared_ptr<::accl::barex::XConnection> conn_snap;
    {
        std::lock_guard<std::mutex> conn_lock(conn_mutex_);
        if (state_ != RdmaConnectionState::CONNECTING) {
            RTP_LLM_LOG_WARNING("rdma connection to %s:%u onConnectSuccess but state is not CONNECTING",
                                ip_.c_str(), port_);
            return;
        }
        connection_ = connection;
        state_      = RdmaConnectionState::CONNECTED;
        conn_snap   = connection_;
    }

    auto to_dispatch = drainPendingRequests();
    RTP_LLM_LOG_INFO("rdma connection to %s:%u connect success", ip_.c_str(), port_);
    dispatchRequests(to_dispatch, conn_snap);
}

void RdmaConnection::onConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& /*connection*/) {
    {
        std::lock_guard<std::mutex> conn_lock(conn_mutex_);
        state_      = RdmaConnectionState::FAILED;
        connection_ = nullptr;
    }

    auto to_fail = drainPendingRequests();
    RTP_LLM_LOG_WARNING("rdma connection to %s:%u connect failed", ip_.c_str(), port_);
    failRequests(to_fail);
}

void RdmaConnection::read(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
                          std::function<void(bool)>                                                  callback,
                          uint64_t                                                                   deadline_ms) {
    auto collector           = std::make_shared<RdmaMetricsCollector>();
    collector->start_time_us = currentTimeUs();
    collector->block_count   = static_cast<int64_t>(local_remote_buffers.size());
    for (const auto& [local_buffer, remote_buffer] : local_remote_buffers) {
        if (local_buffer.defined()) {
            collector->total_block_size += static_cast<int64_t>(local_buffer.nbytes());
        }
    }

    auto callback2 = [collector, callback, metric_reporter = metrics_reporter_](bool success) {
        if (metric_reporter != nullptr) {
            collector->success = success;
            metric_reporter->report<RdmaMetric, RdmaMetricsCollector>(nullptr, collector.get());
        }
        callback(success);
    };

    enqueuePendingRequest({local_remote_buffers, callback2, collector, deadline_ms});

    // Bounded retries: first attempt covers the common case; later attempts (after yield)
    // align with onConnectSuccess/onConnectFailed that may run on another thread after enqueue.
    for (int attempt = 0; attempt < kMaxFlushAttempts; ++attempt) {
        if (attempt > 0) {
            std::this_thread::yield();
        }
        if (!flushPendingForCurrentStateOnce()) {
            return;
        }
    }
}

bool RdmaConnection::flushPendingForCurrentStateOnce() {
    std::shared_ptr<::accl::barex::XConnection> conn_snap;
    RdmaConnectionState                         cur_state;
    {
        std::lock_guard<std::mutex> conn_lock(conn_mutex_);
        cur_state = state_;
        conn_snap = connection_;
    }

    if (cur_state == RdmaConnectionState::CONNECTED) {
        auto to_dispatch = drainPendingRequests();
        dispatchRequests(to_dispatch, conn_snap);
        return false;
    }
    if (cur_state == RdmaConnectionState::FAILED) {
        auto to_fail = drainPendingRequests();
        failRequests(to_fail);
        return false;
    }
    return true;
}

void RdmaConnection::enqueuePendingRequest(PendingReadRequest request) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_requests_.push(std::move(request));
}

std::vector<RdmaConnection::PendingReadRequest> RdmaConnection::drainPendingRequests() {
    std::lock_guard<std::mutex>     lock(pending_mutex_);
    std::vector<PendingReadRequest> requests;
    while (!pending_requests_.empty()) {
        requests.push_back(std::move(pending_requests_.front()));
        pending_requests_.pop();
    }
    return requests;
}

void RdmaConnection::dispatchRequests(std::vector<PendingReadRequest>&             requests,
                                      const std::shared_ptr<::accl::barex::XConnection>& conn) {
    for (auto& req : requests) {
        sendReadRequest(req.local_remote_buffers, req.collector, req.callback, req.deadline_ms, conn);
    }
}

void RdmaConnection::failRequests(std::vector<PendingReadRequest>& requests) {
    for (auto& req : requests) {
        req.callback(false);
    }
}

bool RdmaConnection::validateAndBuildRwMemps(
    const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
    std::vector<::accl::barex::rw_memp_t>&                                      mems,
    uint64_t                                                                    deadline_ms,
    const std::shared_ptr<::accl::barex::XConnection>&                          conn) {

    mems.clear();

    auto concrete_mm = std::dynamic_pointer_cast<RdmaMemoryManager>(memory_manager_);
    if (!concrete_mm) {
        RTP_LLM_LOG_ERROR("rdma connection read failed, memory manager is not RdmaMemoryManager");
        return false;
    }

    for (const auto& [local_buffer, remote_buffer] : local_remote_buffers) {
        if (!local_buffer.defined()) {
            RTP_LLM_LOG_WARNING("rdma connection read failed, local buffer is not defined");
            return false;
        }

        if (!remote_buffer) {
            RTP_LLM_LOG_WARNING("rdma connection read failed, remote buffer is nullptr");
            return false;
        }

        void*  data_ptr   = local_buffer.data_ptr();
        bool   is_gpu     = local_buffer.is_cuda();
        size_t size_bytes = local_buffer.nbytes();

        if (size_bytes != remote_buffer->len) {
            RTP_LLM_LOG_WARNING(
                "rdma connection read failed, size mismatch: local %lu, remote %u", size_bytes, remote_buffer->len);
            return false;
        }

        ::accl::barex::memp_t mem;
        if (!concrete_mm->findMemoryMr(mem, data_ptr, size_bytes, is_gpu)) {
            RTP_LLM_LOG_WARNING("rdma connection read failed, find buffer mr failed");
            return false;
        }

        uint64_t r_addr = static_cast<uint64_t>(remote_buffer->addr);
        uint32_t nic_id = conn->GetChannel()->GetPeerNicId();
        auto     it     = remote_buffer->nic_rkeys->find(nic_id);
        if (it == remote_buffer->nic_rkeys->end()) {
            RTP_LLM_LOG_WARNING("rdma connection read failed, not find rkey for nic %u", nic_id);
            return false;
        }

        ::accl::barex::rw_memp_t rw_mem;
        rw_mem.data     = mem;
        rw_mem.r_addr   = r_addr;
        rw_mem.r_key    = it->second;
        rw_mem.r_ttl_ms = deadline_ms;
        mems.push_back(rw_mem);
    }

    return true;
}

void RdmaConnection::sendReadRequest(
    const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
    const std::shared_ptr<RdmaMetricsCollector>&                               collector,
    std::function<void(bool)>                                                  callback,
    uint64_t                                                                   deadline_ms,
    const std::shared_ptr<::accl::barex::XConnection>&                         conn) {
    collector->pending_done_time_us = currentTimeUs();

    if (!conn) {
        callback(false);
        return;
    }

    std::vector<::accl::barex::rw_memp_t> mems;
    if (!validateAndBuildRwMemps(local_remote_buffers, mems, deadline_ms, conn)) {
        callback(false);
        return;
    }

    auto rdma_callback = [callback, collector](::accl::barex::Status status) {
        collector->write_done_time_us = currentTimeUs();
        if (!status.IsOk()) {
            RTP_LLM_LOG_WARNING("rdma connection read batch failed, err is %s", status.ErrMsg().c_str());
            callback(false);
            return;
        }
        callback(true);
    };

    conn->GetChannel()->ReadBatch(mems, rdma_callback, false);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
