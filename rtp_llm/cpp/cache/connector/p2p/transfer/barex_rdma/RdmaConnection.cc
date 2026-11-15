#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"

#include "aios/network/accl-barex/include/accl/barex/xchannel.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <algorithm>
#include <thread>
#include <cstdlib>

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
    bool close_late_connection = false;
    {
        std::lock_guard<std::mutex> conn_lock(conn_mutex_);
        if (state_ != RdmaConnectionState::CONNECTING) {
            RTP_LLM_LOG_WARNING("rdma connection to %s:%u onConnectSuccess but state is not CONNECTING",
                                ip_.c_str(), port_);
            close_late_connection = true;
        } else {
            connection_ = connection;
            state_ = RdmaConnectionState::CONNECTED;
            conn_snap = connection_;
        }
    }
    if (close_late_connection) {
        if (connection) {
            connection->Close();
        }
        return;
    }

    auto to_dispatch = drainPendingRequests();
    RTP_LLM_LOG_INFO("rdma connection to %s:%u connect success", ip_.c_str(), port_);
    dispatchRequests(to_dispatch, conn_snap);
}

void RdmaConnection::close() {
    std::shared_ptr<::accl::barex::XConnection> conn_to_close;
    {
        std::lock_guard<std::mutex> conn_lock(conn_mutex_);
        if (state_ == RdmaConnectionState::FAILED) {
            return;
        }
        state_ = RdmaConnectionState::FAILED;
        conn_to_close.swap(connection_);
    }

    auto to_fail = drainPendingRequests();
    failRequests(to_fail);
    RTP_LLM_LOG_WARNING("closing rdma connection to %s:%u", ip_.c_str(), port_);
    // Close can synchronously invoke onConnectFailed(), so it must run without conn_mutex_.
    if (conn_to_close) {
        conn_to_close->Close();
    }
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

void RdmaConnection::read(
    const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
    RdmaReadCallback callback,
    uint64_t deadline_ms) {
    auto collector           = std::make_shared<RdmaMetricsCollector>();
    collector->start_time_us = currentTimeUs();
    collector->block_count   = static_cast<int64_t>(local_remote_buffers.size());
    for (const auto& [local_buffer, remote_buffer] : local_remote_buffers) {
        if (local_buffer.defined()) {
            collector->total_block_size += static_cast<int64_t>(local_buffer.nbytes());
        }
    }

    auto callback2 = [collector, callback, metric_reporter = metrics_reporter_](TransferErrorCode ec,
                                                                                const std::string& message) {
        const bool success = ec == TransferErrorCode::OK;
        if (metric_reporter != nullptr) {
            collector->success = success;
            metric_reporter->report<RdmaMetric, RdmaMetricsCollector>(nullptr, collector.get());
        }
        callback(ec, message);
    };

    enqueuePendingRequest({local_remote_buffers, callback2, collector, deadline_ms});

    RTP_LLM_LOG_DEBUG("rdma connection %s:%u read: enqueued pending request, "
                      "deadline_ms=%lu, block_count=%zu, state=%d",
                      ip_.c_str(),
                      port_,
                      deadline_ms,
                      local_remote_buffers.size(),
                      static_cast<int>(getState()));

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
    RTP_LLM_LOG_DEBUG("rdma connection %s:%u read: exhausted %d flush attempts, "
                      "request stays in pending queue (state=%d)",
                      ip_.c_str(),
                      port_,
                      kMaxFlushAttempts,
                      static_cast<int>(getState()));
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
        req.callback(TransferErrorCode::CONNECTION_FAILED,
                     readContext() + "connection failed or closed before READ dispatch");
    }
}

void RdmaConnection::sweepExpiredPendingRequests() {
    std::vector<PendingReadRequest> expired;
    size_t remaining_count = 0;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_requests_.empty()) {
            return;
        }
        int64_t now_ms = currentTimeMs();
        std::queue<PendingReadRequest> remaining;
        while (!pending_requests_.empty()) {
            auto& front = pending_requests_.front();
            if (front.deadline_ms > 0 && static_cast<int64_t>(front.deadline_ms) <= now_ms) {
                expired.push_back(std::move(front));
            } else {
                remaining.push(std::move(front));
            }
            pending_requests_.pop();
        }
        remaining_count = remaining.size();
        pending_requests_ = std::move(remaining);
    }
    if (!expired.empty()) {
        RTP_LLM_LOG_WARNING("rdma connection %s:%u deadline sweep: expiring %zu requests, %zu remaining, state=%d",
                            ip_.c_str(),
                            port_,
                            expired.size(),
                            remaining_count,
                            static_cast<int>(getState()));
    }
    for (auto& req : expired) {
        RTP_LLM_LOG_WARNING("rdma connection %s:%u deadline sweep: failing expired pending request "
                            "(deadline_ms=%lu, block_count=%zu)",
                            ip_.c_str(),
                            port_,
                            req.deadline_ms,
                            req.local_remote_buffers.size());
        req.callback(TransferErrorCode::TIMEOUT,
                     readContext() + "pending READ expired, deadline_ms=" + std::to_string(req.deadline_ms));
    }
}

std::string RdmaConnection::readContext() const {
    return "RDMA READ peer=" + ip_ + ":" + std::to_string(port_) + " ";
}

uint32_t RdmaConnection::remoteRkeyOrAbort(const RemoteBuffer& remote, uint32_t nic_id) const {
    if (!remote.nic_rkeys || remote.nic_rkeys->find(nic_id) == remote.nic_rkeys->end()) {
        RTP_LLM_LOG_ERROR("RDMA READ rkey lookup failed, partition_layer_key=%s stage=remote_rkey_lookup "
                          "peer=%s:%u remote_addr=%ld bytes=%u nic=%u",
                          remote.partition_layer_key.c_str(),
                          ip_.c_str(),
                          port_,
                          remote.addr,
                          remote.len,
                          nic_id);
        std::abort();
    }
    return remote.nic_rkeys->at(nic_id);
}

TransferErrorCode RdmaConnection::validateAndBuildRwMemps(
    const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
    std::vector<::accl::barex::rw_memp_t>& mems,
    uint64_t deadline_ms,
    const std::shared_ptr<::accl::barex::XConnection>& conn,
    std::string& error_message) {

    mems.clear();
    error_message.clear();

    auto concrete_mm = std::dynamic_pointer_cast<RdmaMemoryManager>(memory_manager_);
    if (!concrete_mm) {
        RTP_LLM_LOG_ERROR("rdma connection read failed, memory manager is not RdmaMemoryManager");
        error_message = readContext() + "invalid memory manager";
        return TransferErrorCode::UNKNOWN;
    }

    for (const auto& [local_buffer, remote_buffer] : local_remote_buffers) {
        if (!local_buffer.defined()) {
            RTP_LLM_LOG_WARNING("rdma connection read failed, local buffer is not defined");
            error_message = readContext() + "undefined local buffer";
            return TransferErrorCode::BUFFER_MISMATCH;
        }

        if (!remote_buffer) {
            RTP_LLM_LOG_WARNING("rdma connection read failed, remote buffer is nullptr");
            error_message = readContext() + "null remote buffer";
            return TransferErrorCode::BUFFER_MISMATCH;
        }

        void*  data_ptr   = local_buffer.data_ptr();
        bool   is_gpu     = local_buffer.is_cuda();
        size_t size_bytes = local_buffer.nbytes();

        if (size_bytes != remote_buffer->len) {
            RTP_LLM_LOG_WARNING(
                "rdma connection read failed, size mismatch: local %lu, remote %u", size_bytes, remote_buffer->len);
            error_message = readContext() + "buffer size mismatch: local=" + std::to_string(size_bytes)
                            + " remote=" + std::to_string(remote_buffer->len);
            return TransferErrorCode::BUFFER_MISMATCH;
        }

        ::accl::barex::memp_t mem;
        if (!concrete_mm->findMemoryMr(mem, data_ptr, size_bytes, is_gpu)) {
            RTP_LLM_LOG_ERROR("RDMA READ MR lookup failed, partition_layer_key=%s stage=local_mr_lookup "
                              "peer=%s:%u local_addr=%p bytes=%zu cuda=%d remote_addr=%ld",
                              remote_buffer->partition_layer_key.c_str(),
                              ip_.c_str(),
                              port_,
                              data_ptr,
                              size_bytes,
                              is_gpu,
                              remote_buffer->addr);
            std::abort();
        }

        uint64_t r_addr = static_cast<uint64_t>(remote_buffer->addr);
        uint32_t nic_id = conn->GetChannel()->GetPeerNicId();
        const auto rkey = remoteRkeyOrAbort(*remote_buffer, nic_id);

        ::accl::barex::rw_memp_t rw_mem;
        rw_mem.data     = mem;
        rw_mem.r_addr   = r_addr;
        rw_mem.r_key = rkey;
        rw_mem.r_ttl_ms = deadline_ms;
        mems.push_back(rw_mem);
    }

    return TransferErrorCode::OK;
}

void RdmaConnection::sendReadRequest(
    const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
    const std::shared_ptr<RdmaMetricsCollector>& collector,
    RdmaReadCallback callback,
    uint64_t deadline_ms,
    const std::shared_ptr<::accl::barex::XConnection>& conn) {
    collector->pending_done_time_us = currentTimeUs();

    if (!conn) {
        callback(TransferErrorCode::CONNECTION_FAILED, readContext() + "connection is null before READ dispatch");
        return;
    }

    std::vector<::accl::barex::rw_memp_t> mems;
    std::string error_message;
    const auto ec = validateAndBuildRwMemps(local_remote_buffers, mems, deadline_ms, conn, error_message);
    if (ec != TransferErrorCode::OK) {
        callback(ec, error_message);
        return;
    }

    auto rdma_callback =
        [callback, collector, ip = ip_, port = port_, num_mems = mems.size()](::accl::barex::Status status) {
            collector->write_done_time_us = currentTimeUs();
            int64_t rdma_latency_us = collector->write_done_time_us - collector->pending_done_time_us;
            if (!status.IsOk()) {
                RTP_LLM_LOG_WARNING("rdma connection %s:%u ReadBatch callback failed, mems=%zu, "
                                    "rdma_latency_us=%ld, err=%s",
                                    ip.c_str(),
                                    port,
                                    num_mems,
                                    rdma_latency_us,
                                    status.ErrMsg().c_str());
                callback(TransferErrorCode::RDMA_FAILED,
                         "RDMA ReadBatch peer=" + ip + ":" + std::to_string(port)
                             + " blocks=" + std::to_string(num_mems) + ": " + status.ErrMsg());
                return;
            }
            RTP_LLM_LOG_DEBUG("rdma connection %s:%u ReadBatch callback success, mems=%zu, rdma_latency_us=%ld",
                              ip.c_str(),
                              port,
                              num_mems,
                              rdma_latency_us);
            callback(TransferErrorCode::OK, "");
        };

    RTP_LLM_LOG_DEBUG("rdma connection %s:%u posting ReadBatch, mems=%zu, deadline_ms=%lu",
                      ip_.c_str(),
                      port_,
                      mems.size(),
                      deadline_ms);
    const auto result = conn->GetChannel()->ReadBatch(mems, rdma_callback, false);
    if (result != ::accl::barex::BAREX_SUCCESS) {
        callback(TransferErrorCode::RDMA_FAILED,
                 readContext() + "RDMA ReadBatch submission failed, error=" + std::to_string(result));
    }
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
