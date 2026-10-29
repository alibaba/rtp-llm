#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMetric.h"
#include <memory>
#include <string>
#include <mutex>
#include <vector>
#include <queue>
#include <functional>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

enum class RdmaConnectionState {
    CONNECTING,
    CONNECTED,
    FAILED
};

// Each RdmaConnection wraps one ::accl::barex::XConnection.
//
// Synchronisation model
// ─────────────────────
// conn_mutex_    : guards BOTH state_ AND connection_.
// pending_mutex_ : guards pending_requests_ queue only.
//
// The two mutexes are NEVER held simultaneously.  read() uses an
// "enqueue-first, then check state" pattern so that drainPendingRequests()
// (atomic under pending_mutex_) is the single arbitration point – each
// request is drained exactly once regardless of concurrent state transitions.
//
// After enqueue, read() calls flushPendingForCurrentStateOnce() up to
// kMaxFlushAttempts (see .cc): while state is CONNECTING it yields and
// retries so a concurrent onConnectSuccess/onConnectFailed can be observed
// without blocking indefinitely.
class RdmaConnection: public IRdmaConnection {
public:
    RdmaConnection(const std::shared_ptr<IRdmaMemoryManager>&        memory_manager,
                   const std::string&                                ip,
                   uint32_t                                          port,
                   const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter);
    ~RdmaConnection();

    RdmaConnectionState getState() const;

    /// @brief transport 层异步连接成功时回调
    void onConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection);

    /// @brief transport 层连接失败或断开时回调
    void onConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection);

    /// @brief 发起 RDMA Read；CONNECTED 时立即派发，否则入队等待或同步失败
    void read(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
              std::function<void(bool)>                                                  callback,
              uint64_t                                                                   deadline_ms) override;

    const std::string& getIp() const { return ip_; }
    uint32_t           getPort() const { return port_; }

private:
    struct PendingReadRequest {
        std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>> local_remote_buffers;
        std::function<void(bool)>                                            callback;
        std::shared_ptr<RdmaMetricsCollector>                                collector;
        uint64_t                                                             deadline_ms;
    };

    // Enqueues a single request under pending_mutex_.
    void enqueuePendingRequest(PendingReadRequest request);

    /// Snapshots state_/connection_ under conn_mutex_, then if CONNECTED or FAILED
    /// drains pending and dispatches or fails. Returns true iff state was CONNECTING
    /// (nothing drained; caller may retry after yield).
    bool flushPendingForCurrentStateOnce();

    // Drains all pending requests under pending_mutex_.
    std::vector<PendingReadRequest> drainPendingRequests();

    // Dispatches each request in the list using the given connection (called outside all locks).
    void dispatchRequests(std::vector<PendingReadRequest>& requests,
                          const std::shared_ptr<::accl::barex::XConnection>& conn);

    // Fails all requests in the list (called outside all locks).
    static void failRequests(std::vector<PendingReadRequest>& requests);

    void sendReadRequest(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
                         const std::shared_ptr<RdmaMetricsCollector>&                               collector,
                         std::function<void(bool)>                                                  callback,
                         uint64_t                                                                   deadline_ms,
                         const std::shared_ptr<::accl::barex::XConnection>&                         conn);

    bool validateAndBuildRwMemps(
        const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
        std::vector<::accl::barex::rw_memp_t>&                                      mems,
        uint64_t                                                                    deadline_ms,
        const std::shared_ptr<::accl::barex::XConnection>&                          conn);

private:
    std::shared_ptr<IRdmaMemoryManager>         memory_manager_;
    std::string                                ip_;
    uint32_t                                   port_;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;

    // conn_mutex_ protects BOTH state_ and connection_.  Never held together with pending_mutex_.
    mutable std::mutex                          conn_mutex_;
    RdmaConnectionState                         state_{RdmaConnectionState::CONNECTING};
    std::shared_ptr<::accl::barex::XConnection> connection_;

    std::mutex                     pending_mutex_;
    std::queue<PendingReadRequest> pending_requests_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
