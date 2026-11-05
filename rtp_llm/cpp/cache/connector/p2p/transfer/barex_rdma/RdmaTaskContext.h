#pragma once

#include <google/protobuf/service.h>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include <memory>
#include <optional>
#include <vector>

namespace rdma_transfer {
class RdmaLayerBlockTransferRequest;
class RdmaLayerBlockTransferResponse;
}  // namespace rdma_transfer

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief Encapsulates one RDMA transfer RPC: holds proto request/response/closure and the matching recv task.
class RdmaTaskContext {
public:
    RdmaTaskContext(::google::protobuf::RpcController*                     controller,
                    const ::rdma_transfer::RdmaLayerBlockTransferRequest*  request,
                    ::rdma_transfer::RdmaLayerBlockTransferResponse*       response,
                    ::google::protobuf::Closure*                           done,
                    const kmonitor::MetricsReporterPtr&                    metrics_reporter);
    ~RdmaTaskContext();

public:
    /// @brief 绑定对应的 recv task（由 RdmaTransferService 在匹配到 unique_key 后调用）
    void        setTask(const std::shared_ptr<TransferTask>& task);
    std::string getUniqueKey() const;
    bool        isTimeout() const;
    uint64_t    getDeadlineMs() const;

    /// @brief 委托给底层 TransferTask::startTransfer()。
    /// @return false 表示任务已在 PENDING 阶段被 cancel，调用方应立即报告失败。
    bool startTransfer();

    std::string getRdmaServerIp() const;
    uint32_t    getRdmaServerPort() const;

    /// @brief Builds (local_dest_tensor, remote_buffer) pairs for RDMA Read.
    std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>> getRdmaBlockPair();

    /// @brief 完成 RPC 响应并通知 task（成功或失败）
    void run(bool                          success,
             TransferErrorCode   error_code    = TransferErrorCode::OK,
             const std::string&            error_message = "");

private:
    std::optional<torch::Tensor> findLocalBuffer(int64_t cache_key, int sub_block_idx) const;

private:
    ::google::protobuf::RpcController*                    controller_;
    const ::rdma_transfer::RdmaLayerBlockTransferRequest* request_;
    ::rdma_transfer::RdmaLayerBlockTransferResponse*      response_;
    ::google::protobuf::Closure*                          done_;
    kmonitor::MetricsReporterPtr                          metrics_reporter_;

    std::string unique_key_;

    std::shared_ptr<TransferServerMetricsCollector> collector_;
    int64_t                                             start_time_us_ = 0;

    std::shared_ptr<TransferTask> task_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
