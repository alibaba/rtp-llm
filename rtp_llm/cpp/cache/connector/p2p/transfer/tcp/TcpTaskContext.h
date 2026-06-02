#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.pb.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace rtp_llm {
namespace transfer {
namespace tcp {

/// @brief Encapsulates one TCP transfer RPC: holds proto request/response/closure and the matching recv task.
class TcpTaskContext {
public:
    TcpTaskContext(::google::protobuf::RpcController*                  controller,
                   const ::tcp_transfer::TcpLayerBlockTransferRequest* request,
                   ::tcp_transfer::TcpLayerBlockTransferResponse*      response,
                   ::google::protobuf::Closure*                        done,
                   const kmonitor::MetricsReporterPtr&                 metrics_reporter);
    ~TcpTaskContext();

public:
    /// @brief 绑定对应的 recv task（由 TcpTransferService 在匹配到 unique_key 后调用）
    void               setTask(const std::shared_ptr<TransferTask>& task);
    const std::string& getUniqueKey() const;
    bool               isTimeout() const;
    uint64_t           getDeadlineMs() const;

    /// @brief 委托给底层 TransferTask::startTransfer()。
    /// @return false 表示任务已在 PENDING 阶段被 cancel，调用方应立即报告失败。
    bool startTransfer();

    /// @brief Verifies that every non-empty block in the task's KeyBlockInfoMap exists in the request
    ///        with a matching size, then batch-copies all data to device.
    /// @return true on success; false if any expected block is absent, sizes mismatch, or copy fails.
    bool executeCopy(CudaCopyUtil& cuda_copy_util);

    /// @brief 完成 RPC 响应并通知 task（成功或失败）
    void run(bool success, TransferErrorCode error_code = TransferErrorCode::OK, const std::string& error_message = "");

private:
    ::google::protobuf::RpcController*                  controller_;
    const ::tcp_transfer::TcpLayerBlockTransferRequest* request_;
    ::tcp_transfer::TcpLayerBlockTransferResponse*      response_;
    ::google::protobuf::Closure*                        done_;
    kmonitor::MetricsReporterPtr                        metrics_reporter_;

    std::string unique_key_;

    std::shared_ptr<TransferServerMetricsCollector> collector_;
    int64_t                                         start_time_us_ = 0;

    std::shared_ptr<TransferTask> task_;
};

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
