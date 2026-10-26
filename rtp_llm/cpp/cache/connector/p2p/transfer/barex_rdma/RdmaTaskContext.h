#pragma once

#include <google/protobuf/service.h>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rdma_transfer {
class RdmaLayerBlockTransferRequest;
class RdmaLayerBlockTransferResponse;
}  // namespace rdma_transfer

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

using RdmaBlockPair = std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>;

class RdmaStagingBufferPool {
public:
    static std::shared_ptr<RdmaStagingBufferPool>
    create(const std::shared_ptr<IRdmaMemoryManager>& memory_manager, size_t total_bytes, size_t block_size_bytes);
    ~RdmaStagingBufferPool();

    std::optional<HostStagingBlockPool::HostStagingBlockBatch>
    mallocBatchWithBackoff(size_t count, std::chrono::milliseconds timeout);
    size_t strideBytes() const;
    size_t blockCount() const;

private:
    RdmaStagingBufferPool(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                          size_t block_count,
                          size_t block_size_bytes);
    bool init(size_t block_count);

    std::shared_ptr<IRdmaMemoryManager> memory_manager_;
    std::shared_ptr<HostStagingBlockPool> pool_;
    size_t block_count_;
    size_t stride_bytes_;
    std::vector<BlockInfo> registered_blocks_;
};

struct RdmaStagingTransfer {
    RdmaStagingTransfer(const std::shared_ptr<RdmaStagingBufferPool>& pool,
                        HostStagingBlockPool::HostStagingBlockBatch&& leases,
                        std::vector<tcp::CopyTask>&& h2d_copy_tasks):
        pool(pool), leases(std::move(leases)), h2d_copy_tasks(std::move(h2d_copy_tasks)) {}

    std::shared_ptr<RdmaStagingBufferPool> pool;
    HostStagingBlockPool::HostStagingBlockBatch leases;
    std::vector<tcp::CopyTask> h2d_copy_tasks;
};

struct RdmaBlockPairBuildResult {
    bool success{false};
    std::vector<RdmaBlockPair> pairs;
    std::string error_message;
    std::shared_ptr<RdmaStagingTransfer> staging_transfer;
    TransferErrorCode error_code{TransferErrorCode::BUFFER_MISMATCH};
};

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

    /// @brief Builds and validates all Decode-expected (local_dest_tensor, remote_buffer) pairs for RDMA Read.
    RdmaBlockPairBuildResult buildRdmaBlockPairs(const std::shared_ptr<RdmaStagingBufferPool>& staging_pool = nullptr);

    /// @brief 完成 RPC 响应并通知 task（成功或失败）
    void run(bool success, TransferErrorCode error_code = TransferErrorCode::OK, const std::string& error_message = "");

    /// @brief Only finishes the transfer RPC. Used by the deadline watchdog while
    /// the physical RDMA operation may still be writing the Decode target buffer.
    void recordTransferError(TransferErrorCode error_code, const std::string& error_message);
    void finishRpc(bool success, TransferErrorCode error_code, const std::string& error_message);

    /// @brief Marks the physical RDMA operation complete. Decode target blocks
    /// must remain referenced until this is called from the RDMA callback.
    void notifyTransferDone(bool success, TransferErrorCode error_code, const std::string& error_message);

private:
    static std::optional<torch::Tensor> findLocalBuffer(const KeyBlockInfo& target, int sub_block_idx);

private:
    ::google::protobuf::RpcController*                    controller_;
    const ::rdma_transfer::RdmaLayerBlockTransferRequest* request_;
    ::rdma_transfer::RdmaLayerBlockTransferResponse*      response_;
    ::google::protobuf::Closure*                          done_;
    kmonitor::MetricsReporterPtr                          metrics_reporter_;

    std::string unique_key_;

    std::shared_ptr<TransferServerMetricsCollector> collector_;
    int64_t start_time_us_ = 0;
    int64_t deadline_ms_ = 0;

    std::shared_ptr<TransferTask> task_;
    std::vector<int64_t> batch_keys_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
