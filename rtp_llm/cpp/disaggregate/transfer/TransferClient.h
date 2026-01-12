#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/transfer/RdmaInterface.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/disaggregate/transfer/CudaCopyUtil.h"
#include "rtp_llm/cpp/disaggregate/transfer/proto/service.pb.h"
#include "rtp_llm/cpp/disaggregate/transfer/TransferMetric.h"
#include "autil/NetUtil.h"
#include <memory>
#include <string>
#include <functional>

namespace rtp_llm {

/// @brief TransferClient 用于将 layer_cache_buffer 中的 buffer 内容传输到 TransferServer
/// 参考 CacheStoreServer 的使用方式
class TransferClient {
public:
    TransferClient(const std::shared_ptr<LayerBlockConvertor>& layer_block_convector,
                   const std::shared_ptr<IRdmaMemoryManager>&  rdma_memory_manager = nullptr,
                   const kmonitor::MetricsReporterPtr&         metrics_reporter    = nullptr):
        layer_block_convector_(layer_block_convector),
        rdma_memory_manager_(rdma_memory_manager),
        metrics_reporter_(metrics_reporter),
        cuda_copy_util_(std::make_unique<CudaCopyUtil>()) {}
    ~TransferClient() = default;

public:
    bool init(bool use_rdma, int tcp_io_thread_count, int rdma_io_thread_count, int rdma_worker_thread_count);

    virtual void transfer(const std::string&                       ip,
                          uint32_t                                 port,
                          const std::string&                       unique_key,
                          const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                          uint32_t                                 local_partition_count,
                          uint32_t                                 local_partition_id,
                          uint32_t                                 remote_partition_count,
                          uint32_t                                 remote_partition_id,
                          std::function<void(bool)>                callback,
                          int                                      timeout_ms = 1000);

    bool registerUserMr(const BufferPtr& buffer, uint64_t aligned_size);

    const std::shared_ptr<IRdmaMemoryManager>& getRdmaMemoryManager() const {
        return rdma_memory_manager_;
    }

private:
    std::shared_ptr<::transfer::LayerBlockTransferRequest>
    makeTransferRequest(const std::shared_ptr<LayerCacheBuffer>&                       layer_cache_buffer,
                        const std::string&                                             unique_key,
                        uint32_t                                                       local_partition_count,
                        uint32_t                                                       local_partition_id,
                        uint32_t                                                       remote_partition_count,
                        uint32_t                                                       remote_partition_id,
                        int                                                            timeout_ms,
                        const std::shared_ptr<TransferClientTransferMetricsCollector>& collector);

    bool setBlockBufferInfo(::transfer::BlockBufferInfo* block_buffer_info,
                            int64_t                      cache_key,
                            int                          block_id,
                            BufferPtr                    buffer,
                            std::vector<CopyTask>&       copy_tasks);

    void loadToRemote(const std::string&                                            ip,
                      uint32_t                                                      port,
                      const std::shared_ptr<LayerCacheBuffer>&                      layer_cache_buffer,
                      const std::shared_ptr<::transfer::LayerBlockTransferRequest>& transfer_request,
                      std::function<void(bool)>                                     callback,
                      int                                                           timeout_ms);

private:
    std::shared_ptr<transfer::TcpClient> tcp_client_;
    std::shared_ptr<IRdmaServer>         rdma_server_;
    std::shared_ptr<LayerBlockConvertor> layer_block_convector_;
    std::shared_ptr<IRdmaMemoryManager>  rdma_memory_manager_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
    std::unique_ptr<CudaCopyUtil>        cuda_copy_util_;
    // RDMA IP 和 port（用于填充 transfer request）
    std::string rdma_ip_;
    uint32_t    rdma_listen_port_ = 0;
};

}  // namespace rtp_llm
