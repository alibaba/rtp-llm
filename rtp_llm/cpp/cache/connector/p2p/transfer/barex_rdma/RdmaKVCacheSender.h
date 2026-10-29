#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"
#include <memory>
#include <functional>
#include <string>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief Prefill-side sender: registers memory MRs and packs rkeys into a TCP RPC for RDMA Read.
class RdmaKVCacheSender : public IKVCacheSender {
public:
    RdmaKVCacheSender(const std::shared_ptr<IRdmaMemoryManager>& rdma_memory_manager,
                      const kmonitor::MetricsReporterPtr&         metrics_reporter = nullptr);
    ~RdmaKVCacheSender() = default;

public:
    /// @brief 初始化 TCP client 及 RDMA server
    bool init(int tcp_io_thread_count,
              int rdma_io_thread_count,
              int rdma_worker_thread_count);

    /// @brief 将 block 内存注册为 RDMA MR，并向 RDMA server 注册
    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;

    /// @brief 将 rkey 信息通过 TCP RPC 发送给 Decode 端，触发对方 RDMA Read
    void send(const SendRequest&                                               request,
              std::function<void(TransferErrorCode, const std::string&)> callback) override;

    const std::shared_ptr<IRdmaMemoryManager>& getRdmaMemoryManager() const {
        return rdma_memory_manager_;
    }

private:
    std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferRequest>
    makeTransferRequest(const SendRequest&                                                  request,
                        const std::shared_ptr<TransferClientMetricsCollector>& collector);

    bool setBlockBufferInfo(::rdma_transfer::RdmaBlockBufferInfo* block_buffer_info,
                            int64_t                               cache_key,
                            const BlockInfo&                      block_info);

    void loadToRemote(const std::string&                                                             ip,
                      uint32_t                                                                       port,
                      const std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferRequest>&        transfer_request,
                      std::function<void(TransferErrorCode, const std::string&)>          callback,
                      int64_t                                                                        deadline_ms);

private:
    std::shared_ptr<TcpClient>   tcp_client_;
    std::shared_ptr<IRdmaServer>           rdma_server_;
    std::shared_ptr<IRdmaMemoryManager>    rdma_memory_manager_;
    kmonitor::MetricsReporterPtr           metrics_reporter_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
