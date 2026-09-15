#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.pb.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rtp_llm {
namespace transfer {
namespace tcp {

/// @brief Prefill-side sender: packs block content into a TCP RPC and delivers it to TcpKVCacheReceiver.
class TcpKVCacheSender: public transfer::IKVCacheSender {
public:
    explicit TcpKVCacheSender(const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~TcpKVCacheSender() = default;

public:
    /// @brief 初始化 TCP client；idle_ttl_ms 为 0 时关闭 idle 淘汰，sweep_interval_calls 为 0 时仅 miss 路径清扫
    bool init(int                       io_thread_count,
              std::chrono::milliseconds channel_idle_ttl     = std::chrono::milliseconds::zero(),
              std::uint64_t             sweep_interval_calls = 0);

    /// @brief No-op for TCP mode: memory registration is not required.
    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;

    /// @brief 将 block 数据通过 TCP RPC 发送到 Decode 端，完成后调用 callback
    void send(const transfer::SendRequest&                               request,
              std::function<void(TransferErrorCode, const std::string&)> callback) override;

private:
    std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferRequest>
    makeTransferRequest(const transfer::SendRequest&                           request,
                        const std::shared_ptr<TransferClientMetricsCollector>& collector);

    bool setBlockBufferInfo(::tcp_transfer::TcpBlockBufferInfo* block_buffer_info,
                            int64_t                             cache_key,
                            const BlockInfo&                    block_info,
                            std::vector<CopyTask>&              copy_tasks);

    void loadToRemote(const std::string&                                                   ip,
                      uint32_t                                                             port,
                      const std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferRequest>& transfer_request,
                      std::function<void(TransferErrorCode, const std::string&)>           callback,
                      int64_t                                                              deadline_ms);

private:
    std::shared_ptr<transfer::TcpClient> tcp_client_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
    std::unique_ptr<CudaCopyUtil>        cuda_copy_util_;
};

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
