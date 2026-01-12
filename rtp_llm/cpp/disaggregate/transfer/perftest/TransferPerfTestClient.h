#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/disaggregate/transfer/TransferClient.h"
#include "rtp_llm/cpp/disaggregate/transfer/perftest/PerfTestLayerBlockConvertor.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

/// @brief Client 配置参数
struct PerfTestClientConfig {
    std::string server_ip           = "127.0.0.1";
    uint32_t    server_port         = 8888;
    int         block_count         = 10;
    size_t      block_size          = 1024 * 1024;  // 1MB
    int         transfer_count      = 100;
    int         tcp_io_thread_count = 4;
    int         timeout_ms          = 10000;  // 10秒
    bool        use_rdma            = false;
    bool        enable_metrics      = true;  // 是否启用 metrics 上报
};

/// @brief 传输统计信息
struct PerfTestTransferStats {
    std::atomic<int>    completed{0};
    std::atomic<int>    failed{0};
    std::mutex          latency_mutex;
    std::vector<double> latencies;  // in milliseconds
    int64_t             start_time_us = 0;
    int64_t             end_time_us   = 0;

    void reset() {
        completed = 0;
        failed    = 0;
        latencies.clear();
        start_time_us = 0;
        end_time_us   = 0;
    }
};

/// @brief Transfer 性能测试 Client 封装类
class TransferPerfTestClient {
public:
    TransferPerfTestClient(const PerfTestClientConfig& config);
    ~TransferPerfTestClient() = default;

    /// @brief 初始化客户端
    /// @return 成功返回 true
    bool init();

    /// @brief 运行性能测试
    /// @return 失败数 > 0 返回 1，否则返回 0
    int run();

    /// @brief 获取统计信息
    const PerfTestTransferStats& getStats() const {
        return stats_;
    }

    /// @brief 打印配置信息
    void printConfig() const;

    /// @brief 打印统计结果
    void printStats() const;

private:
    /// @brief 执行单次传输（同步等待完成）
    /// @param transfer_id 传输 ID
    /// @return 成功返回 true
    bool doTransfer(int transfer_id);

private:
    PerfTestClientConfig  config_;
    PerfTestTransferStats stats_;

    DeviceBase*                                  device_ = nullptr;
    std::shared_ptr<PerfTestLayerBlockConvertor> layer_block_convertor_;
    std::shared_ptr<TransferClient>              transfer_client_;
    kmonitor::MetricsReporterPtr                 metrics_reporter_;
};

}  // namespace rtp_llm
