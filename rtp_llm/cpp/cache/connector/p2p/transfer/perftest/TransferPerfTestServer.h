#pragma once

#include <string>
#include <atomic>
#include <memory>
#include <functional>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/perftest/PerfTestLayerBlockConvertor.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

/// @brief Server 配置参数
struct PerfTestServerConfig {
    uint32_t port                                = 8888;
    int      block_count                         = 10;
    size_t   block_size                          = 1024 * 1024;  // 1MB
    int      transfer_count                      = 100;
    int      tcp_io_thread_count                 = 4;
    int      tcp_worker_thread_count             = 8;
    bool     use_rdma                            = false;
    bool     enable_metrics                      = true;  // 是否启用 metrics 上报
    int      rdma_max_block_pairs_per_connection = 0;
};

/// @brief Transfer 性能测试 Server 封装类
class TransferPerfTestServer {
public:
    TransferPerfTestServer(const PerfTestServerConfig& config);
    ~TransferPerfTestServer();

    /// @brief 初始化服务器
    /// @return 成功返回 true
    bool init();

    /// @brief 运行服务器（阻塞，直到所有传输完成或收到停止信号）
    /// @param running 运行标志，外部可通过设置为 false 来停止服务器
    /// @return 成功返回 0
    int run(std::atomic<bool>& running);

    /// @brief 打印配置信息
    void printConfig() const;

    /// @brief 获取完成的传输数量
    int getCompletedCount() const {
        return completed_count_;
    }

private:
    /// @brief 获取任务的唯一标识 key
    std::string getTaskKey(int task_id) const;

    /// @brief 添加一个传输任务
    void addOneTask(int task_id);

private:
    PerfTestServerConfig config_;

    DeviceBase*                                  device_ = nullptr;
    std::shared_ptr<PerfTestLayerBlockConvertor> layer_block_convertor_;
    std::shared_ptr<TransferServer>              transfer_server_;
    kmonitor::MetricsReporterPtr                 metrics_reporter_;

    int completed_count_ = 0;
    int current_task_id_ = 0;
};

}  // namespace rtp_llm
