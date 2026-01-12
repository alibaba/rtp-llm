#include "rtp_llm/cpp/disaggregate/transfer/perftest/TransferPerfTestClient.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <numeric>
#include <algorithm>
#include <condition_variable>

#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

TransferPerfTestClient::TransferPerfTestClient(const PerfTestClientConfig& config): config_(config) {}

bool TransferPerfTestClient::init() {
    // 初始化设备
    ParallelismConfig    parallelism_config;
    ModelConfig          model_config;
    EPLBConfig           eplb_config;
    FMHAConfig           fmha_config;
    DeviceResourceConfig device_resource_config;
    device_resource_config.device_reserve_memory_bytes = 1024L * 1024 * 1024;      // 1GB
    device_resource_config.host_reserve_memory_bytes   = 1L * 1024 * 1024 * 1024;  // 1GB
    MoeConfig                   moe_config;
    SpeculativeExecutionConfig  sp_config;
    MiscellaneousConfig         misc_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    HWKernelConfig              hw_kernel_config;
    ConcurrencyConfig           concurrency_config;
    FfnDisAggregateConfig       ffn_disaggregate_config;
    RuntimeConfig               runtime_config;

    DeviceFactory::initDevices(parallelism_config,
                               model_config,
                               eplb_config,
                               fmha_config,
                               device_resource_config,
                               moe_config,
                               sp_config,
                               misc_config,
                               profiling_debug_logging_config,
                               hw_kernel_config,
                               concurrency_config,
                               ffn_disaggregate_config,
                               runtime_config);
    device_ = DeviceFactory::getDefaultDevice();
    if (!device_) {
        std::cerr << "Failed to get device" << std::endl;
        return false;
    }

    // 创建 metrics reporter
    if (config_.enable_metrics) {
        auto kmon_tags = kmonitor::MetricsTags();
        metrics_reporter_.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
    }

    // 创建 LayerBlockConvertor
    layer_block_convertor_ = std::make_shared<PerfTestLayerBlockConvertor>(device_, config_.block_size);

    // 预分配 buffers (填充测试数据 'A')
    if (!layer_block_convertor_->preallocateBuffers(config_.block_count, 'A')) {
        std::cerr << "Failed to preallocate buffers" << std::endl;
        return false;
    }

    // 创建 TransferClient
    transfer_client_ = std::make_shared<TransferClient>(layer_block_convertor_, nullptr, metrics_reporter_);

    int rdma_io_thread_count     = 1;
    int rdma_worker_thread_count = 1;

    if (!transfer_client_->init(
            config_.use_rdma, config_.tcp_io_thread_count, rdma_io_thread_count, rdma_worker_thread_count)) {
        std::cerr << "Failed to init transfer client" << std::endl;
        return false;
    }

    // register buffers
    auto buffers = layer_block_convertor_->getBuffers();
    for (auto& buffer : buffers) {
        if (!transfer_client_->registerUserMr(buffer, config_.block_size)) {
            std::cerr << "Failed to register user mr" << std::endl;
            return false;
        }
    }

    return true;
}

void TransferPerfTestClient::printConfig() const {
    std::cout << "=== Transfer Performance Test Client ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Server: " << config_.server_ip << ":" << config_.server_port << std::endl;
    std::cout << "  Block count: " << config_.block_count << std::endl;
    std::cout << "  Block size: " << config_.block_size << " bytes (" << (config_.block_size / 1024.0 / 1024.0)
              << " MB)" << std::endl;
    std::cout << "  Transfer count: " << config_.transfer_count << std::endl;
    std::cout << "  TCP IO threads: " << config_.tcp_io_thread_count << std::endl;
    std::cout << "  Timeout: " << config_.timeout_ms << " ms" << std::endl;
    std::cout << "  Use RDMA: " << (config_.use_rdma ? "true" : "false") << std::endl;
    std::cout << "  Enable metrics: " << (config_.enable_metrics ? "true" : "false") << std::endl;

    size_t total_data = config_.block_count * config_.block_size * config_.transfer_count;
    std::cout << "  Total data to transfer: " << (total_data / 1024.0 / 1024.0) << " MB" << std::endl;
}

void TransferPerfTestClient::printStats() const {
    std::cout << "\n=== Transfer Performance Test Results ===" << std::endl;

    int    total        = stats_.completed + stats_.failed;
    double success_rate = (total > 0) ? (100.0 * stats_.completed / total) : 0.0;

    std::cout << "Completed: " << stats_.completed << std::endl;
    std::cout << "Failed: " << stats_.failed << std::endl;
    std::cout << "Success rate: " << success_rate << "%" << std::endl;

    if (stats_.latencies.empty()) {
        std::cout << "No latency data available." << std::endl;
        return;
    }

    // 计算延迟统计
    std::vector<double> latencies;
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(stats_.latency_mutex));
        latencies = stats_.latencies;
    }

    std::sort(latencies.begin(), latencies.end());

    double sum     = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg     = sum / latencies.size();
    double min_val = latencies.front();
    double max_val = latencies.back();
    double p50     = latencies[latencies.size() / 2];
    double p90     = latencies[static_cast<size_t>(latencies.size() * 0.9)];
    double p99     = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::cout << "\nLatency Statistics (ms):" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Avg: " << avg << std::endl;
    std::cout << "  P50: " << p50 << std::endl;
    std::cout << "  P90: " << p90 << std::endl;
    std::cout << "  P99: " << p99 << std::endl;

    // 计算吞吐量
    double total_time_sec    = (stats_.end_time_us - stats_.start_time_us) / 1000000.0;
    double transfers_per_sec = stats_.completed / total_time_sec;

    size_t data_per_transfer = config_.block_count * config_.block_size;
    double data_per_sec_mb   = (stats_.completed * data_per_transfer) / (total_time_sec * 1024 * 1024);

    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  Total time: " << total_time_sec << " seconds" << std::endl;
    std::cout << "  Transfers/sec: " << transfers_per_sec << std::endl;
    std::cout << "  Data throughput: " << data_per_sec_mb << " MB/s" << std::endl;
}

bool TransferPerfTestClient::doTransfer(int transfer_id) {
    std::atomic<bool>       transfer_done{false};
    std::mutex              cv_mutex;
    std::condition_variable cv;
    bool                    success = false;

    // 创建 LayerCacheBuffer
    auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(0);  // layer_id = 0
    for (int b = 0; b < config_.block_count; ++b) {
        int64_t cache_key = b;
        layer_cache_buffer->addBlockId(cache_key, b);
    }

    std::string unique_key        = "transfer_" + std::to_string(transfer_id);
    int64_t     transfer_start_us = currentTimeUs();

    transfer_client_->transfer(
        config_.server_ip,
        config_.server_port,
        unique_key,
        layer_cache_buffer,
        1,  // local_partition_count
        0,  // local_partition_id
        1,  // remote_partition_count
        0,  // remote_partition_id
        [this, &transfer_done, &cv, &success, transfer_start_us, transfer_id](bool result) {
            int64_t end_time_us = currentTimeUs();
            double  latency_ms  = (end_time_us - transfer_start_us) / 1000.0;

            if (result) {
                stats_.completed++;
                {
                    std::lock_guard<std::mutex> lock(stats_.latency_mutex);
                    stats_.latencies.push_back(latency_ms);
                }
                success = true;
            } else {
                stats_.failed++;
                std::cerr << "Transfer " << transfer_id << " failed" << std::endl;
                success = false;
            }

            transfer_done = true;
            cv.notify_one();
        },
        config_.timeout_ms);

    // 等待当前传输完成
    {
        std::unique_lock<std::mutex> lock(cv_mutex);
        cv.wait(lock, [&transfer_done]() { return transfer_done.load(); });
    }

    return success;
}

int TransferPerfTestClient::run() {
    printConfig();

    stats_.reset();

    std::cout << "\nStarting transfers (serial mode)..." << std::endl;
    stats_.start_time_us = currentTimeUs();

    // 串行执行：一个传输完成后再开启下一个
    for (int transfer_id = 0; transfer_id < config_.transfer_count; ++transfer_id) {
        doTransfer(transfer_id);

        // 打印进度
        int completed = stats_.completed + stats_.failed;
        std::cout << "\rProgress: " << completed << "/" << config_.transfer_count << std::flush;
    }

    stats_.end_time_us = currentTimeUs();

    // 打印统计结果
    printStats();

    return stats_.failed > 0 ? 1 : 0;
}

}  // namespace rtp_llm
