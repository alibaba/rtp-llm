#include "rtp_llm/cpp/cache/connector/p2p/transfer/perftest/TransferPerfTestServer.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <map>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

TransferPerfTestServer::TransferPerfTestServer(const PerfTestServerConfig& config): config_(config) {}

TransferPerfTestServer::~TransferPerfTestServer() {
    if (transfer_server_) {
        transfer_server_.reset();
    }
    if (layer_block_convertor_) {
        layer_block_convertor_.reset();
    }
    if (metrics_reporter_) {
        metrics_reporter_.reset();
    }
}

bool TransferPerfTestServer::init() {
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

    // 预分配 buffers
    if (!layer_block_convertor_->preallocateBuffers(config_.block_count)) {
        std::cerr << "Failed to preallocate buffers" << std::endl;
        return false;
    }

    // 创建 TransferServer
    transfer_server_ = std::make_shared<TransferServer>(layer_block_convertor_, nullptr, metrics_reporter_);

    int      rdma_io_thread_count      = 1;
    int      rdma_worker_thread_count  = 1;
    uint32_t rdma_connections_per_host = 1;
    int      connect_timeout_ms        = 5000;

    if (!transfer_server_->init(config_.use_rdma,
                                config_.port,
                                config_.tcp_io_thread_count,
                                config_.tcp_worker_thread_count,
                                rdma_io_thread_count,
                                rdma_worker_thread_count,
                                rdma_connections_per_host,
                                connect_timeout_ms)) {
        std::cerr << "Failed to init transfer server" << std::endl;
        return false;
    }

    // register buffers
    auto buffers = layer_block_convertor_->getBuffers();
    for (auto& buffer : buffers) {
        if (!transfer_server_->registerUserMr(buffer, config_.block_size)) {
            std::cerr << "Failed to register user mr" << std::endl;
            return false;
        }
    }
    return true;
}

void TransferPerfTestServer::printConfig() const {
    std::cout << "=== Transfer Performance Test Server ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Port: " << config_.port << std::endl;
    std::cout << "  Block count: " << config_.block_count << std::endl;
    std::cout << "  Block size: " << config_.block_size << " bytes (" << (config_.block_size / 1024.0 / 1024.0)
              << " MB)" << std::endl;
    std::cout << "  Expected transfer count: " << config_.transfer_count << std::endl;
    std::cout << "  TCP IO threads: " << config_.tcp_io_thread_count << std::endl;
    std::cout << "  TCP worker threads: " << config_.tcp_worker_thread_count << std::endl;
    std::cout << "  Use RDMA: " << (config_.use_rdma ? "true" : "false") << std::endl;
    std::cout << "  Enable metrics: " << (config_.enable_metrics ? "true" : "false") << std::endl;
}

std::string TransferPerfTestServer::getTaskKey(int task_id) const {
    return "transfer_" + std::to_string(task_id);
}

void TransferPerfTestServer::addOneTask(int task_id) {
    auto task_store = transfer_server_->getLayerCacheBufferTaskStore();

    std::string unique_key         = getTaskKey(task_id);
    auto        layer_cache_buffer = std::make_shared<LayerCacheBuffer>(0);  // layer_id = 0

    for (int b = 0; b < config_.block_count; ++b) {
        int64_t cache_key = b;
        layer_cache_buffer->addBlockId(cache_key, b);
    }

    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_buffers;
    layer_buffers[0] = layer_cache_buffer;

    int64_t deadline_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count()
        + 300000;  // 5 minutes deadline

    task_store->addTask(unique_key, layer_buffers, deadline_ms);
}

int TransferPerfTestServer::run(std::atomic<bool>& running) {
    printConfig();

    auto task_store = transfer_server_->getLayerCacheBufferTaskStore();

    // 只添加第一个 Task（串行模式）
    current_task_id_ = 0;
    addOneTask(current_task_id_);
    current_task_id_++;

    std::cout << "Added first task to task store (serial mode)" << std::endl;

    std::cout << "\nServer is running on port " << config_.port << std::endl;
    std::cout << "Press Ctrl+C to stop..." << std::endl;

    // 统计信息
    int64_t start_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    completed_count_   = 0;
    int active_task_id = 0;  // 当前正在处理的任务 ID

    std::vector<std::shared_ptr<LayerCacheBufferTask>> loading_tasks;

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 更频繁地检查

        // 获取当前任务
        std::string current_key = getTaskKey(active_task_id);
        auto        task        = task_store->getTask(current_key);

        if (!task) {
            // 任务不存在，可能还没添加或已被清除
            continue;
        }

        // 使用 task->success() 或 task->isTimeout() 判断任务是否完成
        bool task_done = task->success() || task->isTimeout();

        if (task_done) {
            bool is_success = task->success();

            // 清除已完成的任务
            task_store->stealTask(current_key);

            if (task->hasLoadingLayer()) {
                loading_tasks.push_back(task);
            }

            completed_count_++;

            int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::system_clock::now().time_since_epoch())
                                       .count();
            double elapsed_sec = (current_time - start_time) / 1000.0;
            double throughput  = completed_count_ / elapsed_sec;

            std::cout << "\rCompleted: " << completed_count_ << "/" << config_.transfer_count
                      << " | Throughput: " << throughput << " transfers/sec"
                      << " | Status: " << (is_success ? "success" : "timeout") << std::flush;

            active_task_id++;

            // 如果还有更多 Task 需要添加
            if (current_task_id_ < config_.transfer_count) {
                addOneTask(current_task_id_);
                current_task_id_++;
            } else if (completed_count_ >= config_.transfer_count) {
                // 所有 Task 都已完成
                std::cout << "\nAll transfers completed!" << std::endl;
                break;
            }
        }
    }

    while (!loading_tasks.empty()) {
        auto task = loading_tasks.front();
        loading_tasks.erase(loading_tasks.begin());
        if (task->hasLoadingLayer()) {
            loading_tasks.push_back(task);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Waiting for loading tasks to complete..." << std::endl;
    }

    std::cout << "\nShutting down server..." << std::endl;
    return 0;
}

}  // namespace rtp_llm
