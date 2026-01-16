#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorWorker.h"

#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <thread>

namespace rtp_llm {

P2PConnectorWorker::P2PConnectorWorker(const KVCacheConfig&                        cache_config,
                                       const CacheStoreConfig&                     cache_store_config,
                                       const ParallelismConfig&                    parallelism_config,
                                       const PDSepConfig&                          pd_sep_config,
                                       const ModelConfig&                          model_config,
                                       const std::shared_ptr<LayerBlockConvertor>& layer_block_convertor,
                                       const kmonitor::MetricsReporterPtr&         metrics_reporter):
    cache_config_(cache_config),
    cache_store_config_(cache_store_config),
    parallelism_config_(parallelism_config),
    pd_sep_config_(pd_sep_config),
    model_config_(model_config),
    layer_block_convertor_(layer_block_convertor),
    metrics_reporter_(metrics_reporter),
    asymmetric_tp_util_(std::make_shared<AsymmetricTpUtil>(parallelism_config)),
    computed_buffers_(std::make_shared<ComputedLayerCacheBufferStore>()),
    load_contexts_(std::make_shared<PrefillWorkerLoadContextStore>()) {}

P2PConnectorWorker::~P2PConnectorWorker() {
    if (cleanup_thread_) {
        cleanup_thread_->stop();
    }
}

bool P2PConnectorWorker::init(int64_t store_wait_timeout_ms) {
    RTP_LLM_LOG_INFO("P2PConnectorWorker init start, store_wait_timeout_ms: %ld", store_wait_timeout_ms);
    if (!layer_block_convertor_) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker init failed: layer_block_convertor is null");
        return false;
    }

    // 1. 初始化 transfer client
    transfer_client_ = std::make_shared<TransferClient>(layer_block_convertor_, nullptr, metrics_reporter_);
    if (!transfer_client_->init(cache_store_config_.cache_store_rdma_mode,
                                cache_store_config_.messager_io_thread_count,
                                cache_store_config_.messager_io_thread_count,
                                cache_store_config_.messager_worker_thread_count)) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker init failed: transfer_client init failed");
        return false;
    }

    // 2. 注册所有buffer到 transfer client（供RDMA使用）
    auto buffers = layer_block_convertor_->getAllBuffers();
    for (auto& [buffer, size] : buffers) {
        if (!transfer_client_->registerUserMr(buffer, size)) {
            RTP_LLM_LOG_ERROR(
                "P2PConnectorWorker init failed: register user mr failed, buffer: %p, size: %ld", buffer->data(), size);
            return false;
        }
    }

    // 3. 初始化 store wait 上下文检查器
    store_wait_context_checker_ = std::make_shared<StoreWaitContextChecker>(metrics_reporter_, computed_buffers_);

    // 4. 启动定期清理线程
    cleanup_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&P2PConnectorWorker::loopCheckProc, this), 1000, "P2PConnectorWorkerCleanupThread");
    if (!cleanup_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker init failed: cleanup_thread is null");
        return false;
    }

    // 5. 初始化 transfer server
    transfer_server_ = std::make_shared<TransferServer>(
        layer_block_convertor_, transfer_client_->getRdmaMemoryManager(), metrics_reporter_);
    if (!transfer_server_->init(cache_store_config_.cache_store_rdma_mode,
                                pd_sep_config_.cache_store_listen_port,
                                cache_store_config_.messager_io_thread_count,
                                cache_store_config_.messager_worker_thread_count,
                                cache_store_config_.messager_io_thread_count,
                                cache_store_config_.messager_worker_thread_count,
                                2,
                                cache_store_config_.rdma_connect_timeout_ms)) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker init failed: transfer_server init failed");
        return false;
    }

    // 6. 获取 task store
    layer_cache_buffer_task_store_ = transfer_server_->getLayerCacheBufferTaskStore();
    if (!layer_cache_buffer_task_store_) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker init failed: get layer_cache_buffer_task_store failed");
        return false;
    }

    // 7. 设置超时时间
    store_wait_timeout_ms_ = store_wait_timeout_ms;

    RTP_LLM_LOG_INFO("P2PConnectorWorker init success");
    return true;
}

// ================== Prefill 端功能实现 ==================

bool P2PConnectorWorker::writeByLayer(int                       layer_id,
                                      const KVCacheResourcePtr& resource,
                                      int64_t                   request_id,
                                      DeviceEventPtr            event) {
    RTP_LLM_LOG_INFO("P2PConnectorWorker writeByLayer start, request_id: %ld, layer_id: %d", request_id, layer_id);
    auto collector = std::make_shared<P2PConnectorServerWorkerStoreMetricsCollector>();

    auto layer_cache_buffer = LayerCacheBufferUtil::convertLayer(*resource, 0, layer_id, 0, -1);
    if (!layer_cache_buffer) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker writeByLayer failed: layer_cache_buffer is null");
        if (metrics_reporter_) {
            collector->success = false;
            metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorServerWorkerStoreMetricsCollector>(
                nullptr, collector.get());
        }
        return false;
    }
    collector->total_block_count = layer_cache_buffer->blockIdMap().size();

    int64_t deadline_ms = currentTimeMs() + store_wait_timeout_ms_;
    store_wait_context_checker_->addContext(
        StoreWaitContext(request_id, event, layer_cache_buffer, deadline_ms, collector));
    RTP_LLM_LOG_INFO("P2PConnectorWorker writeByLayer end, request_id: %ld, layer_id: %d", request_id, layer_id);
    return true;
}

void P2PConnectorWorker::loopCheckProc() {
    store_wait_context_checker_->checkOnce();
    computed_buffers_->checkTimeout();
    load_contexts_->checkTimeout();

    // 上报状态 metrics
    if (metrics_reporter_) {
        auto collector = std::make_shared<P2PConnectorServerWorkerStatusMetricsCollector>();
        collector->wait_store_event_count =
            store_wait_context_checker_ ? store_wait_context_checker_->getContextCount() : 0;
        collector->task_count             = load_contexts_->getContextsCount();
        collector->computed_request_count = computed_buffers_->getBuffersCount();
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorServerWorkerStatusMetricsCollector>(nullptr,
                                                                                                       collector.get());
    }
}

bool P2PConnectorWorker::handleRead(int64_t                                              request_id,
                                    const std::string&                                   unique_key,
                                    int64_t                                              deadline_ms,
                                    const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    RTP_LLM_LOG_INFO(
        "P2PConnectorWorker handleRead start, request_id: %ld, unique_key: %s, deadline_ms: %ld, decode_transfer_servers_size: %zu",
        request_id,
        unique_key.c_str(),
        deadline_ms,
        decode_transfer_servers.size());
    int64_t start_time_us = currentTimeUs();
    auto    collector     = std::make_shared<P2PConnectorServerWorkerWriteMetricsCollector>();

    auto asymmetric_tp_contexts = asymmetric_tp_util_->handleAsymmetricTP(decode_transfer_servers);
    if (asymmetric_tp_contexts.empty()) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker handleRead: asymmetric_tp_contexts is empty, unique_key: %s",
                          unique_key.c_str());
        if (metrics_reporter_) {
            collector->success = false;
            metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorServerWorkerWriteMetricsCollector>(
                nullptr, collector.get());
        }
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorWorker handleRead asymmetric_tp_contexts size: %zu, unique_key: %s",
                     asymmetric_tp_contexts.size(),
                     unique_key.c_str());

    auto load_context = load_contexts_->addContext(
        request_id, unique_key, deadline_ms, asymmetric_tp_contexts, model_config_.num_layers);
    auto computed_layer_cache_buffer    = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
    collector->first_layer_wait_time_us = currentTimeUs() - start_time_us;

    // wait until all transfers are started
    while (!load_context->isAllTransferStarted() && !load_context->canceled() && !load_context->timeout()) {
        auto need_transfer_ids       = load_context->getNeedTransferIds();
        auto need_transfer_layer_ids = std::set<int>();
        for (auto id : need_transfer_ids) {
            need_transfer_layer_ids.insert(id / static_cast<int>(asymmetric_tp_contexts.size()));
        }
        auto [total_layer_num, layer_cache_buffers] = computed_layer_cache_buffer->getBuffers(need_transfer_layer_ids);
        RTP_LLM_LOG_INFO("P2PConnectorWorker handleRead layer_cache_buffers size: %zu, unique_key: %s",
                         layer_cache_buffers.size(),
                         unique_key.c_str());
        for (auto layer_cache_buffer : layer_cache_buffers) {
            for (size_t i = 0; i < asymmetric_tp_contexts.size(); i++) {
                auto id = layer_cache_buffer->getLayerId() * static_cast<int>(asymmetric_tp_contexts.size())
                          + static_cast<int>(i);
                if (!load_context->startTransfer(id)) {
                    RTP_LLM_LOG_INFO("P2PConnectorWorker handleRead startTransfer failed, id: %d, unique_key: %s",
                                     id,
                                     unique_key.c_str());
                    continue;
                }
                RTP_LLM_LOG_INFO("P2PConnectorWorker handleRead startTransfer success, id: %d, unique_key: %s",
                                 id,
                                 unique_key.c_str());
                transfer_client_->transfer(
                    asymmetric_tp_contexts[i].decode_ip,
                    asymmetric_tp_contexts[i].decode_port,
                    unique_key,
                    layer_cache_buffer,
                    static_cast<uint32_t>(asymmetric_tp_contexts[i].local_partition_count),
                    static_cast<uint32_t>(asymmetric_tp_contexts[i].local_partition_id),
                    static_cast<uint32_t>(asymmetric_tp_contexts[i].remote_partition_count),
                    static_cast<uint32_t>(asymmetric_tp_contexts[i].remote_partition_id),
                    [load_context, id, unique_key](bool success) {
                        RTP_LLM_LOG_INFO(
                            "P2PConnectorWorker handleRead notifyDone, id: %d, unique_key: %s", id, unique_key.c_str());
                        load_context->notifyDone(id, success);
                    },
                    deadline_ms - currentTimeMs());
            }
        }
        computed_layer_cache_buffer->waitChange(total_layer_num, 50);
    }
    collector->last_layer_wait_time_us = currentTimeUs() - start_time_us;

    // write task done, remove task from store
    load_contexts_->removeContext(request_id);

    // wait until all transfers are done
    while (!load_context->isAllTransfersDone()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (metrics_reporter_) {
        collector->success            = load_context->success();
        collector->total_cost_time_us = currentTimeUs() - start_time_us;
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorServerWorkerWriteMetricsCollector>(nullptr,
                                                                                                      collector.get());
    }

    if (!load_context->success()) {
        RTP_LLM_LOG_WARNING(
            "P2PConnectorWorker handleRead failed, request_id: %ld, unique_key: %s, canceled: %d, timeout: %d",
            request_id,
            unique_key.c_str(),
            load_context->canceled(),
            load_context->timeout());
    }

    RTP_LLM_LOG_INFO("P2PConnectorWorker handleRead end, request_id: %ld, unique_key: %s, success: %d",
                     request_id,
                     unique_key.c_str(),
                     load_context->success());
    return load_context->success();
}

bool P2PConnectorWorker::read(int64_t                                               request_id,
                              const std::string&                                    unique_key,
                              int64_t                                               deadline_ms,
                              const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers) {
    RTP_LLM_LOG_INFO(
        "P2PConnectorWorker read start, request_id: %ld, unique_key: %s, deadline_ms: %ld, layer_cache_buffers_size: %zu",
        request_id,
        unique_key.c_str(),
        deadline_ms,
        layer_cache_buffers.size());
    if (!layer_cache_buffer_task_store_) {
        RTP_LLM_LOG_ERROR("P2PConnectorWorker read failed: layer_cache_buffer_task_store is null");
        return false;
    }

    if (layer_cache_buffers.empty()) {
        // empty layer cache buffers, just return true
        return true;
    }

    // 将 vector 转换为 map
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffer_map;
    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        if (layer_cache_buffer) {
            layer_cache_buffer_map[layer_cache_buffer->getLayerId()] = layer_cache_buffer;
        }
    }

    auto layer_cache_buffer_task =
        layer_cache_buffer_task_store_->addTask(unique_key, layer_cache_buffer_map, deadline_ms);
    if (!layer_cache_buffer_task) {
        RTP_LLM_LOG_WARNING("P2PConnectorWorker read failed: layer_cache_buffer_task is null");
        return false;
    }

    // wait task maybe done
    while (true) {
        // TODO: change to condwait
        if (layer_cache_buffer_task->cancelled()) {
            RTP_LLM_LOG_WARNING("task %s cancelled", unique_key.c_str());
            break;
        }
        if (layer_cache_buffer_task->isTimeout()) {
            RTP_LLM_LOG_WARNING("task %s timeout", unique_key.c_str());
            break;
        }
        if (layer_cache_buffer_task->success()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // remove task from task store, no more loading tasks
    layer_cache_buffer_task_store_->stealTask(unique_key);

    while (layer_cache_buffer_task->hasLoadingLayer()) {
        // wait till all transfer done
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (metrics_reporter_) {
        auto collector                      = std::make_shared<P2PConnectorClientWorkerMetricsCollector>();
        collector->total_block_count        = layer_cache_buffer_task->totalBlockCount();
        collector->success                  = layer_cache_buffer_task->success();
        collector->total_cost_time_us       = layer_cache_buffer_task->totalCostTimeUs();
        collector->first_layer_wait_time_us = layer_cache_buffer_task->firstLayerWaitTimeUs();
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorClientWorkerMetricsCollector>(nullptr,
                                                                                                 collector.get());
    }
    RTP_LLM_LOG_INFO("P2PConnectorWorker read end, request_id: %ld, unique_key: %s, success: %d",
                     request_id,
                     unique_key.c_str(),
                     layer_cache_buffer_task->success());
    return layer_cache_buffer_task->success();
}

}  // namespace rtp_llm