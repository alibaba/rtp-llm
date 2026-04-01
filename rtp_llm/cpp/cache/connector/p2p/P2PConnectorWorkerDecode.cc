#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace rtp_llm {

P2PConnectorWorkerDecode::P2PConnectorWorkerDecode(P2PConnectorWorkerConfig                    config,
                                                   const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                                   const kmonitor::MetricsReporterPtr&         metrics_reporter,
                                                   const transfer::IKVCacheReceiverPtr&        receiver):
    config_(std::move(config)),
    layer_block_converter_(layer_block_converter),
    metrics_reporter_(metrics_reporter),
    receiver_(receiver) {}

ErrorInfo
P2PConnectorWorkerDecode::buildRecvTasks(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                         int                                                   recv_partition_count,
                                         const std::string&                                    unique_key,
                                         int64_t                                               deadline_ms,
                                         const std::shared_ptr<ReadTaskGroup>&                 task_group,
                                         int& total_block_count) const {
    // 与 return_deadline_ms（D - p2p_read_return_before_deadline_ms）对齐，TransferTask / TCP 侧 isTimeout 与 worker
    // 必须结束 read 的时刻一致。
    const int64_t recv_task_deadline_ms = deadline_ms - config_.p2p_read_return_before_deadline_ms;

    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        if (!layer_cache_buffer) {
            continue;
        }
        const int layer_id = layer_cache_buffer->getLayerId();

        for (int partition_id = 0; partition_id < recv_partition_count; ++partition_id) {
            auto key_block_infos = LayerCacheBufferUtil::buildKeyBlockInfos(
                layer_block_converter_, layer_cache_buffer, recv_partition_count, partition_id);

            const std::string partition_layer_key =
                P2PKeyUtil::makePartitionLayerKey(unique_key, layer_id, partition_id);

            transfer::RecvRequest recv_req;
            recv_req.unique_key  = partition_layer_key;
            recv_req.block_info  = std::move(key_block_infos);
            recv_req.deadline_ms = recv_task_deadline_ms;

            auto task = receiver_->recv(recv_req);
            if (!task) {
                const std::string error_msg = "read: create recv task failed for layer=" + std::to_string(layer_id)
                                              + " partition=" + std::to_string(partition_id)
                                              + " unique_key=" + unique_key;
                RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
                return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, error_msg);
            }
            task_group->partition_keys.push_back(partition_layer_key);
            task_group->tasks.push_back(task);
            total_block_count += static_cast<int>(layer_cache_buffer->blockIdMap().size());
        }
    }
    return ErrorInfo::OkStatus();
}

namespace {
constexpr int kBackoffInitialMs = 1;
constexpr int kBackoffCapMs     = 8;
}  // namespace

P2PConnectorWorkerDecode::ReadWaitOutcome
P2PConnectorWorkerDecode::waitRecvTasksWithReadDeadlinePolicy(const std::shared_ptr<ReadTaskGroup>& task_group,
                                                              int64_t                               deadline_ms,
                                                              int64_t                               request_id,
                                                              const std::string&                    unique_key) const {
    // deadline_ms：scheduler 下发的绝对时间戳（ms），与 currentTimeMs() 同单位。
    // return_deadline_ms = D - return_before_ms：须在此刻前结束等待并向 scheduler 返回 RPC，为链路留出 return_before_ms
    // 余量。 steal_deadline_ms：到达 D - steal_before_ms 时从 recv store steal 各 partition key，阻止后续 sender
    // 再匹配到新传输； 与 return 线取 min，避免配置错误时 steal 时刻晚于必须返回时刻。
    const int64_t steal_before_ms  = config_.p2p_read_steal_before_deadline_ms;
    const int64_t return_before_ms = config_.p2p_read_return_before_deadline_ms;

    const int64_t return_deadline_ms = deadline_ms - return_before_ms;
    const int64_t steal_deadline_ms  = std::min(deadline_ms - steal_before_ms, return_deadline_ms);

    bool store_stolen = false;
    int  sleep_ms     = kBackoffInitialMs;

    // 退避轮询：cancel / 全部 task done 则返回；否则先 steal（仅一次）再判断是否到达 return 截止；
    // ReturnDeadlineIncomplete 时由 read() 返回 TRANSFER_NOT_DONE，不对未完成 task forceCancel。
    while (true) {
        if (task_group->cancelled.load()) {
            return ReadWaitOutcome::Cancelled;
        }
        bool all_tasks_done = true;
        for (const auto& task : task_group->tasks) {
            if (!task->done()) {
                all_tasks_done = false;
                break;
            }
        }
        if (all_tasks_done) {
            return ReadWaitOutcome::AllDone;
        }

        const int64_t now = currentTimeMs();
        if (now >= steal_deadline_ms && !store_stolen) {
            for (const auto& key : task_group->partition_keys) {
                receiver_->stealTask(key);
            }
            store_stolen = true;
            RTP_LLM_LOG_DEBUG(
                "read: stole recv tasks from store at steal_deadline_ms=%ld, request_id=%ld, unique_key=%s",
                steal_deadline_ms,
                request_id,
                unique_key.c_str());
        }
        if (now >= return_deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "read: return deadline reached (D-%ld ms) with pending transfers, request_id=%ld, unique_key=%s",
                return_before_ms,
                request_id,
                unique_key.c_str());
            return ReadWaitOutcome::ReturnDeadlineIncomplete;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        sleep_ms = std::min(sleep_ms * 2, kBackoffCapMs);
    }
}

void P2PConnectorWorkerDecode::reportReadMetrics(int     total_block_count,
                                                 bool    success,
                                                 int64_t read_start_time_us) const {
    if (!metrics_reporter_) {
        return;
    }
    auto collector                      = std::make_shared<DecodeWorkerMetricsCollector>();
    collector->total_block_count        = total_block_count;
    collector->success                  = success;
    collector->total_cost_time_us       = currentTimeUs() - read_start_time_us;
    collector->first_layer_wait_time_us = 0;
    metrics_reporter_->report<P2PConnectorMetrics, DecodeWorkerMetricsCollector>(nullptr, collector.get());
}

ErrorInfo P2PConnectorWorkerDecode::read(int64_t                                               request_id,
                                         const std::string&                                    unique_key,
                                         int64_t                                               deadline_ms,
                                         const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                         int                                                   remote_tp_size) {
    int recv_partition_count = calculateRecvPartitionCount(remote_tp_size);

    RTP_LLM_LOG_DEBUG(
        "read start, request_id: %ld, unique_key: %s, deadline_ms: %ld, layers: %zu, remote_tp_size: %d, recv_partition_count: %d",
        request_id,
        unique_key.c_str(),
        deadline_ms,
        layer_cache_buffers.size(),
        remote_tp_size,
        recv_partition_count);

    if (layer_cache_buffers.empty()) {
        return ErrorInfo::OkStatus();
    }

    const int64_t read_start_time_us = currentTimeUs();
    auto          task_group         = std::make_shared<ReadTaskGroup>();
    int           total_block_count  = 0;

    ErrorInfo build_result = buildRecvTasks(
        layer_cache_buffers, recv_partition_count, unique_key, deadline_ms, task_group, total_block_count);
    if (build_result.hasError()) {
        return build_result;
    }

    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        read_tasks_[unique_key] = task_group;
    }

    const ReadWaitOutcome outcome =
        waitRecvTasksWithReadDeadlinePolicy(task_group, deadline_ms, request_id, unique_key);

    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        read_tasks_.erase(unique_key);
    }

    if (outcome == ReadWaitOutcome::ReturnDeadlineIncomplete) {
        reportReadMetrics(total_block_count, false, read_start_time_us);
        const std::string msg = "read: transfers not all done before return deadline (D-"
                                + std::to_string(config_.p2p_read_return_before_deadline_ms) + "ms)";
        RTP_LLM_LOG_WARNING(
            "read failed, request_id: %ld, unique_key: %s, %s", request_id, unique_key.c_str(), msg.c_str());
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE, msg);
    }

    auto recv_result = aggregateRecvTaskResults(task_group);

    reportReadMetrics(total_block_count, recv_result.success, read_start_time_us);

    if (!recv_result.success) {
        RTP_LLM_LOG_WARNING("read failed, request_id: %ld, unique_key: %s, error_code: %s, error_msg: %s",
                            request_id,
                            unique_key.c_str(),
                            ErrorCodeToString(recv_result.error_code).c_str(),
                            recv_result.error_msg.c_str());
        return ErrorInfo(recv_result.error_code, recv_result.error_msg);
    }

    RTP_LLM_LOG_DEBUG("read end, request_id: %ld, unique_key: %s, success: true", request_id, unique_key.c_str());
    return ErrorInfo::OkStatus();
}

P2PConnectorWorkerDecode::RecvResultInfo
P2PConnectorWorkerDecode::aggregateRecvTaskResults(const std::shared_ptr<ReadTaskGroup>& task_group) const {
    RecvResultInfo result;
    for (const auto& task : task_group->tasks) {
        if (!task->success()) {
            result.success = false;
            if (result.error_code == ErrorCode::NONE_ERROR) {
                result.error_code = transfer::toErrorCode(task->errorCode());
                result.error_msg  = task->errorMessage();
            }
        }
    }
    if (task_group->cancelled.load()) {
        result.success    = false;
        result.error_code = ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
        result.error_msg  = "read cancelled";
        return result;
    }
    if (!result.success && result.error_code == ErrorCode::NONE_ERROR) {
        result.error_code = ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED;
    }
    return result;
}

int P2PConnectorWorkerDecode::calculateRecvPartitionCount(int remote_tp_size) const {
    if (remote_tp_size <= 0 || config_.tp_size <= 0) {
        return 1;
    }
    return std::max(1, remote_tp_size / static_cast<int>(config_.tp_size));
}

bool P2PConnectorWorkerDecode::cancelRead(const std::string& unique_key) {
    RTP_LLM_LOG_DEBUG("cancelRead start, unique_key: %s", unique_key.c_str());
    std::shared_ptr<ReadTaskGroup> task_group;
    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        auto                        it = read_tasks_.find(unique_key);
        if (it == read_tasks_.end()) {
            RTP_LLM_LOG_INFO("cancelRead: task not found, unique_key: %s", unique_key.c_str());
            return false;
        }
        task_group = it->second;
    }

    task_group->cancelled.store(true);
    for (const auto& task : task_group->tasks) {
        task->cancel();
    }
    RTP_LLM_LOG_DEBUG("cancelRead success, unique_key: %s", unique_key.c_str());
    return true;
}

}  // namespace rtp_llm
