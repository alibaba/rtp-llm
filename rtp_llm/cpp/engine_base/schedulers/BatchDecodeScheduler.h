#pragma once

#include "autil/legacy/jsonizable.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <list>
#include <memory>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace rtp_llm {

struct BatchDecodeSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
        json.Jsonize("mode", mode_, "decode");
    }
    int64_t     batch_size_ = 0;
    std::string mode_;
};
class BatchDecodeScheduler: public SchedulerBase {
public:
    enum SchedulerType : std::uint8_t {
        kBatchDecode  = 0,
        kBatchPrefill = 1
    };
    BatchDecodeScheduler(const RuntimeConfig&                   runtime_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter,
                         int                                    dp_rank = 0) {
        cache_manager_    = cache_manager;
        metrics_reporter_ = metrics_reporter;
        batch_size_       = checkedBatchSize(runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size);
        scheduler_type_   = runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type == 0 ?
                                SchedulerType::kBatchDecode : SchedulerType::kBatchPrefill;
        dp_rank_          = dp_rank;
        perf_local_requests_ = runtime_config.use_batch_decode_scheduler
                               && autil::EnvUtil::getEnv("RTP_LLM_PERF_LOCAL_REQUESTS", false);
        const char* env   = std::getenv("RTP_LLM_WORKER_STATUS_SNAPSHOT");
        worker_status_snapshot_enabled_ = env != nullptr && std::strcmp(env, "1") == 0;
        publishWorkerStatusSnapshotLocked();
    }
    virtual ~BatchDecodeScheduler() = default;

    absl::Status enqueue(const GenerateStreamPtr& stream) override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            if (stop_) {
                stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
                stream->moveToNext();
                return absl::CancelledError("scheduler stopped");
            }
            stream->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds());
            waiting_streams_.emplace_back(stream);
            if (waiting_streams_.size() % 16 == 0) {
                RTP_LLM_LOG_DEBUG("BatchDecodeScheduler::enqueue: waiting_streams_.size() = %d",
                                  waiting_streams_.size());
            }
        }
        cond_.notify_all();
        return absl::OkStatus();
    }

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override {
        if (hasMixedExecutionModes(streams)) {
            for (const auto& stream : streams) {
                if (!stream->hasError()) {
                    stream->reportError(ErrorCode::INVALID_PARAMS, kMixedForceBatchGroupError);
                }
            }
            return {std::vector<bool>(streams.size(), false), streams};
        }

        for (const auto& stream : streams) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "BatchDecodeScheduler::enqueueGroup is not implemented");
        }
        return {std::vector<bool>(streams.size(), false), streams};  // Not implemented for BatchDecodeScheduler
    }

    void updateSchedulerInfo(const std::string& scheduler_info) override {
        BatchDecodeSchedulerConfigLocal config;
        autil::legacy::FromJsonString(config, scheduler_info);
        const auto batch_size = checkedBatchSize(config.batch_size_);
        {
            std::lock_guard<std::mutex> lock(lock_);
            batch_size_ = batch_size;
            if (config.mode_ == "decode") {
                scheduler_type_ = SchedulerType::kBatchDecode;
            } else if (config.mode_ == "prefill") {
                scheduler_type_ = SchedulerType::kBatchPrefill;
            }
            RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %d, mode to %d", batch_size_, int(scheduler_type_));
        }
        cond_.notify_all();
    }

    // 根据状态机转移后的目标状态，将 stream 路由到对应的队列
    void addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
        switch (new_state) {
            case StreamState::WAITING:
                waiting_streams_.push_back(stream);
                break;
            case StreamState::LOADING_CACHE:
                loading_cache_streams_.push_back(stream);
                break;
            case StreamState::RUNNING:
                running_streams_.push_back(stream);
                break;
            case StreamState::FINISHED:
                mixed_mode_cohort_.erase(stream);
                break;
            default:
                RTP_LLM_LOG_ERROR(
                    "Unknown state: %d for stream [%ld]", static_cast<int>(new_state), stream->streamId());
                break;
        }
    }

    // 通过 GenerateStateMachine 驱动每个 stream 的状态转移，状态变化的 stream 移入对应队列
    bool evaluateAndUpdateStreams(std::list<GenerateStreamPtr>& streams) {
        bool changed = false;
        for (auto it = streams.begin(); it != streams.end();) {
            auto state     = (*it)->getStatus();
            auto new_state = (*it)->moveToNext();
            if (new_state != state) {
                addStreamToNewState(*it, new_state);
                it      = streams.erase(it);
                changed = true;
            } else {
                it++;
            }
        }
        return changed;
    }

    void cancelStreams(std::list<GenerateStreamPtr>& streams) {
        for (auto& stream : streams) {
            stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
            stream->moveToNext();
        }
        streams.clear();
    }

    bool hasMixedModeTail() const {
        return std::any_of(waiting_streams_.begin(), waiting_streams_.end(), [this](const auto& stream) {
            return mixed_mode_cohort_.count(stream) != 0;
        });
    }

    bool evaluateWaitingStreams() {
        // Finalize errored streams before selecting a batch so they cannot count toward its size.
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
            if (!(*it)->hasError()) {
                ++it;
                continue;
            }
            (*it)->moveToNext();
            mixed_mode_cohort_.erase(*it);
            it = waiting_streams_.erase(it);
        }

        // The outer gate counted entries that may just have been removed. Only an already-draining
        // mixed-mode tail may bypass the batch-size gate after errors are reaped.
        const bool draining_mixed_modes = hasMixedModeTail();
        if (waiting_streams_.size() < batch_size_ && !draining_mixed_modes) {
            return false;
        }

        // Homogeneous queues run exactly batch_size_ streams; a mixed queue may yield a smaller
        // mode-homogeneous batch and drain its remaining tail in subsequent rounds.
        std::list<GenerateStreamPtr> selected_streams;
        bool                         has_execution_mode = false;
        bool                         prefill_only       = false;
        bool                         has_other_mode     = false;
        for (const auto& stream : waiting_streams_) {
            if (draining_mixed_modes && mixed_mode_cohort_.count(stream) == 0) {
                continue;
            }
            const bool stream_prefill_only = isPrefillOnly(stream);
            if (!has_execution_mode) {
                has_execution_mode = true;
                prefill_only       = stream_prefill_only;
            }
            if (stream_prefill_only != prefill_only) {
                has_other_mode = true;
                continue;
            }
            if (selected_streams.size() < batch_size_) {
                selected_streams.push_back(stream);
            }
        }

        // Freeze the original mixed queue, not just the next batch. Retain membership until
        // termination so cache reloads can rejoin the drain, but later arrivals cannot extend it.
        if (!draining_mixed_modes && has_other_mode) {
            mixed_mode_cohort_.insert(waiting_streams_.begin(), waiting_streams_.end());
        }
        for (auto& stream : selected_streams) {
            stream->reportEvent(StreamEvents::CanRun);
            if (perf_local_requests_ && scheduler_type_ == SchedulerType::kBatchDecode && !isPrefillOnly(stream)
                && stream->resourceContext().role_type == RoleType::DECODE) {
                // No prefill peer exists in this synthetic perf path. Use real
                // blocks so the normal state machine can grow/release them.
                if (!perf_cache_initialized_) {
                    for (auto& buffer : cache_manager_->gpuCacheTensors()) {
                        buffer.zero_();
                        c10::impl::VirtualGuardImpl(buffer.device().type())
                            .getStream(buffer.device()).synchronize();
                    }
                    perf_cache_initialized_ = true;
                }
                stream->setPerfTest(true);
                stream->setIsContextStream(false);
                const auto status = stream->initKVBlock();
                if (status.ok()) {
                    stream->reportEvent(StreamEvents::LoadInitiated);
                    stream->reportEvent(StreamEvents::CachePrepared);
                } else {
                    stream->reportError(ErrorCode::MALLOC_FAILED, std::string(status.message()));
                }
            }
            // 忙等stream load cache done, 和原有SyncLoadCache逻辑等效
            while (stream->getStatus() != StreamState::FINISHED && stream->moveToNext() != StreamState::RUNNING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        for (const auto& stream : selected_streams) {
            waiting_streams_.remove(stream);
            if (stream->getStatus() == StreamState::RUNNING) {
                running_streams_.push_back(stream);
            } else if (stream->getStatus() == StreamState::FINISHED) {
                mixed_mode_cohort_.erase(stream);
            }
        }
        return !selected_streams.empty();
    }

    void initRunningStreams() {
        // set kvcache block
        for (auto it = running_streams_.begin(); it != running_streams_.end(); it++) {
            (*it)->setPerfTest(true);
            // reset start time，to get more accurate avg token time
            (*it)->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());
            // only set gen_timeline = True for first rank
            if (dp_rank_ != 0) {
                (*it)->setGenTimeline(false);
            }
            if (scheduler_type_ == SchedulerType::kBatchDecode && !isPrefillOnly(*it)) {
                (*it)->setIsContextStream(false);
                // for linear attn, incrKVBlock to clear unused linear block
                (*it)->moveToNext();
            }
        }
    }

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        cond_.wait_for(lock, std::chrono::seconds(30), [this] {
            return stop_ || waiting_streams_.size() >= batch_size_ || running_streams_.size() > 0
                   || !loading_cache_streams_.empty() || hasMixedModeTail();
        });
        if (stop_) {
            return std::list<GenerateStreamPtr>{};
        }

        // 统一通过状态机驱动各队列中 stream 的状态转移
        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        bool running_streams_changed = evaluateAndUpdateStreams(running_streams_);

        if (running_streams_.empty() && (waiting_streams_.size() >= batch_size_ || hasMixedModeTail())) {
            running_streams_changed = evaluateWaitingStreams() || running_streams_changed;
            if (!running_streams_.empty()) {
                initRunningStreams();
                RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run",
                                 running_streams_.size());
            }
        }

        if (running_streams_changed) {
            publishWorkerStatusSnapshotLocked();
        }
        return running_streams_;
    }

    absl::Status stop() override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            stop_ = true;
            cancelStreams(waiting_streams_);
            cancelStreams(loading_cache_streams_);
            cancelStreams(running_streams_);
            mixed_mode_cohort_.clear();
            publishWorkerStatusSnapshotLocked();
        }
        cond_.notify_all();
        return absl::OkStatus();
    }

    bool empty() override {
        // Not implemented
        return true;  // 默认返回值
    }

    int64_t lastScheduleTime() override {
        return 0;  // 默认返回值
    }

    int64_t onflightStreams() override {
        std::lock_guard<std::mutex> lock(lock_);
        return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size();
    }

    std::shared_ptr<const std::unordered_set<int64_t>> workerStatusRunningTaskIdsSnapshot() const override {
        if (!worker_status_snapshot_enabled_) {
            return nullptr;
        }
        return std::atomic_load_explicit(&worker_status_running_task_ids_snapshot_, std::memory_order_acquire);
    }

private:
    static uint32_t checkedBatchSize(int64_t batch_size) {
        if (batch_size <= 0 || batch_size > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("BatchDecodeScheduler batch_size must be in [1, UINT32_MAX]");
        }
        return static_cast<uint32_t>(batch_size);
    }

    void publishWorkerStatusSnapshotLocked() {
        if (!worker_status_snapshot_enabled_) {
            return;
        }
        auto snapshot = std::make_shared<std::unordered_set<int64_t>>();
        snapshot->reserve(running_streams_.size());
        for (const auto& stream : running_streams_) {
            snapshot->insert(stream->streamId());
        }
        std::atomic_store_explicit(&worker_status_running_task_ids_snapshot_,
                                   std::shared_ptr<const std::unordered_set<int64_t>>(std::move(snapshot)),
                                   std::memory_order_release);
    }

    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> loading_cache_streams_;
    std::list<GenerateStreamPtr> running_streams_;
    bool                         stop_ = false;
    uint32_t                     batch_size_;
    bool                         reorder_request_;
    uint32_t                     current_step_         = 0;
    std::unordered_set<GenerateStreamPtr> mixed_mode_cohort_;

    std::shared_ptr<KVCacheManager>                    cache_manager_;
    kmonitor::MetricsReporterPtr                       metrics_reporter_;
    SchedulerType                                      scheduler_type_;
    int                                                dp_rank_                        = 0;
    bool                                               worker_status_snapshot_enabled_ = false;
    bool                                               perf_local_requests_ = false;
    bool                                               perf_cache_initialized_ = false;
    std::shared_ptr<const std::unordered_set<int64_t>> worker_status_running_task_ids_snapshot_;
};

}  // namespace rtp_llm
