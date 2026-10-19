#pragma once

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <list>

namespace rtp_llm {

struct BatchDecodeSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
        json.Jsonize("mode", mode_, "decode");
    }
    uint32_t    batch_size_;
    std::string mode_;
};
class BatchDecodeScheduler: public SchedulerBase {
public:
    enum SchedulerType : std::uint8_t {
        kBatchDecode  = 0,
        kBatchPrefill = 1
    };
    // evaluateWaitingStreams() 的同步 cache-load 忙等上限：合法的同步加载在秒级
    // 内完成，RETRYABLE KV 申请不会自行转错误态，只能靠这里的 bail-out 走
    // MALLOC_FAILED 错误流，避免 schedule() 持 lock_ 无限自旋。
    static constexpr std::chrono::seconds kLoadCacheBusyWaitTimeout{5};
    // 忙等超过该时长先告警一次，给 hang 判读留下调度器侧证据。
    static constexpr std::chrono::seconds kLoadCacheBusyWaitSlowLog{1};
    BatchDecodeScheduler(const RuntimeConfig&                   runtime_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter,
                         int                                    dp_rank = 0) {
        cache_manager_    = cache_manager;
        metrics_reporter_ = metrics_reporter;
        batch_size_       = runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size;
        scheduler_type_   = SchedulerType::kBatchDecode;
        dp_rank_          = dp_rank;
    }
    virtual ~BatchDecodeScheduler() = default;

    // Reject inputs longer than the KV cache can hold; mark the stream errored so the caller
    // sees the failure via collectStreamOutput / pollStreamOutput. Mirrors FIFOScheduler.
    bool checkInputLength(const GenerateStreamPtr& stream) {
        if (cache_manager_ && stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
            stream->reportError(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                                "input len " + std::to_string(stream->inputLength())
                                    + " is greater than kv cache max available tokens num "
                                    + std::to_string(cache_manager_->maxAvailableTokensNum()));
            return false;
        }
        return true;
    }

    absl::Status enqueue(const GenerateStreamPtr& stream) override {
        if (!checkInputLength(stream)) {
            return absl::InvalidArgumentError("Check input length failed");
        }
        {
            std::lock_guard<std::mutex> lock(lock_);
            waiting_streams_.emplace_back(stream);
            if (waiting_streams_.size() % 16 == 0) {
                RTP_LLM_LOG_DEBUG("BatchDecodeScheduler::enqueue: waiting_streams_.size() = %d",
                                  waiting_streams_.size());
            }
        }
        cond_.notify_all();
        return absl::OkStatus();
    }

    // Returns the input vector unchanged so callers can index 1:1 with their original list.
    // Streams that fail checkInputLength are NOT added to the waiting queue; their success flag
    // is false and their error is already reported via reportError(). No group co-scheduling:
    // valid streams are admitted as ordinary individual streams.
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

        std::vector<bool> enqueue_successes;
        enqueue_successes.reserve(streams.size());
        std::vector<GenerateStreamPtr> stream_enqueued;
        stream_enqueued.reserve(streams.size());
        for (const auto& stream : streams) {
            const bool success = checkInputLength(stream);
            enqueue_successes.push_back(success);
            if (success) {
                stream_enqueued.emplace_back(stream);
            }
        }
        {
            std::lock_guard<std::mutex> lock(lock_);
            waiting_streams_.insert(waiting_streams_.end(), stream_enqueued.begin(), stream_enqueued.end());
        }
        cond_.notify_all();
        return {std::move(enqueue_successes), streams};
    }

    void updateSchedulerInfo(const std::string& scheduler_info) override {
        BatchDecodeSchedulerConfigLocal config;
        autil::legacy::FromJsonString(config, scheduler_info);
        batch_size_ = config.batch_size_;
        if (config.mode_ == "decode") {
            scheduler_type_ = SchedulerType::kBatchDecode;
        } else if (config.mode_ == "prefill") {
            scheduler_type_ = SchedulerType::kBatchPrefill;
        }
        RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %d, mode to %d", batch_size_, int(scheduler_type_));
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
                break;
            default:
                RTP_LLM_LOG_ERROR(
                    "Unknown state: %d for stream [%ld]", static_cast<int>(new_state), stream->streamId());
                break;
        }
    }

    // 通过 GenerateStateMachine 驱动每个 stream 的状态转移，状态变化的 stream 移入对应队列
    void evaluateAndUpdateStreams(std::list<GenerateStreamPtr>& streams) {
        for (auto it = streams.begin(); it != streams.end();) {
            auto state     = (*it)->getStatus();
            auto new_state = (*it)->moveToNext();
            if (new_state != state) {
                addStreamToNewState(*it, new_state);
                it = streams.erase(it);
            } else {
                it++;
            }
        }
    }

    void evaluateWaitingStreams() {
        // Finalize errored streams before selecting a batch so they cannot count toward its size.
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
            if (!(*it)->hasError()) {
                ++it;
                continue;
            }
            (*it)->moveToNext();
            it = waiting_streams_.erase(it);
        }

        // The outer gate counted entries that may just have been removed. Only an already-draining
        // mixed-mode tail may bypass the batch-size gate after errors are reaped.
        if (waiting_streams_.size() < batch_size_ && !draining_mixed_modes_) {
            return;
        }

        // Homogeneous queues run exactly batch_size_ streams; a mixed queue may yield a smaller
        // mode-homogeneous batch and drain its remaining tail in subsequent rounds.
        std::list<GenerateStreamPtr> selected_streams;
        bool                         has_execution_mode = false;
        bool                         prefill_only       = false;
        bool                         has_other_mode     = false;
        for (const auto& stream : waiting_streams_) {
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

        // Refresh the drain state from the current queue. This batch may drain a mixed-mode tail,
        // but later homogeneous arrivals must go through the normal batch-size gate.
        draining_mixed_modes_ = has_other_mode;
        for (auto& stream : selected_streams) {
            stream->reportEvent(StreamEvents::CanRun);
            // 忙等stream load cache done, 和原有SyncLoadCache逻辑等效。
            // RETRYABLE 的 KV 申请会让 stream 停留在 WAITING（无错误状态转移），
            // 无 bail-out 时这里会持 lock_ 无限自旋并卡死整个 engine loop
            // （enqueue/stop 全被锁住）；超过 5s 转 MALLOC_FAILED 错误流。
            const auto wait_begin = std::chrono::steady_clock::now();
            bool      warned_slow = false;
            while (stream->getStatus() != StreamState::FINISHED && stream->moveToNext() != StreamState::RUNNING) {
                const auto waited = std::chrono::steady_clock::now() - wait_begin;
                if (waited >= kLoadCacheBusyWaitTimeout) {
                    RTP_LLM_LOG_ERROR(
                        "BatchDecodeScheduler busy-wait for stream [%ld] timed out after %llds (status=%d); "
                        "marking MALLOC_FAILED so the engine loop can keep scheduling",
                        stream->streamId(),
                        static_cast<long long>(kLoadCacheBusyWaitTimeout.count()),
                        static_cast<int>(stream->getStatus()));
                    stream->reportError(ErrorCode::MALLOC_FAILED,
                                        "BatchDecodeScheduler cache-load busy-wait timed out after "
                                            + std::to_string(kLoadCacheBusyWaitTimeout.count())
                                            + "s (retryable KV malloc did not resolve)");
                    (void)stream->moveToNext();  // Error 事件优先: -> FINISHED 并释放资源
                    break;
                }
                if (!warned_slow && waited >= kLoadCacheBusyWaitSlowLog) {
                    warned_slow = true;
                    RTP_LLM_LOG_WARNING("BatchDecodeScheduler busy-wait for stream [%ld] exceeded %llds (status=%d)",
                                        stream->streamId(),
                                        static_cast<long long>(kLoadCacheBusyWaitSlowLog.count()),
                                        static_cast<int>(stream->getStatus()));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        for (const auto& stream : selected_streams) {
            waiting_streams_.remove(stream);
            if (stream->getStatus() == StreamState::RUNNING) {
                running_streams_.push_back(stream);
            }
        }
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
            return waiting_streams_.size() >= batch_size_ || running_streams_.size() > 0
                   || !loading_cache_streams_.empty() || (draining_mixed_modes_ && !waiting_streams_.empty());
        });

        // 统一通过状态机驱动各队列中 stream 的状态转移
        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        evaluateAndUpdateStreams(running_streams_);

        if (waiting_streams_.empty()) {
            draining_mixed_modes_ = false;
        }

        if (running_streams_.empty()
            && (waiting_streams_.size() >= batch_size_ || (draining_mixed_modes_ && !waiting_streams_.empty()))) {
            evaluateWaitingStreams();
            if (!running_streams_.empty()) {
                initRunningStreams();
                RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run",
                                 running_streams_.size());
            }
        }

        return running_streams_;
    }

    absl::Status stop() override {
        // Not implemented
        return absl::UnimplementedError("BatchDecodeScheduler::stop not implemented");
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

private:
    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> loading_cache_streams_;
    std::list<GenerateStreamPtr> running_streams_;
    uint32_t                     batch_size_;
    bool                         reorder_request_;
    uint32_t                     current_step_ = 0;
    bool                         draining_mixed_modes_ = false;

    std::shared_ptr<KVCacheManager> cache_manager_;
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    SchedulerType                   scheduler_type_;
    int                             dp_rank_ = 0;
};

}  // namespace rtp_llm
