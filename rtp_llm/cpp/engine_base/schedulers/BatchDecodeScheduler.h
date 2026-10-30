#pragma once

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <atomic>
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
    BatchDecodeScheduler(const RuntimeConfig&                   runtime_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter,
                         int                                    dp_rank              = 0,
                         int64_t                                max_execution_tokens = 0,
                         const ParallelismConfig&               parallelism_config   = ParallelismConfig{},
                         const ModelExecutionCapacity&          execution_capacity   = ModelExecutionCapacity{}) {
        cache_manager_    = cache_manager;
        metrics_reporter_ = metrics_reporter;
        batch_size_       = runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size;
        max_batch_size_           = std::max<int64_t>(runtime_config.max_generate_batch_size, batch_size_);
        const auto tp             = std::max<int64_t>(parallelism_config.tp_size, 1);
        max_prefill_capacity_     = max_execution_tokens > 0 ? max_execution_tokens / tp * tp : 0;
        max_prefill_draft_tokens_ = execution_capacity.max_prefill_draft_tokens;
        draft_prefill_cp_size_    = execution_capacity.draft_prefill_cp_size;
        target_only_runs_draft_   = execution_capacity.target_only_runs_draft;
        prefill_cp_size_          = parallelism_config.prefill_cp_config.is_enabled() ?
                                        static_cast<size_t>(std::max<int64_t>(parallelism_config.tp_size, 1)) :
                                        1;
        scheduler_type_   = SchedulerType::kBatchDecode;
        dp_rank_          = dp_rank;
    }
    virtual ~BatchDecodeScheduler() = default;

    // Reject inputs longer than the KV cache can hold; mark the stream errored so the caller
    // sees the failure via collectStreamOutput / pollStreamOutput. Mirrors FIFOScheduler.
    bool checkInputLength(const GenerateStreamPtr& stream) {
        if (static_cast<size_t>(stream->maxBatchSize()) > max_batch_size_) {
            stream->reportError(ErrorCode::INVALID_PARAMS,
                                "request sequences exceed initialized batch capacity "
                                    + std::to_string(max_batch_size_));
            return false;
        }
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
        if (!stream_enqueued.empty()) {
            // One timestamp per group, mirroring FIFOSchedulerBase::enqueueGroup: publishes the
            // recorded request identity before the streams become visible to the scheduler loop.
            const auto enqueue_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            for (auto& stream : stream_enqueued) {
                stream->recordSchedulerEnqueueTime(enqueue_time_us);
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
        if (config.batch_size_ == 0 || config.batch_size_ > max_batch_size_) {
            RTP_LLM_LOG_WARNING(
                "reject batch size %u outside initialized capacity [1, %u]", config.batch_size_, max_batch_size_);
            return;
        }
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
        // A runtime batch reduction must not leave an oversized request waiting forever.
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
            if (!(*it)->hasError() && static_cast<size_t>((*it)->maxBatchSize()) > batch_size_) {
                (*it)->reportError(ErrorCode::INVALID_PARAMS,
                                   "request sequences exceed current fixed batch size " + std::to_string(batch_size_));
            }
            if (!(*it)->hasError()) {
                ++it;
                continue;
            }
            (*it)->moveToNext();
            it = waiting_streams_.erase(it);
        }

        if (!waitingBatchReady()) {
            return;
        }

        // Count expanded sequences, including the maximum later beam width.
        std::list<GenerateStreamPtr> selected_streams;
        size_t                       selected_sequences = 0;
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
            const auto sequences = static_cast<size_t>(stream->maxBatchSize());
            if (sequences <= batch_size_ - selected_sequences) {
                selected_streams.push_back(stream);
                selected_sequences += sequences;
            }
        }

        // Refresh the drain state from the current queue. This batch may drain a mixed-mode tail,
        // but later homogeneous arrivals must go through the normal batch-size gate.
        draining_mixed_modes_ = has_other_mode;
        for (const auto& stream : selected_streams) {
            if (!stream->streamCacheResource().reserveLinearReplaySlot()) {
                return;
            }
        }
        for (auto& stream : selected_streams) {
            if (scheduler_type_ == SchedulerType::kBatchDecode && !isPrefillOnly(stream)
                && cache_manager_->cacheConfig().linear_replay.enabled()) {
                // Direct Decode starts from the prompt's final token, with its
                // already-processed prefix initialized before the first grow.
                stream->setPerfTest(true);
                stream->setIsContextStream(false);
            }
            stream->reportEvent(StreamEvents::CanRun);
            // 忙等stream load cache done, 和原有SyncLoadCache逻辑等效
            while (stream->getStatus() != StreamState::FINISHED && stream->moveToNext() != StreamState::RUNNING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        for (const auto& stream : selected_streams) {
            const auto state = stream->getStatus();
            if (state == StreamState::RUNNING) {
                waiting_streams_.remove(stream);
                running_streams_.push_back(stream);
            } else if (state == StreamState::FINISHED) {
                waiting_streams_.remove(stream);
            }
        }
    }

    void initRunningStreams() {
        // set kvcache block
        for (auto it = running_streams_.begin(); it != running_streams_.end();) {
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
            if ((*it)->getStatus() == StreamState::FINISHED) {
                it = running_streams_.erase(it);
            } else {
                ++it;
            }
        }
        size_t  sequences      = 0;
        int64_t prefill_tokens = 0, draft_tokens = 0;
        for (const auto& stream : running_streams_) {
            const auto stream_sequences = static_cast<size_t>(stream->maxBatchSize());
            bool       fits             = stream_sequences <= max_batch_size_ - sequences;
            if (fits) {
                sequences += stream_sequences;
            }
            if (fits && stream->isContextStream()) {
                if (max_prefill_capacity_ > 0) {
                    const auto tokens = prefillTokenCostWithoutCache(stream, prefill_cp_size_);
                    fits              = tokens <= static_cast<size_t>(max_prefill_capacity_ - prefill_tokens);
                    if (fits) {
                        prefill_tokens += tokens;
                    }
                }
                if (fits && max_prefill_draft_tokens_ > 0 && (!isPrefillOnly(stream) || target_only_runs_draft_)) {
                    const auto tokens = prefillTokenCostWithoutCache(stream, draft_prefill_cp_size_);
                    fits              = tokens <= static_cast<size_t>(max_prefill_draft_tokens_ - draft_tokens);
                    if (fits) {
                        draft_tokens += tokens;
                    }
                }
            }
            if (!fits) {
                for (const auto& rejected : running_streams_) {
                    rejected->reportError(ErrorCode::INVALID_PARAMS,
                                          "fixed batch exceeds initialized model execution capacity");
                    rejected->moveToNext();
                }
                running_streams_.clear();
                return;
            }
        }
    }

    absl::StatusOr<ScheduleOutput> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        cond_.wait_for(lock, std::chrono::seconds(30), [this] {
            return waitingBatchReady() || !running_streams_.empty() || !loading_cache_streams_.empty();
        });

        // 统一通过状态机驱动各队列中 stream 的状态转移
        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        evaluateAndUpdateStreams(running_streams_);

        if (waiting_streams_.empty()) {
            draining_mixed_modes_ = false;
        }

        if (running_streams_.empty() && waitingBatchReady()) {
            evaluateWaitingStreams();
            if (!running_streams_.empty()) {
                initRunningStreams();
                RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run",
                                 running_streams_.size());
            }
        }

        return ScheduleOutput{running_streams_};
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
    bool waitingBatchReady() const {
        size_t sequences = 0;
        for (const auto& stream : waiting_streams_) {
            const auto count = static_cast<size_t>(stream->maxBatchSize());
            if (stream->hasError() || count >= batch_size_ - sequences) {
                return true;
            }
            sequences += count;
        }
        return draining_mixed_modes_ && !waiting_streams_.empty();
    }

    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> loading_cache_streams_;
    std::list<GenerateStreamPtr> running_streams_;
    uint32_t                     batch_size_;
    uint32_t                     max_batch_size_;
    int64_t                      max_prefill_capacity_     = 0;
    int64_t                      max_prefill_draft_tokens_ = 0;
    bool                         target_only_runs_draft_   = false;
    size_t                       prefill_cp_size_          = 1;
    size_t                       draft_prefill_cp_size_    = 1;
    bool                         reorder_request_;
    uint32_t                     current_step_ = 0;
    bool                         draining_mixed_modes_ = false;

    std::shared_ptr<KVCacheManager> cache_manager_;
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    SchedulerType                   scheduler_type_;
    int                             dp_rank_ = 0;
};

}  // namespace rtp_llm
