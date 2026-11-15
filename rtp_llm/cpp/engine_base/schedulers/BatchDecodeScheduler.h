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
        batch_size_ = config.batch_size_;
        if (config.mode_ == "decode") {
            scheduler_type_ = SchedulerType::kBatchDecode;
        } else if (config.mode_ == "prefill") {
            scheduler_type_ = SchedulerType::kBatchPrefill;
        }
        RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %d, mode to %d", batch_size_, int(scheduler_type_));
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
        bool                         has_execution_mode = false;
        bool                         prefill_only       = false;
        bool                         has_other_mode     = false;
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
            const auto  current             = it++;
            const auto& stream              = *current;
            const bool stream_prefill_only = isPrefillOnly(stream);
            if (!has_execution_mode) {
                has_execution_mode = true;
                prefill_only       = stream_prefill_only;
            }
            if (stream_prefill_only != prefill_only) {
                has_other_mode = true;
                continue;
            }
            if (admitting_streams_.size() < batch_size_) {
                stream->reportEvent(StreamEvents::CanRun);
                admitting_streams_.splice(admitting_streams_.end(), waiting_streams_, current);
            }
        }

        // Refresh the drain state from the current queue. This batch may drain a mixed-mode tail,
        // but later homogeneous arrivals must go through the normal batch-size gate.
        draining_mixed_modes_ = has_other_mode;
    }

    void advanceAdmission() {
        for (auto it = admitting_streams_.begin(); it != admitting_streams_.end();) {
            const auto  current      = it++;
            const auto& stream       = *current;
            const auto  before       = stream->getStatus();
            const bool  load_started = stream->hasEvent(StreamEvents::LoadInitiated);
            auto        state        = stream->moveToNext();
            // Synchronous Decode allocation first sets LoadInitiated, then enters RUNNING.
            if (state == StreamState::WAITING
                && (before == StreamState::LOADING_CACHE
                    || (!load_started && stream->hasEvent(StreamEvents::LoadInitiated)))) {
                state = stream->moveToNext();
            }
            if (state == StreamState::RUNNING) {
                running_streams_.splice(running_streams_.end(), admitting_streams_, current);
            } else if (state == StreamState::FINISHED) {
                admitting_streams_.erase(current);
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

    absl::StatusOr<ScheduleOutput> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        if (running_streams_.empty() && !admitting_streams_.empty()) {
            // KV completion and capacity changes may come without an enqueue notification.
            // Wait with the scheduler lock released and retry admission on the next pass.
            cond_.wait_for(lock, std::chrono::milliseconds(1));
        } else {
            cond_.wait_for(lock, std::chrono::seconds(30), [this] {
                return waiting_streams_.size() >= batch_size_ || !running_streams_.empty()
                       || (draining_mixed_modes_ && !waiting_streams_.empty());
            });
        }
        running_streams_.remove_if([](const auto& stream) { return stream->moveToNext() == StreamState::FINISHED; });

        if (waiting_streams_.empty()) {
            draining_mixed_modes_ = false;
        }

        if (running_streams_.empty()) {
            if (admitting_streams_.empty()) {
                evaluateWaitingStreams();
            }
            advanceAdmission();
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
        return waiting_streams_.size() + admitting_streams_.size() + running_streams_.size();
    }

private:
    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> admitting_streams_;
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
