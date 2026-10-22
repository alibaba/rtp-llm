#pragma once

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <atomic>
#include <map>
#include <mutex>
#include <condition_variable>
#include <list>
#include <unordered_set>

namespace rtp_llm {

struct BatchDecodeSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
        json.Jsonize("mode", mode_, "decode");
        json.Jsonize("require_full_batch", require_full_batch_, false);
    }
    uint32_t    batch_size_;
    std::string mode_;
    bool        require_full_batch_ = false;
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
        require_full_batch_ = config.require_full_batch_;
        if (config.mode_ == "decode") {
            scheduler_type_ = SchedulerType::kBatchDecode;
        } else if (config.mode_ == "prefill") {
            scheduler_type_ = SchedulerType::kBatchPrefill;
        }
        RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %d, mode to %d, require_full_batch=%d",
                         batch_size_, int(scheduler_type_), int(config.require_full_batch_));
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
        // Finalize errors before selecting; cancelled streams must release their resources.
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
            if ((*it)->hasError()) {
                (*it)->moveToNext();
                it = waiting_streams_.erase(it);
            } else if ((*it)->getStatus() == StreamState::FINISHED) {
                it = waiting_streams_.erase(it);
            } else {
                ++it;
            }
        }
        draining_mixed_modes_ = false;
        if (waiting_streams_.empty()) {
            return;
        }

        // Preserve the main prefill/decode boundary. Only the oldest execution mode
        // participates in this round's Native ReturnAllProbs grouping and FIFO policy.
        const bool prefill_only = isPrefillOnly(waiting_streams_.front());
        std::map<ReturnAllProbsMode, std::list<GenerateStreamPtr>> mode_groups;
        for (auto& stream : waiting_streams_) {
            if (isPrefillOnly(stream) != prefill_only) {
                draining_mixed_modes_ = true;
                continue;
            }
            mode_groups[stream->getReturnAllProbs()].push_back(stream);
        }

        // Choose which ReturnAllProbsMode group to serve this round.
        //
        // Default policy favours throughput: pick the non-NONE group with the
        // most streams.  That alone can starve a minority mode (e.g. a few
        // ORIGINAL requests stuck behind a steady stream of DEFAULT ones) AND
        // can starve plain NONE requests, which otherwise only ride along as
        // wildcards behind whatever concrete group is chosen.  So a fairness
        // override kicks in: if ANY group's oldest stream (NONE included) has
        // waited longer than the starvation threshold, serve the group whose
        // oldest stream arrived earliest.  This bounds every group's worst-case
        // wait to ~starvation_threshold instead of "forever".
        const int64_t now_us = autil::TimeUtility::currentTimeInMicroSeconds();
        const int64_t starvation_threshold_us =
            std::chrono::duration_cast<std::chrono::microseconds>(2 * kFlushTimeoutMs).count();

        ReturnAllProbsMode selected_mode = ReturnAllProbsMode::NONE;
        size_t             max_count     = 0;

        ReturnAllProbsMode starved_mode       = ReturnAllProbsMode::NONE;
        int64_t            starved_arrival_us = 0;
        bool               has_starved        = false;
        for (auto& [mode, streams] : mode_groups) {
            // NONE participates in starvation detection too: a long-waiting NONE
            // request must be able to trigger the fairness override rather than
            // starve behind a steady all-probs backlog.
            if (streams.empty()) {
                continue;
            }
            // Groups are built by iterating waiting_streams_ in arrival order,
            // so front() is the group's oldest stream.
            const int64_t arrival_us = streams.front()->enqueueTime();
            if (now_us - arrival_us > starvation_threshold_us
                && (!has_starved || arrival_us < starved_arrival_us)) {
                has_starved        = true;
                starved_arrival_us = arrival_us;
                starved_mode       = mode;
            }
        }

        if (has_starved) {
            selected_mode = starved_mode;
            max_count     = mode_groups[selected_mode].size();
        } else {
            // Throughput default: the non-NONE group with the most streams.
            for (auto& [mode, streams] : mode_groups) {
                if (mode == ReturnAllProbsMode::NONE) {
                    continue;
                }
                if (streams.size() > max_count) {
                    max_count     = streams.size();
                    selected_mode = mode;
                }
            }
        }
        auto& none_streams = mode_groups[ReturnAllProbsMode::NONE];
        if (selected_mode == ReturnAllProbsMode::NONE && !none_streams.empty()) {
            selected_mode = ReturnAllProbsMode::NONE;
            max_count     = none_streams.size();
        }

        if (max_count == 0) {
            return;
        }

        // Build the candidate list from the selected concrete mode plus all NONE
        // streams.  This lets compatible wildcard requests batch together instead
        // of being stranded in their own group.
        std::list<GenerateStreamPtr> candidates;
        if (selected_mode != ReturnAllProbsMode::NONE) {
            auto& selected_group = mode_groups[selected_mode];
            candidates.splice(candidates.end(), selected_group);
        }
        candidates.splice(candidates.end(), none_streams);

        // Merge by global enqueue order so the oldest compatible requests win the
        // limited batch slots.  Without this, NONE (spliced last) only ever gets
        // leftover slots and starves whenever the concrete group alone fills
        // batch_size_; sorting by arrival time gives all compatible modes FIFO
        // fairness within the batch.
        candidates.sort([](const GenerateStreamPtr& a, const GenerateStreamPtr& b) {
            return a->enqueueTime() < b->enqueueTime();
        });

        std::list<GenerateStreamPtr> new_streams;
        for (auto& s : candidates) {
            new_streams.push_back(s);
            if (new_streams.size() >= batch_size_) {
                break;
            }
        }
        // Ordinary calls flush partial compatible groups to avoid stranding mixed
        // modes. Fixed-batch benchmarks wait for the configured batch size.
        const bool should_schedule = !new_streams.empty()
                                     && (!require_full_batch_.load() || new_streams.size() >= batch_size_);
        if (should_schedule) {
            for (auto& stream : new_streams) {
                stream->reportEvent(StreamEvents::CanRun);
                // 忙等stream load cache done, 和原有SyncLoadCache逻辑等效
                while (stream->getStatus() != StreamState::FINISHED && stream->moveToNext() != StreamState::RUNNING) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            // Remove every selected stream, including one that finished while loading.
            // Keep Native's linear queue cleanup while preserving main's state routing.
            std::unordered_set<GenerateStream*> scheduled_ptrs;
            scheduled_ptrs.reserve(new_streams.size());
            for (const auto& stream : new_streams) {
                scheduled_ptrs.insert(stream.get());
                if (stream->getStatus() == StreamState::RUNNING) {
                    running_streams_.push_back(stream);
                }
            }
            waiting_streams_.remove_if([&](const GenerateStreamPtr& stream) {
                return scheduled_ptrs.count(stream.get()) > 0;
            });
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

    // Maximum time a compatible group of waiting streams may sit before being
    // scheduled as a partial batch.  Prevents mixed ReturnAllProbsMode groups
    // (or low-traffic periods) from waiting forever for a full batch_size_.
    static constexpr std::chrono::milliseconds kFlushTimeoutMs{100};

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        // Phase 1: park until there is any work at all. A long timeout avoids CPU spinning while
        // fully idle; enqueue() notifies as soon as a request arrives, so we wake promptly on the
        // first waiting stream rather than sleeping the whole interval.
        cond_.wait_for(lock, std::chrono::seconds(5), [this] {
            return !waiting_streams_.empty() || running_streams_.size() > 0
                   || !loading_cache_streams_.empty();
        });

        // Phase 2: we have a waiting batch that has not yet reached batch_size_ and nothing is
        // running. Give it up to kFlushTimeoutMs to fill; wake early if it reaches batch_size_ or
        // running/loading work appears. When the timer expires the predicate stays false and we
        // fall through to evaluate the batch. Ordinary calls can flush a partial batch;
        // fixed-batch benchmarks keep it waiting until enough compatible streams arrive.
        if (running_streams_.empty() && !waiting_streams_.empty()
            && waiting_streams_.size() < batch_size_) {
            cond_.wait_for(lock, kFlushTimeoutMs, [this] {
                return waiting_streams_.size() >= batch_size_ || running_streams_.size() > 0
                       || !loading_cache_streams_.empty();
            });
        }

        // 统一通过状态机驱动各队列中 stream 的状态转移
        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        evaluateAndUpdateStreams(running_streams_);

        if (waiting_streams_.empty()) {
            draining_mixed_modes_ = false;
        }

        // When no work is running, evaluate the waiting group using the requested batch policy.
        if (running_streams_.empty() && !waiting_streams_.empty()) {
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
    std::atomic<bool>            require_full_batch_{false};
    bool                         reorder_request_;
    bool                         draining_mixed_modes_ = false;
    uint32_t                     current_step_ = 0;

    std::shared_ptr<KVCacheManager> cache_manager_;
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    SchedulerType                   scheduler_type_;
    int                             dp_rank_ = 0;
};

}  // namespace rtp_llm
