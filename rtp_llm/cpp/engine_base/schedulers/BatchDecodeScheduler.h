#pragma once

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
#include <list>
#include <unordered_set>
#include <exception>
#include <limits>

namespace rtp_llm {

struct BatchDecodeSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
        json.Jsonize("mode", mode_, "decode");
    }
    int64_t     batch_size_ = 0;
    std::string mode_;
};
// Test-only scheduler that preserves explicitly requested batches.
class BatchDecodeScheduler: public SchedulerBase {
public:
    enum SchedulerType : std::uint8_t {
        kBatchDecode  = 0,
        kBatchPrefill = 1
    };
    BatchDecodeScheduler(const RuntimeConfig&                   runtime_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter,
                         int                                    dp_rank                 = 0,
                         std::chrono::milliseconds              allocation_wait_timeout = std::chrono::seconds(30),
                         std::function<std::chrono::steady_clock::time_point()> now = std::chrono::steady_clock::now):
        allocation_wait_timeout_(allocation_wait_timeout), now_(std::move(now)) {
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
            if (stop_requested_.load()) {
                stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
                // Never admitted to the engine; no execution can own its KV.
                stream->moveToNext();
                return absl::CancelledError("scheduler stopped");
            }
            stream->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds());
            addStreamToNewState(stream, StreamState::WAITING);
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
            if (stop_requested_.load()) {
                for (const auto& stream : stream_enqueued) {
                    stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
                    stream->moveToNext();
                }
                return {std::vector<bool>(streams.size(), false), streams};
            }
            for (const auto& stream : stream_enqueued) {
                addStreamToNewState(stream, StreamState::WAITING);
            }
        }
        cond_.notify_all();
        return {std::move(enqueue_successes), streams};
    }

    absl::Status updateSchedulerInfo(const std::string& scheduler_info) override {
        BatchDecodeSchedulerConfigLocal config;
        try {
            autil::legacy::FromJsonString(config, scheduler_info);
        } catch (const std::exception& e) {
            return absl::InvalidArgumentError(e.what());
        }
        if (config.batch_size_ <= 0 || config.batch_size_ > std::numeric_limits<uint32_t>::max()) {
            return absl::InvalidArgumentError("BatchDecodeScheduler requires a positive batch_size fitting uint32");
        }
        if (config.mode_ != "decode" && config.mode_ != "prefill") {
            return absl::InvalidArgumentError("BatchDecodeScheduler mode must be decode or prefill");
        }
        if (cache_manager_) {
            const auto& replay = cache_manager_->cacheConfig().linear_replay;
            if (replay.enabled() && config.batch_size_ > replay.slotCount()) {
                return absl::InvalidArgumentError(
                    "BatchDecodeScheduler batch_size " + std::to_string(config.batch_size_)
                    + " exceeds LINEAR replay slot capacity " + std::to_string(replay.slotCount()));
            }
        }
        const auto batch_size = static_cast<uint32_t>(config.batch_size_);
        const auto mode       = config.mode_ == "decode" ? SchedulerType::kBatchDecode : SchedulerType::kBatchPrefill;
        {
            std::lock_guard<std::mutex> lock(lock_);
            batch_size_     = batch_size;
            scheduler_type_ = mode;
        }
        cond_.notify_all();
        RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %u, mode to %d", batch_size, int(mode));
        return absl::OkStatus();
    }

    // 根据状态机转移后的目标状态，将 stream 路由到对应的队列
    void addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
        switch (new_state) {
            case StreamState::WAITING:
                if (stream->hasHiddenStateCaptureReservation()) {
                    waiting_reservations_.insert(stream);
                }
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
                if (state == StreamState::LOADING_CACHE && new_state == StreamState::FINISHED) {
                    capture_batch_.remove(*it);
                    (*it)->cancelHiddenStateCapture();
                }
                addStreamToNewState(*it, new_state);
                it = streams.erase(it);
            } else {
                it++;
            }
        }
    }

    static bool isReservedStrictCapture(const GenerateStreamPtr& stream) {
        return stream->generateConfig()->requiresHiddenStateCaptureAdmission()
               && stream->canExecuteReservedHiddenStateCapture();
    }

    bool waitingBatchReady() const {
        return waiting_streams_.size() >= batch_size_
               || (!waiting_streams_.empty() && (draining_mixed_modes_ || !capture_batch_.empty()))
               || std::any_of(waiting_reservations_.begin(), waiting_reservations_.end(), [](const auto& stream) {
                      return stream->hasError() || isReservedStrictCapture(stream);
                  });
    }

    std::chrono::milliseconds evaluateWaitingStreams() {
        using namespace std::chrono_literals;
        // Finalize errored streams before selecting a batch so they cannot count toward its size.
        const auto reap = [this](auto it) {
            (*it)->moveToNext();
            (*it)->cancelHiddenStateCapture();
            waiting_reservations_.erase(*it);
            capture_batch_.remove(*it);
            return waiting_streams_.erase(it);
        };
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
            if (!(*it)->isFinished() && !(*it)->hasError()) {
                ++it;
                continue;
            }
            it = reap(it);
        }

        std::list<GenerateStreamPtr> selected_streams;
        selected_streams.swap(capture_batch_);
        const bool continuing = !selected_streams.empty();
        const bool strict_only = !continuing && waiting_streams_.size() < batch_size_ && !draining_mixed_modes_;
        bool has_other_mode = false;
        if (!continuing) {
            // Select main's batch before considering capture; only a native zero/positive mix opens the tail gate.
            for (const auto& stream : waiting_streams_) {
                if (strict_only && !isReservedStrictCapture(stream)) {
                    continue;
                }
                if (!selected_streams.empty() && isPrefillOnly(selected_streams.front()) != isPrefillOnly(stream)) {
                    has_other_mode = true;
                } else if (selected_streams.size() < batch_size_) {
                    selected_streams.push_back(stream);
                }
            }
        }
        if (selected_streams.empty()) {
            draining_mixed_modes_ = false;
            return 0ms;
        }

        // Only positive-token batches consume replay slots. Capture continuations are prefill-only.
        const bool replay_batch = !isPrefillOnly(selected_streams.front()) && cache_manager_
                                  && cache_manager_->cacheConfig().linear_replay.enabled();
        if (replay_batch) {
            if (auto blocked = tryReserveBatchReplaySlots(selected_streams)) {
                if (stop_requested_.load()) {
                    return 0ms;
                }
                if (blocked->isFinished() || blocked->hasError()) {
                    reap(std::find(waiting_streams_.begin(), waiting_streams_.end(), blocked));
                    return 0ms;
                }
                return 1ms;
            }
        }
        if (!continuing) {
            draining_mixed_modes_ = has_other_mode;
        }

        const auto reserved = [](const auto& stream) { return stream->canExecuteReservedHiddenStateCapture(); };
        const bool single_step = continuing || strict_only
                                 || (isPrefillOnly(selected_streams.front()) && !waiting_reservations_.empty()
                                     && std::any_of(selected_streams.begin(), selected_streams.end(), reserved));
        if (!single_step) {
            for (auto& stream : selected_streams) {
                if (stop_requested_.load()) {
                    return 0ms;
                }
                if (scheduler_type_ == SchedulerType::kBatchDecode && replay_batch) {
                    // Initialize the processed prefix before the first KV grow in direct decode.
                    stream->setPerfTest(true);
                    stream->setIsContextStream(false);
                }
                stream->reportEvent(StreamEvents::CanRun);
                // This synchronous benchmark scheduler cannot execute the selected streams to
                // reclaim KV while admitting the batch. Bound consecutive allocation retries, but
                // leave actual cache I/O to the stream/request and connector timeout policies.
                auto allocation_deadline = now_() + allocation_wait_timeout_;
                while (!stop_requested_.load() && stream->getStatus() != StreamState::FINISHED) {
                    const auto previous_state = stream->getStatus();
                    const auto state          = stream->moveToNext();
                    if (state == StreamState::RUNNING || state == StreamState::FINISHED) {
                        break;
                    }
                    if (state == StreamState::LOADING_CACHE || previous_state == StreamState::LOADING_CACHE) {
                        // Loading may return to WAITING for materialization/reallocation. Give
                        // that allocation its own budget, independent of time spent doing I/O.
                        allocation_deadline = now_() + allocation_wait_timeout_;
                    } else if (state == StreamState::WAITING && now_() >= allocation_deadline) {
                        stream->reportError(ErrorCode::MALLOC_FAILED,
                                            "BatchDecodeScheduler: KV allocation reached the "
                                                + std::to_string(allocation_wait_timeout_.count()) + " ms wait limit");
                        stream->moveToNext();
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                // Return terminal capture quota before another stream's cache wait or a stop exit.
                if (stream->getStatus() == StreamState::FINISHED) {
                    stream->cancelHiddenStateCapture();
                }
                // Keep selected streams in waiting_streams_ for finalizeStop().
                if (stop_requested_.load()) {
                    return 0ms;
                }
            }
        }
        const bool capture = single_step && reserved(selected_streams.front());
        std::list<GenerateStreamPtr> deferred;
        bool retry = false;
        for (const auto& stream : selected_streams) {
            auto state = stream->getStatus();
            if (single_step) {
                if (stop_requested_.load()) {
                    return 0ms;
                }
                if (strict_only && !isReservedStrictCapture(stream)) {
                    continue;
                }
                if (reserved(stream) != capture) {
                    deferred.push_back(stream);
                    continue;
                }
                if (continuing && state == StreamState::LOADING_CACHE) {
                    capture_batch_.push_back(stream);
                    continue;
                }
                stream->reportEvent(StreamEvents::CanRun);
                state = stream->moveToNext();
                if (state == StreamState::WAITING || state == StreamState::LOADING_CACHE) {
                    stream->clearCanRun();
                    retry = true;
                    if (!strict_only) {
                        capture_batch_.push_back(stream);
                    }
                }
                if (state == StreamState::WAITING) {
                    continue;
                }
                if (state == StreamState::FINISHED) {
                    stream->cancelHiddenStateCapture();
                }
            }
            waiting_streams_.remove(stream);
            waiting_reservations_.erase(stream);
            addStreamToNewState(stream, state);
        }
        // Finish this partition (including any KV-owning loads) before starting its incompatible peers.
        capture_batch_.splice(capture_batch_.end(), deferred);
        return retry ? 10ms : 0ms;
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
                // Forced decode bypasses the context gather that would consume this reservation.
                (*it)->cancelHiddenStateCapture();
                (*it)->setIsContextStream(false);
                // for linear attn, incrKVBlock to clear unused linear block
                (*it)->moveToNext();
            }
        }
    }

    absl::StatusOr<ScheduleOutput> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        using Clock = std::chrono::steady_clock;
        const auto deadline = Clock::now() + std::chrono::seconds(30);
        while (!stop_requested_.load() && running_streams_.empty() && loading_cache_streams_.empty()
               && !waitingBatchReady()) {
            const auto now = Clock::now();
            if (now >= deadline) {
                break;
            }
            // Enqueue can introduce the first reservation during a long wait. Its later RPC error
            // won't notify cond_, so choose the next wait again after every wakeup.
            const auto wake = waiting_reservations_.empty() ? deadline :
                                  std::min(deadline, now + std::chrono::milliseconds(10));
            cond_.wait_until(lock, wake);
        }

        if (stop_requested_.load()) {
            return ScheduleOutput{{}, {}};
        }

        // 统一通过状态机驱动各队列中 stream 的状态转移
        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        evaluateAndUpdateStreams(running_streams_);

        if (waiting_streams_.empty()) {
            draining_mixed_modes_ = false;
        }

        std::chrono::milliseconds retry_delay{0};
        if (running_streams_.empty() && waitingBatchReady()) {
            retry_delay = evaluateWaitingStreams();
            if (stop_requested_.load()) {
                return ScheduleOutput{{}, {}};
            }
            if (!running_streams_.empty()) {
                initRunningStreams();
                RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run",
                                 running_streams_.size());
            }
        }

        if (!loading_cache_streams_.empty() && retry_delay.count() == 0) {
            retry_delay = std::chrono::milliseconds(10);
        }
        if (running_streams_.empty() && retry_delay.count() > 0) {
            // Neither replay-slot nor KV release notifies cond_; keep the retry interruptible by stop.
            cond_.wait_for(lock, retry_delay, [this] { return stop_requested_.load(); });
        }
        return ScheduleOutput{running_streams_};
    }

    absl::Status stop() override {
        // Cache-load polling holds lock_, so publish the stop request first.
        stop_requested_.store(true);
        {
            // Pair with schedule's predicate check to avoid a lost wakeup.
            std::lock_guard<std::mutex> lock(lock_);
        }
        cond_.notify_all();
        // The engine can still be executing. Keep every stream and its KV alive.
        return absl::OkStatus();
    }

    absl::Status finalizeStop() override {
        std::lock_guard<std::mutex> lock(lock_);
        if (!stop_requested_.load()) {
            return absl::FailedPreconditionError("finalizeStop requires stop first");
        }
        // Called only after the engine loop joins and executor dispatch drains.
        for (auto* streams : {&waiting_streams_, &loading_cache_streams_, &running_streams_}) {
            for (const auto& stream : *streams) {
                if (!stream->isFinished() && !stream->hasEvent(StreamEvents::GenerateDone) && !stream->hasError()) {
                    stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
                }
                stream->moveToNext();
            }
            streams->clear();
        }
        waiting_reservations_.clear();
        capture_batch_.clear();
        draining_mixed_modes_ = false;
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

private:
    GenerateStreamPtr tryReserveBatchReplaySlots(const std::list<GenerateStreamPtr>& streams) {
        std::vector<StreamCacheResource*> reservations;
        for (const auto& stream : streams) {
            auto& resource       = stream->streamCacheResource();
            bool  newly_reserved = false;
            if (!resource.reserveLinearReplaySlot(&newly_reserved)) {
                for (auto* reserved : reservations) {
                    reserved->releaseUnusedLinearReplaySlot();
                }
                return stream;
            }
            if (newly_reserved) {
                reservations.push_back(&resource);
            }
        }
        return nullptr;
    }

    std::atomic<bool>            stop_requested_{false};
    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> loading_cache_streams_;
    std::list<GenerateStreamPtr> running_streams_;
    uint32_t                     batch_size_;
    bool                         reorder_request_;
    uint32_t                     current_step_         = 0;
    bool                         draining_mixed_modes_ = false;
    // Index reservations at queue entry, not by rescanning ordinary underfilled queues on wakeup.
    std::unordered_set<GenerateStreamPtr> waiting_reservations_;
    // Outstanding members of one normally admitted capture-bearing zero-token batch (at most B).
    std::list<GenerateStreamPtr> capture_batch_;

    std::shared_ptr<KVCacheManager>                              cache_manager_;
    kmonitor::MetricsReporterPtr                                 metrics_reporter_;
    SchedulerType                                                scheduler_type_;
    int                                                          dp_rank_ = 0;
    const std::chrono::milliseconds                              allocation_wait_timeout_;
    const std::function<std::chrono::steady_clock::time_point()> now_;
};

}  // namespace rtp_llm
