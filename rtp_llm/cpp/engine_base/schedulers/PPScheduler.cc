#include "rtp_llm/cpp/engine_base/schedulers/PPScheduler.h"

#include <algorithm>
#include <chrono>

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

PPScheduler::PPScheduler(const RuntimeConfig&                   runtime_config,
                         const ModelConfig&                     model_config,
                         const PDSepConfig&                     pd_sep_config,
                         const ParallelismConfig&               parallelism_config,
                         const ModelSpecificConfig&             model_specific_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter):
    FIFOSchedulerBase(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      model_specific_config,
                      cache_manager,
                      metrics_reporter),
    max_batch_tokens_without_cache_(static_cast<size_t>(
        std::max<int64_t>(runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache, 0))) {}

PPScheduler::~PPScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destroy PPScheduler");
}

list<GenerateStreamPtr> PPScheduler::evaluateRunningStreams() {
    list<GenerateStreamPtr> scheduled_streams;
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        auto stream = *it;
        if (stream->isPPInflight()) {
            ++it;
            continue;
        }

        const auto new_state = stream->moveToNext();
        if (new_state == StreamState::RUNNING) {
            scheduled_streams.push_back(stream);
            ++it;
        } else {
            addStreamToNewState(stream, new_state);
            it = running_streams_.erase(it);
        }
    }
    return scheduled_streams;
}

void PPScheduler::admitWaitingStreams(list<GenerateStreamPtr>& scheduled_streams) {
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;

    if (!waiting_streams_.empty()) {
        auto oldest_enqueue_time_us = waiting_streams_.front()->schedulerEnqueueTimeUs();
        for (const auto& stream : waiting_streams_) {
            oldest_enqueue_time_us = std::min(oldest_enqueue_time_us, stream->schedulerEnqueueTimeUs());
        }
        last_waiting_oldest_age_us_ =
            std::max<int64_t>(0, autil::TimeUtility::currentTimeInMicroSeconds() - oldest_enqueue_time_us);
    }

    ScheduleRuntime schedule_runtime;
    initScheduleRuntime(scheduled_streams, schedule_runtime);

    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        auto current = it++;
        auto stream  = *current;

        /** Errored streams bypass admission and advance directly to their terminal state. */
        const auto state = stream->getStatus();
        if (stream->hasError()) {
            const auto new_state = stream->moveToNext();
            if (new_state != state) {
                addStreamToNewState(stream, new_state);
                waiting_streams_.erase(current);
            }
            continue;
        }

        /**
         * CanRun is persistent and may have been set before this scheduling round. It is not proof of
         * admission, so both the initialized-KV limit and current batch limits must be checked first.
         * A stream that already owns KV may continue even when the initialized-stream limit is full.
         */
        const bool already_inited_kv   = stream->curBlocksNum() > 0;
        const bool already_load_inited = stream->hasEvent(StreamEvents::LoadInitiated);

        if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv
            && schedule_runtime.inited_kv_stream_count >= max_inited_kv_cache_streams_) {
            continue;
        }

        if (!fitsCurrentBatch(schedule_runtime, stream)) {
            continue;
        }

        if (!stream->hasEvent(StreamEvents::CanRun)) {
            stream->reportEvent(StreamEvents::CanRun);
        }

        const auto new_state            = stream->moveToNext();
        const bool new_inited_kv        = !already_inited_kv && stream->curBlocksNum() > 0;
        const bool new_load_inited      = !already_load_inited && stream->hasEvent(StreamEvents::LoadInitiated);
        const bool admission_progressed = new_state == StreamState::RUNNING || new_state == StreamState::LOADING_CACHE
                                          || (new_state == StreamState::WAITING && (new_inited_kv || new_load_inited));

        /** Admission progresses when the stream enters RUNNING/LOADING_CACHE,
         * or initializes KV/cache loading while remaining in WAITING. */
        if (admission_progressed) {
            updateScheduleRuntime(schedule_runtime, stream, new_inited_kv);

            if (stream->isContextStream()) {
                ++last_admitted_context_batch_size_;
                last_admitted_context_token_size_ += stream->contextLength();
            }
            if (new_state == StreamState::RUNNING) {
                scheduled_streams.push_back(stream);
            }
        }

        if (new_state != state) {
            addStreamToNewState(stream, new_state);
            waiting_streams_.erase(current);
        }
    }
}

bool PPScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) {
    ScheduleRuntime schedule_runtime;
    initScheduleRuntime(streams, schedule_runtime);
    return fitsCurrentBatch(schedule_runtime, new_stream);
}

void PPScheduler::initScheduleRuntime(const std::list<GenerateStreamPtr>& scheduled_streams,
                                      ScheduleRuntime&                    schedule_runtime) const {

    schedule_runtime.inited_kv_stream_count = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;
    for (const auto& stream : scheduled_streams) {
        updateScheduleRuntime(schedule_runtime, stream);
    }
}

void PPScheduler::updateScheduleRuntime(ScheduleRuntime&         schedule_runtime,
                                        const GenerateStreamPtr& stream,
                                        bool                     new_inited_kv) const {
    if (stream->getStatus() == StreamState::RUNNING) {
        ++schedule_runtime.scheduled_stream_count;
        schedule_runtime.scheduled_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
        schedule_runtime.scheduled_prefill_max_seq_len_with_cache =
            std::max(schedule_runtime.scheduled_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
        schedule_runtime.scheduled_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());
    }

    if (new_inited_kv) {
        schedule_runtime.inited_kv_stream_count++;
    }

    if (stream->isContextStream()) {
        schedule_runtime.admitted_prefill_token_size_without_cache += prefillTokenCostWithoutCache(stream);
    }
}

bool PPScheduler::fitsCurrentBatch(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& candidate) const {
    if (schedule_runtime.scheduled_stream_count + 1 > max_generate_batch_size_) {
        return false;
    }

    if (pd_sep_config_.role_type == RoleType::DECODE) {
        return true;
    }

    if (max_batch_tokens_without_cache_ > 0
        && schedule_runtime.admitted_prefill_token_size_without_cache >= max_batch_tokens_without_cache_) {
        return false;
    }

    return fitsPrefillTokenLimits(schedule_runtime, candidate);
}

bool PPScheduler::fitsPrefillTokenLimits(const ScheduleRuntime&   schedule_runtime,
                                         const GenerateStreamPtr& candidate) const {
    if (schedule_runtime.scheduled_stream_count == 0 && candidate->contextLength() < static_cast<int>(max_seq_len_)) {
        return true;
    }

    if (schedule_runtime.scheduled_prefill_token_size_with_cache >= max_batch_tokens_size_) {
        return false;
    }
    const auto candidate_tokens = prefillTokenCostWithCache(candidate);
    if (candidate_tokens >= max_batch_tokens_size_ - schedule_runtime.scheduled_prefill_token_size_with_cache) {
        return false;
    }

    const auto sequence_count =
        schedule_runtime.scheduled_prefill_sequence_count + static_cast<size_t>(candidate->currentBatchSize());
    const auto max_seq_len =
        std::max(schedule_runtime.scheduled_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(candidate));
    return max_seq_len == 0 || sequence_count <= (max_batch_tokens_size_ - 1) / max_seq_len;
}

size_t PPScheduler::prefillSeqLenWithCache(const GenerateStreamPtr& stream) const {
    return static_cast<size_t>(std::max(stream->contextLength(), 0))
           + static_cast<size_t>(std::max(stream->prefixLength(), 0));
}

size_t PPScheduler::prefillTokenCostWithCache(const GenerateStreamPtr& stream) const {
    return prefillSeqLenWithCache(stream) * static_cast<size_t>(stream->currentBatchSize());
}

size_t PPScheduler::prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const {
    return static_cast<size_t>(std::max(stream->contextLength(), 0)) * static_cast<size_t>(stream->currentBatchSize());
}

bool PPScheduler::waitPredicate() {
    return stop_ || schedule_trigger_ || !waiting_streams_.empty() || !loading_cache_streams_.empty()
           || !running_streams_.empty();
}

absl::StatusOr<list<GenerateStreamPtr>> PPScheduler::schedule() {
    RTP_LLM_PROFILE_FUNCTION();
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    schedule_trigger_ = false;

    /** 1. Handle all running streams, skip the in-flight streams. */
    auto scheduled_streams = evaluateRunningStreams();

    /** 2. Handle all loading cache streams. All of loaded-done streams are put in waiting_streams_  */
    evaluateAndUpdateStreams(loading_cache_streams_);

    /** 3. Handle the waiting_streams_ based on the role. */
    switch (pd_sep_config_.role_type) {
        case RoleType::PDFUSION:
            /** Ready decode streams take priority over waiting prefill streams. */
            if (scheduled_streams.empty()) {
                admitWaitingStreams(scheduled_streams);
            }
            break;
        case RoleType::PREFILL:
            /** Do not mix a ready fallback stream with a new prefill batch. */
            if (scheduled_streams.empty()) {
                admitWaitingStreams(scheduled_streams);
            }
            break;
        case RoleType::DECODE:
            /** Waiting decode streams may top up the ready decode batch. */
            admitWaitingStreams(scheduled_streams);
            break;
        default:
            RTP_LLM_LOG_ERROR("Unsupported role type %d in PPScheduler", static_cast<int>(pd_sep_config_.role_type));
            break;
    }
    /** 4. update the running_streams. */
    running_streams_.splice(running_streams_.end(), new_streams_);

    for (const auto& stream : scheduled_streams) {
        stream->setPPInflight();
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return scheduled_streams;
}

}  // namespace rtp_llm
