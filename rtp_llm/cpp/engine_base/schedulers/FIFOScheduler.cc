#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <algorithm>
#include <chrono>
#include <mutex>

using namespace std;
namespace rtp_llm {
namespace {

bool earlierEnqueueTime(const GenerateStreamPtr& lhs, const GenerateStreamPtr& rhs) {
    return lhs->schedulerEnqueueTimeUs() < rhs->schedulerEnqueueTimeUs();
}

bool isChunkedContext(const GenerateStreamPtr& stream) {
    return stream->chunkedPrefillEnabled() && stream->isContextStream();
}

bool isLoadedChunkContext(const GenerateStreamPtr& stream) {
    return stream->hasEvent(StreamEvents::LoadInitiated) && isChunkedContext(stream);
}

}  // namespace

FIFOScheduler::FIFOScheduler(const RuntimeConfig&                   runtime_config,
                             const ModelConfig&                     model_config,
                             const PDSepConfig&                     pd_sep_config,
                             const ParallelismConfig&               parallelism_config,
                             const std::shared_ptr<KVCacheManager>& cache_manager,
                             const kmonitor::MetricsReporterPtr     metrics_reporter,
                             const int                              max_score_len):
    FIFOSchedulerBase(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      cache_manager,
                      metrics_reporter),
    max_batch_kv_len_(static_cast<size_t>(std::max<int64_t>(runtime_config.fifo_scheduler_config.max_batch_kv_len, 0))),
    max_batch_tokens_without_cache_(static_cast<size_t>(
        std::max<int64_t>(runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache, 0))),
    prefill_cp_size_(parallelism_config.prefill_cp_config.is_enabled() ?
                         static_cast<size_t>(std::max<int64_t>(parallelism_config.tp_size, 1)) :
                         1) {
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%zu], max_batch_tokens_size is [%zu], "
                     "max_batch_tokens_without_cache is [%zu], cp_force_single_prefill is [%d], "
                     "prefill_cp_size is [%zu], max_inited_kv_cache_streams is [%zu]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_,
                     max_batch_tokens_without_cache_,
                     force_single_prefill_,
                     prefill_cp_size_,
                     max_inited_kv_cache_streams_);
}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory FIFOScheduler");
}

void FIFOScheduler::cancelGroups(StreamGroupQueue& group_queue) {
    for (auto& group : group_queue) {
        cancelStreams(group);
    }
    group_queue.clear();
}

void FIFOScheduler::cancelExtraStreams() {
    cancelGroups(waiting_group_queue_);
    cancelGroups(loading_cache_group_queue_);
    cancelStreams(pending_decode_streams_);
}

int64_t FIFOScheduler::extraOnflightStreams() const {
    return groupQueueStreamsSize(waiting_group_queue_) + groupQueueStreamsSize(loading_cache_group_queue_)
           + pending_decode_streams_.size();
}

void FIFOScheduler::fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {
    collector.wait_stream_size += groupQueueStreamsSize(waiting_group_queue_);
    collector.loading_cache_stream_size += groupQueueStreamsSize(loading_cache_group_queue_);
    collector.pending_decode_stream_size = pending_decode_streams_.size();
    collector.group_fallback_count       = pending_group_fallback_count_.exchange(0, std::memory_order_relaxed);
}

void FIFOScheduler::appendExtraRunningTaskList(std::vector<EngineScheduleInfo::TaskInfo>& task_list) const {
    appendTaskInfos(task_list, pending_decode_streams_);
}

std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
FIFOScheduler::enqueueGroup(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
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
    std::vector<GenerateStreamPtr> valid_streams;
    valid_streams.reserve(streams.size());
    for (const auto& stream : streams) {
        const bool success = checkInputLength(stream) && checkChunkedPrefillRequest(stream).ok();
        enqueue_successes.push_back(success);
        if (success) {
            valid_streams.push_back(stream);
        }
    }

    if (valid_streams.empty()) {
        return {std::move(enqueue_successes), streams};
    }

    const bool exceeds_inited_kv_limit =
        max_inited_kv_cache_streams_ > 0 && valid_streams.size() > max_inited_kv_cache_streams_;
    const bool exceeds_batch_limit            = valid_streams.size() > max_generate_batch_size_;
    const bool fallback_to_individual_streams = exceeds_inited_kv_limit || exceeds_batch_limit;
    if (fallback_to_individual_streams) {
        RTP_LLM_LOG_DEBUG("enqueue group exceeds scheduler limits; fallback to individual streams: "
                          "group_size=%zu max_generate_batch_size=%zu max_inited_kv_cache_streams=%zu",
                          valid_streams.size(),
                          max_generate_batch_size_,
                          max_inited_kv_cache_streams_);
    }
    {
        std::lock_guard<std::mutex> lock(lock_);
        const auto                  enqueue_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        for (auto& stream : valid_streams) {
            stream->recordSchedulerEnqueueTime(enqueue_time_us);
        }
        if (fallback_to_individual_streams) {
            waiting_streams_.insert(waiting_streams_.end(), valid_streams.begin(), valid_streams.end());
            pending_group_fallback_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            waiting_group_queue_.emplace_back(valid_streams.begin(), valid_streams.end());
        }
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return {std::move(enqueue_successes), streams};
}

bool FIFOScheduler::usesChunkedPrefillAdmission() const {
    return pd_sep_config_.role_type == RoleType::PREFILL && prefill_chunk_size_ > 0
           && active_admission_lane_ != AdmissionLane::GROUP;
}

bool FIFOScheduler::hasPendingLocalDecode() const {
    const auto needs_local_decode = [](const auto& stream) {
        return stream && !stream->queryPdSep() && !isPrefillOnly(stream)
               && stream->generateConfig()->max_new_tokens > 1 && stream->isActive()
               && !stream->hasEvent(StreamEvents::GenerateDone);
    };
    for (const auto* streams : {&pending_decode_streams_, &running_streams_, &loading_cache_streams_}) {
        if (std::any_of(streams->begin(), streams->end(), needs_local_decode)) {
            return true;
        }
    }
    return false;
}

bool FIFOScheduler::evaluateRunningBatch(const ScheduleRuntime&   schedule_runtime,
                                         const GenerateStreamPtr& new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    const auto admitted_stream_count = schedule_runtime.admitted_stream_count;
    const bool continuous_prefill    = usesChunkedPrefillAdmission();
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        // Decode-only scheduling can top up an existing running decode batch.
        // max_generate_batch_size_ is an inclusive cap; only requests above it
        // should be rejected.
        if (running_streams_.size() + admitted_stream_count + 1 <= max_generate_batch_size_) {
            return true;
        }
    }
    // Active decode rows cannot mix with prefill; parked decode rows may coexist with a chunked context round
    // because pending_decode_streams_ is excluded from the returned prefill batch.
    const bool is_chunked_context = isChunkedContext(new_stream);
    const bool running_compatible =
        running_streams_.empty()
        || (continuous_prefill && is_chunked_context
            && std::all_of(running_streams_.begin(), running_streams_.end(), isChunkedContext));
    if (!running_compatible || (!pending_decode_streams_.empty() && !is_chunked_context)) {
        return false;
    }
    // Conservative CP prefill mode: cap at one stream per round unless
    // runtime config explicitly allows CP prefill batching.
    if (force_single_prefill_ && admitted_stream_count > 0) {
        return false;
    }
    const auto existing_stream_count =
        continuous_prefill ? 0 : running_streams_.size() + pending_decode_streams_.size();
    if (existing_stream_count + admitted_stream_count + 1 > max_generate_batch_size_) {
        return false;
    }

    return fitsPrefillTokenLimits(schedule_runtime, new_stream, continuous_prefill);
}

// Overload for group-queue admission: the admitted streams are tracked as a list
// because ScheduleRuntime is not built in the group-queue path.
bool FIFOScheduler::evaluateRunningBatch(const std::list<GenerateStreamPtr>& streams,
                                         const GenerateStreamPtr&            new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    const auto admitted_count = streams.size();
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        if (running_streams_.size() + admitted_count + 1 <= max_generate_batch_size_) {
            return true;
        }
    }
    if (!running_streams_.empty()) {
        return false;
    }
    if (force_single_prefill_ && admitted_count > 0) {
        return false;
    }
    if (running_streams_.size() + admitted_count + 1 > max_generate_batch_size_) {
        return false;
    }
    ScheduleRuntime schedule_runtime;
    schedule_runtime.admitted_stream_count  = admitted_count;
    schedule_runtime.scheduled_stream_count = admitted_count;
    for (const auto& stream : streams) {
        schedule_runtime.admitted_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
        schedule_runtime.admitted_prefill_token_size_without_cache += prefillTokenCostWithoutCache(stream);
        schedule_runtime.admitted_prefill_max_seq_len_with_cache =
            std::max(schedule_runtime.admitted_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
        schedule_runtime.admitted_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());
    }
    return fitsPrefillTokenLimits(schedule_runtime, new_stream);
}

bool FIFOScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                          const GenerateStreamPtr&       new_stream) {
    // FIFOScheduler's own scheduling loop uses the counter-based evaluateRunningBatch();
    // this list-based FIFOSchedulerBase entry point delegates to the list overload so
    // both paths agree.
    return evaluateRunningBatch(streams, new_stream);
}

bool FIFOScheduler::fitsPrefillTokenLimits(const ScheduleRuntime&   schedule_runtime,
                                           const GenerateStreamPtr& candidate,
                                           bool                     continuous_prefill) const {
    // Continuous admission already accounts for running work.
    const auto running_token_reserve = continuous_prefill ? 0 : running_streams_.size();
    const bool prefill_only          = isPrefillOnly(candidate);
    const bool continuous_chunk      = continuous_prefill && isChunkedContext(candidate);
    // Outstanding reservations consume capacity; only this round's progress blocks singleton admission.
    const auto singleton_blocker_count =
        continuous_chunk ? schedule_runtime.scheduled_stream_count : schedule_runtime.admitted_stream_count;
    const auto singleton_tokens =
        (continuous_prefill ? prefillSeqLenWithCache(candidate) : candidate->contextLength()) + running_token_reserve;
    if (singleton_blocker_count == 0
        && (singleton_tokens < max_seq_len_ || (prefill_only && singleton_tokens == max_seq_len_))) {
        return true;
    }

    const auto admitted_tokens = schedule_runtime.admitted_prefill_token_size_with_cache;
    if (max_batch_kv_len_ > 0) {
        const auto fits = [running_token_reserve,
                           prefill_only](size_t admitted, size_t candidate_tokens, size_t limit) {
            if (running_token_reserve >= limit) {
                return false;
            }
            const auto available_tokens = limit - running_token_reserve - (prefill_only ? 0 : 1);
            return admitted <= available_tokens && candidate_tokens <= available_tokens - admitted;
        };
        auto admitted_queries  = schedule_runtime.admitted_prefill_token_size_without_cache;
        auto candidate_queries = prefillTokenCostWithoutCache(candidate);
        if (continuous_chunk) {
            // KV reserves the full prompt. Query only charges what can run in this chunk.
            const auto chunk_limit = static_cast<size_t>(prefill_chunk_size_);
            admitted_queries       = std::min(admitted_queries, chunk_limit);
            candidate_queries      = std::min(candidate_queries, chunk_limit - admitted_queries);
        }
        return fits(admitted_tokens, prefillTokenCostWithCache(candidate), max_batch_kv_len_)
               && fits(admitted_queries, candidate_queries, max_batch_tokens_size_);
    }

    if (running_token_reserve >= max_batch_tokens_size_) {
        return false;
    }
    // Admission partitions execution modes before reaching this helper. On this
    // budget path, only prefill-only batches may fill the cap; generation retains its strict reserve.
    const auto available_tokens = max_batch_tokens_size_ - running_token_reserve - (prefill_only ? 0 : 1);

    // Include reused prefixes. Subtraction avoids overflow in the token sum.
    if (admitted_tokens > available_tokens) {
        return false;
    }
    const auto candidate_tokens = prefillTokenCostWithCache(candidate);
    if (candidate_tokens > available_tokens - admitted_tokens) {
        return false;
    }

    // The model executes prefill as a padded rectangle. Use the full logical
    // sequence length (prefix included) and the real sequence width represented
    // by currentBatchSize(). Division and subtraction avoid overflow in both
    // max_seq_len * width and the combined sequence count.
    const auto candidate_sequence_count = static_cast<size_t>(candidate->currentBatchSize());
    const auto max_seq_len =
        std::max(schedule_runtime.admitted_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(candidate));
    if (max_seq_len == 0) {
        return true;
    }
    const auto max_sequence_count      = available_tokens / max_seq_len;
    const auto admitted_sequence_count = schedule_runtime.admitted_prefill_sequence_count;
    return admitted_sequence_count <= max_sequence_count
           && candidate_sequence_count <= max_sequence_count - admitted_sequence_count;
}

size_t FIFOScheduler::prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const {
    // Match the token count that CP presents to the model after per-sequence padding.
    auto       token_count      = static_cast<size_t>(std::max(stream->contextLength(), 0));
    const bool continuous_chunk = usesChunkedPrefillAdmission() && isChunkedContext(stream);
    if (continuous_chunk) {
        // A running stream still carries its previous round's grant. The next
        // round may grant more, so reserve its remaining compute instead.
        token_count = static_cast<size_t>(std::max(stream->seqLength() - stream->reuseLength(), 0));
    }
    if (prefill_cp_size_ > 1) {
        token_count = makeZigzagTokenLayout(token_count, prefill_cp_size_).padded_token_count;
    }
    const auto compute_tokens = token_count * static_cast<size_t>(stream->currentBatchSize());
    return continuous_chunk ? std::min(compute_tokens, static_cast<size_t>(prefill_chunk_size_)) : compute_tokens;
}

size_t FIFOScheduler::prefillSeqLenWithCache(const GenerateStreamPtr& stream) const {
    // max_batch_tokens_size bounds the full logical sequence. Unlike the compute-only
    // quota, this count includes the reused prefix and therefore does not apply CP padding.
    // contextLength() is only the current chunk for a running chunked stream.
    // Its outstanding reservation must continue to include the entire prompt.
    if (usesChunkedPrefillAdmission() && isChunkedContext(stream)) {
        return static_cast<size_t>(std::max(stream->seqLength(), 0));
    }
    const auto input_length  = static_cast<size_t>(std::max(stream->contextLength(), 0));
    const auto prefix_length = static_cast<size_t>(std::max(stream->prefixLength(), 0));
    return input_length + prefix_length;
}

size_t FIFOScheduler::prefillTokenCostWithCache(const GenerateStreamPtr& stream) const {
    return prefillSeqLenWithCache(stream) * static_cast<size_t>(stream->currentBatchSize());
}

size_t FIFOScheduler::countInitedKVCacheStreams() const {
    const auto count_inited = [](const list<GenerateStreamPtr>& streams) {
        size_t count = 0;
        for (const auto& stream : streams) {
            if (stream && stream->curBlocksNum() > 0) {
                ++count;
            }
        }
        return count;
    };

    size_t count = count_inited(waiting_streams_) + count_inited(loading_cache_streams_)
                   + count_inited(running_streams_) + count_inited(new_streams_)
                   + count_inited(pending_decode_streams_);
    for (const auto& group : waiting_group_queue_) {
        count += count_inited(group);
    }
    for (const auto& group : loading_cache_group_queue_) {
        count += count_inited(group);
    }
    return count;
}

size_t FIFOScheduler::groupQueueStreamsSize(const StreamGroupQueue& group_queue) const {
    size_t count = 0;
    for (const auto& group : group_queue) {
        count += group.size();
    }
    return count;
}

void FIFOScheduler::accountBatchMetrics(const GenerateStreamPtr& new_stream) {
    for (auto& stream : running_streams_) {
        stream->incBatchWithPrefillTimes(1);
        stream->incBatchWithPrefillLen(new_stream->currentExecuteTokenSize());
    }
}

void FIFOScheduler::onRunningStream(const GenerateStreamPtr& stream) {
    accountBatchMetrics(stream);
}

bool FIFOScheduler::waitPredicate() {
    // Check streams directly without calling empty() which acquires lock_ (already held by schedule())
    // Pending loads and capacity-blocked waiters are polled by the timed wait below. They are not
    // wakeup events themselves, otherwise a retryable KV shortage spins the scheduler.
    return stop_ || schedule_trigger_ || !running_streams_.empty() || !pending_decode_streams_.empty();
}

void FIFOScheduler::admitWaitingStreams(list<GenerateStreamPtr>&       waiting_streams,
                                        const list<GenerateStreamPtr>& already_admitted_streams,
                                        bool                           prepared_only) {
    RTP_LLM_PROFILE_FUNCTION();
    const bool continuous_prefill = usesChunkedPrefillAdmission();
    if (continuous_prefill) {
        // Drain already-owned Prefill work so a parked local Decode can resume.
        // Fresh admissions could otherwise keep context work alive indefinitely.
        prepared_only = prepared_only || hasPendingLocalDecode();
        // A completed cache load rejoins WAITING, but keeps its original FIFO
        // position instead of falling behind requests that arrived during I/O.
        waiting_streams.sort(earlierEnqueueTime);
    }
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;
    if (!waiting_streams.empty()) {
        const auto& oldest_stream =
            continuous_prefill ? waiting_streams.front() :
                                 *std::min_element(waiting_streams.begin(), waiting_streams.end(), earlierEnqueueTime);
        last_waiting_oldest_age_us_ = std::max<int64_t>(
            0, autil::TimeUtility::currentTimeInMicroSeconds() - oldest_stream->schedulerEnqueueTimeUs());
    }
    const auto update_waiting_queue = [&](auto waiter, StreamState state, StreamState new_state) {
        if (new_state != state) {
            addStreamToNewState(*waiter, new_state);
            waiting_streams.erase(waiter);
        }
    };
    if (continuous_prefill) {
        // Reap expired or errored waiters before accounting reservations and
        // execution mode so terminal streams cannot hold slots for this round.
        for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
            auto current = it++;
            if ((*current)->checkTimeoutAndHasError()) {
                const auto state = (*current)->getStatus();
                update_waiting_queue(current, state, (*current)->moveToNext());
            }
        }
    }
    const size_t inited_kv_streams = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;

    // Explicit groups are scheduled through enqueueGroup() and the dedicated group
    // queues. group_id/group_size on a stream in waiting_streams are status metadata;
    // they must not delay or isolate ordinary FIFO admission.

    GenerateStreamPtr execution_mode_stream;
    ScheduleRuntime   schedule_runtime;
    const auto        has_reservation = [continuous_prefill](const GenerateStreamPtr& stream) {
        return continuous_prefill && !stream->hasError() && isLoadedChunkContext(stream) && stream->curBlocksNum() > 0;
    };
    const auto account_reservation = [&](const GenerateStreamPtr& stream, bool capture_mode = true) {
        if (capture_mode && !execution_mode_stream && stream && !stream->hasError()) {
            execution_mode_stream = stream;
        }
        ++schedule_runtime.admitted_stream_count;
        if (!continuous_prefill || stream->isContextStream()) {
            schedule_runtime.admitted_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
            schedule_runtime.admitted_prefill_max_seq_len_with_cache =
                std::max(schedule_runtime.admitted_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
            schedule_runtime.admitted_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());
        }
    };
    const auto without_reservation = [&](const GenerateStreamPtr& stream) {
        auto candidate_runtime = schedule_runtime;
        --candidate_runtime.admitted_stream_count;
        candidate_runtime.admitted_prefill_token_size_with_cache -= prefillTokenCostWithCache(stream);
        candidate_runtime.admitted_prefill_sequence_count -= static_cast<size_t>(stream->currentBatchSize());
        // Keep the maximum length: adding this same candidate restores the
        // original rectangle, even when it is the longest reservation.
        return candidate_runtime;
    };
    const auto account_progress = [&](const GenerateStreamPtr& stream, bool continuation = false) {
        ++schedule_runtime.scheduled_stream_count;
        if (stream->isContextStream()) {
            schedule_runtime.admitted_prefill_token_size_without_cache += prefillTokenCostWithoutCache(stream);
            if (!continuation) {
                ++last_admitted_context_batch_size_;
                last_admitted_context_token_size_ += stream->contextLength();
            }
        }
    };
    if (continuous_prefill) {
        // Preparation owns capacity across rounds, including while asynchronous
        // loads are pending and after they return to WAITING. A ready request
        // reuses this reservation; it does not allocate another batch slot.
        for (const auto& stream : running_streams_) {
            account_reservation(stream);
            account_progress(stream, /*continuation=*/true);
        }
        for (const auto& stream : pending_decode_streams_) {
            account_reservation(stream, /*capture_mode=*/false);
        }
        for (const auto& stream : loading_cache_streams_) {
            account_reservation(stream);
        }
        schedule_runtime.admitted_prefill_token_size_without_cache = std::min(
            schedule_runtime.admitted_prefill_token_size_without_cache, static_cast<size_t>(prefill_chunk_size_));
        for (const auto& stream : waiting_streams) {
            if (has_reservation(stream)) {
                account_reservation(stream);
            }
        }
    } else {
        for (const auto& stream : running_streams_) {
            if (stream && !stream->hasError() && stream->getStatus() == StreamState::RUNNING) {
                execution_mode_stream = stream;
                break;
            }
        }
    }
    for (const auto& stream : already_admitted_streams) {
        if (!stream || stream->getStatus() != StreamState::RUNNING) {
            continue;
        }
        account_reservation(stream);
        account_progress(stream);
    }

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto  current = it++;
        auto& stream  = *current;

        if (stream->checkTimeoutAndHasError()) {
            const auto state = stream->getStatus();
            update_waiting_queue(current, state, stream->moveToNext());
            continue;
        }

        // Once an older waiter is blocked, drain existing reservations before admitting newer work.
        const bool already_reserved = has_reservation(stream);
        if (prepared_only && !already_reserved) {
            continue;
        }
        const bool mode_mismatch =
            execution_mode_stream && isPrefillOnly(execution_mode_stream) != isPrefillOnly(stream);
        const bool compute_quota_reached =
            max_batch_tokens_without_cache_ > 0
            && schedule_runtime.admitted_prefill_token_size_without_cache >= max_batch_tokens_without_cache_;
        const bool already_inited_kv = stream->curBlocksNum() > 0;
        const bool kv_stream_limit_reached =
            max_inited_kv_cache_streams_ > 0 && !already_inited_kv
            && inited_kv_streams + schedule_runtime.newly_inited_kv_streams >= max_inited_kv_cache_streams_;
        const auto candidate_runtime = already_reserved ? without_reservation(stream) : schedule_runtime;
        if (mode_mismatch || compute_quota_reached || kv_stream_limit_reached
            || !evaluateRunningBatch(candidate_runtime, stream)) {
            // Freeze new admissions only for a waiter that could fit after existing work drains.
            const bool permanently_over_budget =
                continuous_prefill && !already_reserved && stream->isContextStream()
                && !fitsPrefillTokenLimits(ScheduleRuntime{}, stream, /*continuous_prefill=*/true);
            if (permanently_over_budget) {
                const auto state = stream->getStatus();
                stream->reportError(ErrorCode::INVALID_PARAMS,
                                    "input len " + std::to_string(stream->inputLength()) + " with batch token cost "
                                        + std::to_string(prefillTokenCostWithCache(stream))
                                        + " cannot fit max_batch_tokens_size=" + std::to_string(max_batch_tokens_size_)
                                        + ", max_batch_kv_len=" + std::to_string(max_batch_kv_len_)
                                        + ", max_seq_len=" + std::to_string(max_seq_len_)
                                        + " in an empty prefill batch");
                update_waiting_queue(current, state, stream->moveToNext());
            } else {
                prepared_only = prepared_only || continuous_prefill;
            }
            continue;
        }

        bool newly_reserved = false;
        if (!stream->streamCacheResource().reserveLinearReplaySlot(&newly_reserved)) {
            prepared_only = prepared_only || continuous_prefill;
            continue;
        }

        const auto state                 = stream->getStatus();
        const bool load_initiated_before = stream->hasEvent(StreamEvents::LoadInitiated);
        // CanRun can predate FIFO admission on PD Decode streams. Advance only after this round's checks.
        if (!stream->hasEvent(StreamEvents::CanRun)) {
            stream->reportEvent(StreamEvents::CanRun);
        }

        const auto new_state           = stream->moveToNext();
        const bool kv_initialized      = !already_inited_kv && stream->curBlocksNum() > 0;
        const bool load_initiated      = !load_initiated_before && stream->hasEvent(StreamEvents::LoadInitiated);
        const bool scheduling_progress = new_state == StreamState::RUNNING || new_state == StreamState::LOADING_CACHE
                                         || (new_state == StreamState::WAITING && (kv_initialized || load_initiated));

        if (newly_reserved && !scheduling_progress) {
            stream->streamCacheResource().releaseUnusedLinearReplaySlot();
        }
        if (scheduling_progress) {
            // A failed KV allocation must not pin the execution mode.
            if (!already_reserved) {
                account_reservation(stream);
            }
            account_progress(stream);
            schedule_runtime.newly_inited_kv_streams += kv_initialized;
        } else if (continuous_prefill && new_state == StreamState::WAITING && !stream->hasError()) {
            prepared_only = true;
        }

        update_waiting_queue(current, state, new_state);
    }
}

void FIFOScheduler::advanceLoadingGroup(StreamGroup& group) {
    for (auto it = group.begin(); it != group.end();) {
        auto state = (*it)->moveToNext();
        // Cache completion returns to WAITING, but this group already owns its admission.
        // Resume it immediately so a ready group does not consume another scheduling round.
        if (state == StreamState::WAITING) {
            state = (*it)->moveToNext();
        }
        if (state == StreamState::FINISHED) {
            it = group.erase(it);
            continue;
        }
        ++it;
    }
}

void FIFOScheduler::moveGroupToNewStreams(StreamGroup& group) {
    for (auto it = group.begin(); it != group.end();) {
        accountBatchMetrics(*it);
        new_streams_.splice(new_streams_.end(), group, it++);
    }
}

void FIFOScheduler::moveGroupToAllocatingGroup(StreamGroup& group) {
    loading_cache_group_queue_.push_back(std::move(group));
}

void FIFOScheduler::dispatchPreparedGroup(StreamGroup& group) {
    const bool all_running = std::all_of(
        group.begin(), group.end(), [](const auto& stream) { return stream->getStatus() == StreamState::RUNNING; });
    if (all_running) {
        moveGroupToNewStreams(group);
    } else {
        moveGroupToAllocatingGroup(group);
    }
}

void FIFOScheduler::evaluateLoadingCacheGroupQueue() {
    if (loading_cache_group_queue_.empty()) {
        return;
    }

    auto& group = loading_cache_group_queue_.front();
    advanceLoadingGroup(group);
    if (group.empty()) {
        loading_cache_group_queue_.pop_front();
        return;
    }
    if (std::any_of(group.begin(), group.end(), [](const auto& stream) {
            return stream->getStatus() != StreamState::RUNNING;
        })) {
        return;
    }
    if (!running_streams_.empty() || !new_streams_.empty()) {
        return;
    }

    moveGroupToNewStreams(group);
    loading_cache_group_queue_.pop_front();
}

bool FIFOScheduler::loadingGroupReady() const {
    return !loading_cache_group_queue_.empty() && !loading_cache_group_queue_.front().empty()
           && std::all_of(loading_cache_group_queue_.front().begin(),
                          loading_cache_group_queue_.front().end(),
                          [](const auto& stream) { return stream->getStatus() == StreamState::RUNNING; });
}

void FIFOScheduler::evaluateWaitingGroupQueue() {
    if (!running_streams_.empty() || !new_streams_.empty() || !loading_cache_group_queue_.empty()
        || waiting_group_queue_.empty()) {
        return;
    }

    // Evaluate at most one waiting group per scheduling round.
    auto&        group         = waiting_group_queue_.front();
    const size_t original_size = group.size();
    for (auto it = group.begin(); it != group.end();) {
        if ((*it)->hasError()) {
            (*it)->moveToNext();
            it = group.erase(it);
        } else {
            ++it;
        }
    }
    if (group.empty()) {
        waiting_group_queue_.pop_front();
        return;
    }

    StreamGroup  admitted_streams;
    const size_t inited_kv_streams       = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;
    size_t       newly_inited_kv_streams = 0;
    for (auto it = group.begin(); it != group.end();) {
        auto       current           = it++;
        auto&      stream            = *current;
        const bool already_inited_kv = stream->curBlocksNum() > 0;
        if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv
            && inited_kv_streams + newly_inited_kv_streams >= max_inited_kv_cache_streams_) {
            continue;
        }
        bool newly_reserved = false;
        if (!evaluateRunningBatch(admitted_streams, stream)
            || !stream->streamCacheResource().reserveLinearReplaySlot(&newly_reserved)) {
            continue;
        }

        if (!stream->hasEvent(StreamEvents::CanRun)) {
            stream->reportEvent(StreamEvents::CanRun);
        }
        const auto state = stream->moveToNext();
        if (newly_reserved && state == StreamState::WAITING) {
            stream->streamCacheResource().releaseUnusedLinearReplaySlot();
        }
        if (!already_inited_kv && stream->curBlocksNum() > 0) {
            ++newly_inited_kv_streams;
        }
        if (state == StreamState::FINISHED) {
            group.erase(current);
            continue;
        }
        if (state == StreamState::WAITING) {
            // Keep unresolved members in the residual group and continue the
            // stable scan. This is normally a retryable KV shortage; a DECODE
            // stream can also need one more state-machine transition after its
            // first successful allocation.
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(state == StreamState::RUNNING || state == StreamState::LOADING_CACHE,
                                "group stream must be RUNNING or LOADING_CACHE after scheduler admission");
        admitted_streams.splice(admitted_streams.end(), group, current);
    }

    if (group.empty()) {
        waiting_group_queue_.pop_front();
        if (admitted_streams.empty()) {
            return;
        }
        dispatchPreparedGroup(admitted_streams);
        return;
    }

    if (!admitted_streams.empty()) {
        RTP_LLM_LOG_DEBUG("group partially admitted; keeping residual group at queue head: "
                          "original=%zu admitted=%zu deferred=%zu",
                          original_size,
                          admitted_streams.size(),
                          group.size());
        dispatchPreparedGroup(admitted_streams);
        pending_group_fallback_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

bool FIFOScheduler::partitionChunkContinuations(bool has_waiting_prefill) {
    refreshAndReapTerminalStreams(running_streams_);
    const bool has_context =
        has_waiting_prefill || std::any_of(running_streams_.begin(), running_streams_.end(), [](const auto& stream) {
            return stream->isContextStream();
        });

    if (has_context) {
        for (auto it = running_streams_.begin(); it != running_streams_.end();) {
            if (!(*it)->isContextStream()) {
                auto current = it++;
                pending_decode_streams_.splice(pending_decode_streams_.end(), running_streams_, current);
            } else {
                ++it;
            }
        }
    } else if (!pending_decode_streams_.empty()) {
        running_streams_.splice(running_streams_.begin(), pending_decode_streams_);
        return true;
    }
    return false;
}

absl::StatusOr<ScheduleOutput> FIFOScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_ || !loading_cache_streams_.empty() || !loading_cache_group_queue_.empty()
        || !waiting_group_queue_.empty()) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else if (!waiting_streams_.empty()) {
        // No running work can release capacity. Poll PD-held block releases without turning a
        // large retryable queue into a tight malloc/eviction loop.
        cond_.wait_for(lock, std::chrono::milliseconds(100), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    schedule_trigger_                 = false;
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;

    const auto has_waiting_chunk = [this](bool waiting_queue_grew) {
        const bool continuous_prefill = usesChunkedPrefillAdmission();
        if (!continuous_prefill && !waiting_queue_grew) {
            return false;
        }
        return std::any_of(waiting_streams_.begin(), waiting_streams_.end(), [&](const auto& stream) {
            if (!isLoadedChunkContext(stream)) {
                return false;
            }
            // Continuous admission also picks up loads requeued in an earlier round.
            return continuous_prefill ? !stream->hasError() : stream->hasEvent(StreamEvents::CanRun);
        });
    };

    // A running explicit group owns the execution boundary. Leave ordinary loads queued until it drains.
    const size_t waiting_size_before_cache_update = waiting_streams_.size();
    if (running_streams_.empty() || active_admission_lane_ != AdmissionLane::GROUP) {
        evaluateAndUpdateStreams(loading_cache_streams_);
    }
    bool       has_requeued_prefill_stream = has_waiting_chunk(waiting_streams_.size() > waiting_size_before_cache_update);
    const bool had_active_normal_lane      = active_admission_lane_ != AdmissionLane::GROUP
                                        && (!running_streams_.empty() || !pending_decode_streams_.empty());
    // Park Decode before growing KV when a loaded context needs to rejoin this round.
    refreshAndReapTerminalStreams(pending_decode_streams_);
    if (prefill_chunk_size_ > 0 || has_requeued_prefill_stream) {
        partitionChunkContinuations(has_requeued_prefill_stream);
    }
    if (!has_requeued_prefill_stream) {
        evaluateAndUpdateStreams(running_streams_);
    }

    const size_t waiting_size_before_admission = waiting_streams_.size();

    const bool parked_active_normal_lane =
        had_active_normal_lane && has_requeued_prefill_stream && !pending_decode_streams_.empty();
    if (running_streams_.empty() && !parked_active_normal_lane) {
        active_admission_lane_ = AdmissionLane::NONE;
    }

    // Advance the group-loading lane first. It can publish to new_streams_ only
    // when the current execution batch is empty, so a ready explicit group
    // owns that boundary without being mixed with ordinary work.
    if (!parked_active_normal_lane) {
        evaluateLoadingCacheGroupQueue();
    }
    const bool loaded_group_selected = !new_streams_.empty();
    const bool had_running_batch     = !running_streams_.empty() || parked_active_normal_lane;
    if (!loaded_group_selected && active_admission_lane_ != AdmissionLane::GROUP) {
        // Poll again at the admission boundary. A load may have completed while advancing running streams.
        evaluateAndUpdateStreams(loading_cache_streams_);
        if (has_waiting_chunk(/*waiting_queue_grew=*/false)) {
            partitionChunkContinuations(/*has_waiting_prefill=*/true);
            has_requeued_prefill_stream = true;
        }
    }
    if (loaded_group_selected) {
        active_admission_lane_ = AdmissionLane::GROUP;
        prefer_group_next_     = false;
    } else if (had_running_batch) {
        // The late poll may park Decode, but the round still belongs to the running lane.
        if (active_admission_lane_ != AdmissionLane::GROUP) {
            active_admission_lane_   = AdmissionLane::NORMAL;
            const bool group_barrier = loadingGroupReady() || !waiting_group_queue_.empty();
            if (group_barrier) {
                prefer_group_next_ = true;
            }
            if (!group_barrier || usesChunkedPrefillAdmission()) {
                admitWaitingStreams(waiting_streams_, new_streams_, /*prepared_only=*/group_barrier);
            }
        }
    } else {
        // At an empty execution boundary, try the preferred lane, then the other if it cannot run.
        const bool has_waiting_group = !waiting_group_queue_.empty();
        const bool try_group_first =
            new_streams_.empty() && has_waiting_group && (prefer_group_next_ || waiting_streams_.empty());
        bool group_selected = false;
        if (try_group_first) {
            evaluateWaitingGroupQueue();
            group_selected = !new_streams_.empty();
        }
        if (new_streams_.empty() || !has_waiting_group) {
            admitWaitingStreams(waiting_streams_, new_streams_);
        }
        if (new_streams_.empty() && has_waiting_group && !try_group_first) {
            evaluateWaitingGroupQueue();
            group_selected = !new_streams_.empty();
        }
        if (!new_streams_.empty()) {
            active_admission_lane_ = group_selected ? AdmissionLane::GROUP : AdmissionLane::NORMAL;
            prefer_group_next_     = !group_selected && has_waiting_group;
        }
    }
    running_streams_.insert(running_streams_.end(), new_streams_.begin(), new_streams_.end());
    new_streams_.clear();
    updateCacheExposedWait();

    // If streams were scheduled, trigger next scheduling round
    if (waiting_streams_.size() < waiting_size_before_admission) {
        schedule_trigger_ = true;
    }

    if (has_requeued_prefill_stream) {
        // Partition after load-done streams rejoin so final-prefill streams cannot grow decode KV
        // or hide context streams behind a decode row.
        partitionChunkContinuations();
        if (!running_streams_.empty() && !running_streams_.front()->isContextStream()) {
            evaluateAndUpdateStreams(running_streams_);
        }
    }

    if (prefill_chunk_size_ > 0 && !running_streams_.empty() && running_streams_.front()->isContextStream()) {
        if (usesChunkedPrefillAdmission()) {
            // A newer request may already be ready while an older one is still
            // loading. Restore FIFO over the entire candidate pool when it joins.
            running_streams_.sort(earlierEnqueueTime);
        }
        auto prefill_batch = selectPrefillPrefix(running_streams_);
        if (!prefill_batch.empty()) {
            reportMetrics();
            last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
            return ScheduleOutput{std::move(prefill_batch)};
        }
        // All context candidates were terminal/invalid. Restore any deferred decode work now.
        if (partitionChunkContinuations()) {
            evaluateAndUpdateStreams(running_streams_);
        }
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return ScheduleOutput{running_streams_};
}

void FIFOScheduler::updateCacheExposedWait() {
    // Measure scheduler-observed stalls between batches, after both lanes had
    // a chance to provide runnable work. A group parked here may already be ready.
    const bool waiting_for_cache =
        running_streams_.empty()
        && (!loading_cache_streams_.empty()
            || (!loading_cache_group_queue_.empty()
                && std::any_of(loading_cache_group_queue_.front().begin(),
                               loading_cache_group_queue_.front().end(),
                               [](const auto& stream) { return stream->getStatus() == StreamState::LOADING_CACHE; })));
    if (telemetry::TelemetryRuntime::isActive()) {
        const auto observation_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        // When ordinary or group waiters also remain, an empty batch has more
        // than one possible blocker; do not attribute that interval to cache.
        const bool cache_only_stall =
            waiting_for_cache && waiting_streams_.empty() && waiting_group_queue_.empty() && !need_fill_fake_stream_;
        const auto observe_stream = [cache_only_stall, observation_time_us](const GenerateStreamPtr& stream) {
            if (!stream->isContextStream() || stream->getStatus() != StreamState::LOADING_CACHE) {
                return;
            }
            if (cache_only_stall) {
                stream->startCacheExposedWait(observation_time_us);
            } else {
                stream->endCacheExposedWait(observation_time_us);
            }
        };
        for (const auto& stream : loading_cache_streams_) {
            observe_stream(stream);
        }
        if (!loading_cache_group_queue_.empty()) {
            for (const auto& stream : loading_cache_group_queue_.front()) {
                observe_stream(stream);
            }
        }
    }
}

int64_t FIFOScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + groupQueueStreamsSize(waiting_group_queue_);
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::waitingTaskList() {
    std::lock_guard<mutex> lock(lock_);
    waiting_task_list_.clear();
    waiting_task_list_.reserve(waiting_streams_.size() + groupQueueStreamsSize(waiting_group_queue_));
    appendTaskInfos(waiting_task_list_, waiting_streams_);
    for (const auto& group : waiting_group_queue_) {
        appendTaskInfos(waiting_task_list_, group);
    }
    return waiting_task_list_;
}

}  // namespace rtp_llm
