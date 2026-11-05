#include "rtp_llm/cpp/engine_base/schedulers/PPScheduler.h"

#include <algorithm>
#include <chrono>

#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

PPScheduler::PPScheduler(const RuntimeConfig&                   runtime_config,
                         const ModelConfig&                     model_config,
                         const PDSepConfig&                     pd_sep_config,
                         const ParallelismConfig&               parallelism_config,
                         const SpeculativeExecutionConfig&      sp_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter):
    FIFOSchedulerBase(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      cache_manager,
                      metrics_reporter),
    max_batch_tokens_without_cache_(
        static_cast<size_t>(std::max<int64_t>(runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache, 0))),
    cp_force_single_prefill_(parallelism_config.prefill_cp_config.is_enabled()
                             && runtime_config.fifo_scheduler_config.cp_force_single_prefill),
    prefill_cp_size_(parallelism_config.prefill_cp_config.is_enabled() ?
                         static_cast<size_t>(std::max<int64_t>(parallelism_config.tp_size, 1)) :
                         1),
    sp_config_(sp_config) {}

PPScheduler::~PPScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destroy PPScheduler");
}

void PPScheduler::cancelStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto it = streams.begin(); it != streams.end();) {
        auto& stream = *it;
        stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        if (stream->moveToNext() == StreamState::FINISHED) {
            it = streams.erase(it);
        } else {
            // Retain unfinished requests for finalization after PP execution drains.
            ++it;
        }
    }
}

bool PPScheduler::refreshAndReapTerminalStreams(std::list<GenerateStreamPtr>& streams) {
    bool reaped = false;
    for (auto it = streams.begin(); it != streams.end();) {
        auto& stream = *it;
        if (!stream->isFinished()) {
            stream->checkTimeoutAndHasError();
            if (stream->isPPInflight()
                || (!stream->hasError() && !stream->hasEvent(StreamEvents::GenerateDone))) {
                ++it;
                continue;
            }
            const auto new_state = stream->moveToNext();
            if (new_state != StreamState::FINISHED) {
                RTP_LLM_LOG_ERROR("Unexpected state %d when reaping terminal stream [%ld]",
                                  static_cast<int>(new_state),
                                  stream->streamId());
                ++it;
                continue;
            }
        }
        addStreamToNewState(stream, StreamState::FINISHED);
        it     = streams.erase(it);
        reaped = true;
    }
    return reaped;
}

void PPScheduler::finalizeCancelledStreams() {
    lock_guard<mutex> lock(lock_);
    refreshAndReapTerminalStreams(waiting_streams_);
    refreshAndReapTerminalStreams(loading_cache_streams_);
    refreshAndReapTerminalStreams(running_streams_);
    refreshAndReapTerminalStreams(pending_decode_streams_);
}

void PPScheduler::cancelExtraStreams() {
    cancelStreams(pending_decode_streams_);
}

int64_t PPScheduler::extraOnflightStreams() const {
    return pending_decode_streams_.size();
}

void PPScheduler::fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {
    collector.pending_decode_stream_size = pending_decode_streams_.size();
}

void PPScheduler::appendExtraRunningTaskList(std::vector<EngineScheduleInfo::TaskInfo>& task_list) const {
    appendTaskInfos(task_list, pending_decode_streams_);
}

size_t PPScheduler::countInitedKVCacheStreams() const {
    return FIFOSchedulerBase::countInitedKVCacheStreams()
           + std::count_if(pending_decode_streams_.begin(), pending_decode_streams_.end(), [](const auto& stream) {
                 return stream->curBlocksNum() > 0;
             });
}

bool PPScheduler::canSchedule(const GenerateStreamPtr& stream) const {
    if (!stream->isPPInflight()) {
        return true;
    }
    return !stream->hasError() && !stream->isFinished() && !stream->hasEvent(StreamEvents::GenerateDone)
           && stream->hasPendingChunk();
}

bool PPScheduler::grantChunk(const GenerateStreamPtr& stream, int64_t& budget) {
    const int64_t rows      = stream->currentBatchSize();
    const int64_t begin     = stream->reuseLength();
    const int64_t remaining = stream->seqLength() - begin;
    const int64_t block     = stream->seqSizePerBlock();
    if (remaining <= 0 || block <= 0 || begin < 0 || begin % block != 0) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "invalid PP chunk window");
        return false;
    }
    const auto grant = computeChunkGrant(budget, rows, remaining, block);
    if (grant <= 0) {
        if (computeChunkGrant(prefill_chunk_size_, rows, remaining, block) <= 0) {
            stream->reportError(ErrorCode::INVALID_PARAMS,
                                "full prefill budget cannot produce a positive aligned grant");
        }
        return false;
    }
    stream->setChunkSize(static_cast<int>(grant));
    budget -= rows * grant;
    return true;
}

void PPScheduler::moveMiddleChunksToBack(std::list<GenerateStreamPtr>& streams) const {
    std::list<GenerateStreamPtr> middle;
    for (auto it = streams.begin(); it != streams.end();) {
        auto current = it++;
        if ((*current)->isMiddleChunk()) {
            middle.splice(middle.end(), streams, current);
        }
    }
    streams.splice(streams.end(), middle);
}

/** PP separates prefill and decode batches, so legacy batch_with_prefill_* metrics are not updated. */
void PPScheduler::onRunningStream(const GenerateStreamPtr& stream) {
    if (pd_sep_config_.role_type != RoleType::DECODE || sp_config_.type == SP_TYPE_NONE) {
        return;
    }
    if (stream->isContextStream() || stream->isFakeStream() || stream->isPerfTest()) {
        return;
    }

    /** This hook runs on entry to RUNNING; later decode rounds retain the last stage's proposals. */
    initializeDecodeCandidates(stream);
}

void PPScheduler::initializeDecodeCandidates(const GenerateStreamPtr& stream) const {
    /**
     * MTP/EAGLE hands off [anchor, d1]. DSpARK's padded tokens define an implicit point-mass
     * proposal for the first target verification; later rounds carry sampled tokens and dense q.
     */
    auto candidates = stream->getProposeToken();
    candidates.resize(sp_config_.gen_num_per_cycle + 1, 0);
    candidates[0] = stream->currentExecuteTokens()[0];

    auto buffer = stream->getSPOutputBuffer();
    if (!buffer) {
        buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    }
    buffer->propose_step = sp_config_.gen_num_per_cycle;
    buffer->tokens       = torch::empty({1, static_cast<int64_t>(candidates.size())},
                                  torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    std::copy(candidates.begin(), candidates.end(), buffer->tokens.data_ptr<int32_t>());
    stream->setSPOutputBuffer(buffer);
    stream->setProposeToken(candidates);
    stream->setContainProposeToken(true);
}

void PPScheduler::addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
    if (new_state == StreamState::FINISHED) {
        finished_request_ids_.push_back(stream->streamId());
    }
    FIFOSchedulerBase::addStreamToNewState(stream, new_state);
}

list<GenerateStreamPtr> PPScheduler::evaluateRunningStreams() {
    bool has_context = false;
    if (chunkedPrefillConfigured() && pd_sep_config_.role_type == RoleType::PDFUSION) {
        refreshAndReapTerminalStreams(pending_decode_streams_);
        refreshAndReapTerminalStreams(running_streams_);
        has_context = std::any_of(
            running_streams_.begin(), running_streams_.end(), [](const auto& stream) { return stream->isContextStream(); });

        // Finish the current context cohort before resuming its local decode streams.
        if (!has_context) {
            running_streams_.splice(running_streams_.begin(), pending_decode_streams_);
        }
    }

    list<GenerateStreamPtr> scheduled_streams;
    int64_t chunk_budget = prefill_chunk_size_;
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        auto stream = *it;
        if (!canSchedule(stream)) {
            ++it;
            continue;
        }

        // A final result may have changed this stream to decode after the context scan.
        if (has_context && !stream->isContextStream()) {
            auto current = it++;
            pending_decode_streams_.splice(pending_decode_streams_.end(), running_streams_, current);
            continue;
        }
        const auto new_state = stream->moveToNext();
        if (new_state != StreamState::RUNNING) {
            addStreamToNewState(stream, new_state);
            it = running_streams_.erase(it);
            continue;
        }

        if (stream->useChunkWindow() && !grantChunk(stream, chunk_budget)) {
            ++it;
            continue;
        }

        scheduled_streams.push_back(stream);
        ++it;
    }
    return scheduled_streams;
}

void PPScheduler::resetAdmissionMetrics() {
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;
}

void PPScheduler::maintainWaitingStreams() {
    const auto now_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
    auto       oldest_enqueue_time_us = now_us;

    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        auto current = it++;
        auto stream  = *current;

        /** Preserve the pre-admission age snapshot, including waiters retired below. */
        oldest_enqueue_time_us = std::min(oldest_enqueue_time_us, stream->schedulerEnqueueTimeUs());

        /** P instances only support multi-token requests through PD; reject local generation before KV admission. */
        if (pd_sep_config_.role_type == RoleType::PREFILL && !stream->queryPdSep()
            && stream->generateConfig()->max_new_tokens > 1) {
            stream->reportError(ErrorCode::INVALID_PARAMS,
                                "PP PREFILL does not support local generation with max_new_tokens > 1; "
                                "this request is not using PD separation.");
        }

        /** Mixing context and decode streams violates the speculative verify input layout.
         * Reject context streams on DECODE instances to keep scheduled batches decode-only.
         */
        if (pd_sep_config_.role_type == RoleType::DECODE && stream->isContextStream()) {
            stream->reportError(ErrorCode::INVALID_PARAMS,
                                "PP DECODE does not support local prefill; "
                                "prefill must be completed on a PREFILL instance before decode scheduling.");
        }

        /** Only errors may advance here: a persistent CanRun event must not bypass this round's admission. */
        const auto state = stream->getStatus();
        if (stream->checkTimeoutAndHasError()) {
            const auto new_state = stream->moveToNext();
            if (new_state != state) {
                addStreamToNewState(stream, new_state);
                waiting_streams_.erase(current);
            }
        }
    }
    last_waiting_oldest_age_us_ = now_us - oldest_enqueue_time_us;
}

void PPScheduler::admitWaitingStreams(list<GenerateStreamPtr>& scheduled_streams) {
    ScheduleRuntime schedule_runtime;
    initScheduleRuntime(scheduled_streams, schedule_runtime);
    int64_t chunk_budget = prefill_chunk_size_;

    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        auto current = it++;
        auto stream  = *current;

        /** CanRun is persistent and not proof of admission: the initialized-KV limit and
         * current batch limits must both be checked; a stream already owning KV may
         * continue even when the initialized-stream limit is full.
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

        const auto state                = stream->getStatus();
        const auto new_state            = stream->moveToNext();
        const bool new_inited_kv        = !already_inited_kv && stream->curBlocksNum() > 0;
        const bool new_load_inited      = !already_load_inited && stream->hasEvent(StreamEvents::LoadInitiated);
        const bool admission_progressed = new_state == StreamState::RUNNING || new_state == StreamState::LOADING_CACHE
                                          || (new_state == StreamState::WAITING && (new_inited_kv || new_load_inited));

        /** Failed allocations must leave capacity and token budgets available to later waiters. */
        if (admission_progressed) {
            // KV admission finalizes cache reuse. Grant the executable chunk using that final reuse length.
            if (new_state == StreamState::RUNNING
                && (!stream->useChunkWindow() || grantChunk(stream, chunk_budget))) {
                scheduled_streams.push_back(stream);
            }
            // Admission limits also cover streams whose execution is deferred by the chunk budget.
            // TODO: As in FIFO, these streams retain allocated KV until a later round has chunk budget.
            updateScheduleRuntime(schedule_runtime, stream, true, new_inited_kv);

            if (stream->isContextStream()) {
                ++last_admitted_context_batch_size_;
                last_admitted_context_token_size_ += stream->contextLength();
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
    schedule_runtime.concurrency_stream_count = running_streams_.size() + pending_decode_streams_.size();
    for (const auto& stream : streams) {
        updateScheduleRuntime(schedule_runtime, stream, true, false);
    }
    return fitsCurrentBatch(schedule_runtime, new_stream);
}

void PPScheduler::initScheduleRuntime(const std::list<GenerateStreamPtr>& scheduled_streams,
                                      ScheduleRuntime&                    schedule_runtime) const {
    /** Inflight requests still occupy concurrency slots, so the cap must include every running request. */
    schedule_runtime.concurrency_stream_count = running_streams_.size() + pending_decode_streams_.size();
    schedule_runtime.inited_kv_stream_count   = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;
    for (const auto& stream : scheduled_streams) {
        updateScheduleRuntime(schedule_runtime, stream, false, false);
    }
}

void PPScheduler::updateScheduleRuntime(ScheduleRuntime&         schedule_runtime,
                                        const GenerateStreamPtr& stream,
                                        bool                     new_admission,
                                        bool                     new_inited_kv) const {
    if (new_admission) {
        ++schedule_runtime.concurrency_stream_count;
    }
    if (new_inited_kv) {
        ++schedule_runtime.inited_kv_stream_count;
    }

    /** A batch must not mix prefill-only and generation requests. */
    if (!schedule_runtime.has_execution_mode) {
        schedule_runtime.has_execution_mode = true;
        schedule_runtime.prefill_only       = isPrefillOnly(stream);
    }

    /** Admitted cache loads must consume CP and token budgets, or later requests could exceed this round's limits. */
    ++schedule_runtime.scheduled_stream_count;
    schedule_runtime.scheduled_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
    schedule_runtime.scheduled_prefill_max_seq_len_with_cache =
        std::max(schedule_runtime.scheduled_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
    schedule_runtime.scheduled_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());

    if (stream->isContextStream()) {
        schedule_runtime.admitted_prefill_token_size_without_cache += prefillTokenCostWithoutCache(stream);
    }
}

bool PPScheduler::fitsCurrentBatch(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& candidate) const {
    if (schedule_runtime.has_execution_mode && schedule_runtime.prefill_only != isPrefillOnly(candidate)) {
        return false;
    }

    if (schedule_runtime.concurrency_stream_count + 1 > max_generate_batch_size_) {
        return false;
    }

    if (pd_sep_config_.role_type == RoleType::DECODE) {
        return true;
    }

    if (cp_force_single_prefill_ && schedule_runtime.scheduled_stream_count > 0) {
        return false;
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

    if (max_batch_tokens_size_ == 0) {
        return false;
    }
    /** Prefill-only produces no token; both logical and padded costs may fill the budget. */
    const auto available_tokens = max_batch_tokens_size_ - (isPrefillOnly(candidate) ? 0 : 1);
    if (schedule_runtime.scheduled_prefill_token_size_with_cache > available_tokens) {
        return false;
    }
    const auto candidate_tokens = prefillTokenCostWithCache(candidate);
    if (candidate_tokens > available_tokens - schedule_runtime.scheduled_prefill_token_size_with_cache) {
        return false;
    }

    const auto candidate_sequence_count = static_cast<size_t>(candidate->currentBatchSize());
    const auto max_seq_len =
        std::max(schedule_runtime.scheduled_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(candidate));
    if (max_seq_len == 0) {
        return true;
    }
    const auto max_sequence_count = available_tokens / max_seq_len;
    return schedule_runtime.scheduled_prefill_sequence_count <= max_sequence_count
           && candidate_sequence_count <= max_sequence_count - schedule_runtime.scheduled_prefill_sequence_count;
}

size_t PPScheduler::prefillSeqLenWithCache(const GenerateStreamPtr& stream) const {
    return static_cast<size_t>(std::max(stream->contextLength(), 0))
           + static_cast<size_t>(std::max(stream->prefixLength(), 0));
}

size_t PPScheduler::prefillTokenCostWithCache(const GenerateStreamPtr& stream) const {
    return prefillSeqLenWithCache(stream) * static_cast<size_t>(stream->currentBatchSize());
}

size_t PPScheduler::prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const {
    auto token_count = static_cast<size_t>(std::max(stream->contextLength(), 0));
    if (prefill_cp_size_ > 1) {
        token_count = makeZigzagTokenLayout(token_count, prefill_cp_size_).padded_token_count;
    }
    return token_count * static_cast<size_t>(stream->currentBatchSize());
}

bool PPScheduler::waitPredicate() {
    // Inflight streams advance PP; waiting and loading streams use timed polling.
    return stop_ || schedule_trigger_ || !running_streams_.empty() || !pending_decode_streams_.empty();
}

absl::StatusOr<ScheduleOutput> PPScheduler::schedule() {
    RTP_LLM_PROFILE_FUNCTION();
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_ || !loading_cache_streams_.empty()) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else if (!waiting_streams_.empty()) {
        cond_.wait_for(lock, std::chrono::milliseconds(100), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    schedule_trigger_ = false;
    finished_request_ids_.clear();
    // Ready decode can skip admission. Metrics still describe this scheduling round.
    resetAdmissionMetrics();

    /** 1. Handle all running streams, skip the in-flight streams. */
    auto scheduled_streams = evaluateRunningStreams();

    /** 2. Handle all loading cache streams. All of loaded-done streams are put in waiting_streams_  */
    evaluateAndUpdateStreams(loading_cache_streams_);

    /** 3. Maintain waiting streams even when the role's ready-stream policy defers all new admissions. */
    maintainWaitingStreams();
    switch (pd_sep_config_.role_type) {
        case RoleType::PDFUSION:
            /** Do not extend the context cohort while decode streams are waiting. */
            if (scheduled_streams.empty() && pending_decode_streams_.empty()) {
                admitWaitingStreams(scheduled_streams);
            }
            break;
        case RoleType::PREFILL:
            /** Keep the original PP admission boundary; PREFILL has no local decode fallback. */
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

    if (chunkedPrefillConfigured()) {
        moveMiddleChunksToBack(scheduled_streams);
    }
    for (const auto& stream : scheduled_streams) {
        stream->setPPInflight();
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return ScheduleOutput{std::move(scheduled_streams), std::move(finished_request_ids_)};
}

}  // namespace rtp_llm
