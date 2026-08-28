#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <algorithm>
#include <chrono>
#include <mutex>

using namespace std;
namespace rtp_llm {

FIFOScheduler::FIFOScheduler(const RuntimeConfig&                   runtime_config,
                             const ModelConfig&                     model_config,
                             const PDSepConfig&                     pd_sep_config,
                             const ParallelismConfig&               parallelism_config,
                             const ModelSpecificConfig&             model_specific_config,
                             const std::shared_ptr<KVCacheManager>& cache_manager,
                             const kmonitor::MetricsReporterPtr     metrics_reporter,
                             const int                              max_score_len):
    FIFOSchedulerBase(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      model_specific_config,
                      cache_manager,
                      metrics_reporter),
    cp_force_single_prefill_(parallelism_config.prefill_cp_config.is_enabled()
                             && runtime_config.fifo_scheduler_config.cp_force_single_prefill),
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
                     cp_force_single_prefill_,
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
}

bool FIFOScheduler::hasExtraStreams() const {
    return !waiting_group_queue_.empty() || !loading_cache_group_queue_.empty();
}

int64_t FIFOScheduler::extraOnflightStreams() const {
    return groupQueueStreamsSize(waiting_group_queue_) + groupQueueStreamsSize(loading_cache_group_queue_);
}

void FIFOScheduler::fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {
    collector.wait_stream_size += groupQueueStreamsSize(waiting_group_queue_);
    collector.loading_cache_stream_size += groupQueueStreamsSize(loading_cache_group_queue_);
    collector.group_fallback_count = pending_group_fallback_count_.exchange(0, std::memory_order_relaxed);
}

std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
FIFOScheduler::enqueueGroup(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<bool> enqueue_successes;
    enqueue_successes.reserve(streams.size());
    std::vector<GenerateStreamPtr> valid_streams;
    valid_streams.reserve(streams.size());
    for (const auto& stream : streams) {
        const bool success = checkInputLength(stream);
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

bool FIFOScheduler::evaluateRunningBatch(const ScheduleRuntime&   schedule_runtime,
                                         const GenerateStreamPtr& new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    const auto admitted_stream_count = schedule_runtime.admitted_stream_count;
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        // Decode-only scheduling can top up an existing running decode batch.
        // max_generate_batch_size_ is an inclusive cap; only requests above it
        // should be rejected.
        if (running_streams_.size() + admitted_stream_count + 1 <= max_generate_batch_size_) {
            return true;
        }
    }
    // prefill and decode not mixed together
    if (!running_streams_.empty()) {
        return false;
    }
    // Conservative CP prefill mode: cap at one stream per round unless
    // runtime config explicitly allows CP prefill batching.
    if (cp_force_single_prefill_ && admitted_stream_count > 0) {
        return false;
    }
    if (running_streams_.size() + admitted_stream_count + 1 > max_generate_batch_size_) {
        return false;
    }

    return fitsPrefillTokenLimits(admitted_stream_count,
                                  schedule_runtime.admitted_prefill_token_size_with_cache,
                                  schedule_runtime.admitted_prefill_max_seq_len_with_cache,
                                  schedule_runtime.admitted_prefill_sequence_count,
                                  new_stream);
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
    if (cp_force_single_prefill_ && admitted_count > 0) {
        return false;
    }
    if (running_streams_.size() + admitted_count + 1 > max_generate_batch_size_) {
        return false;
    }
    size_t admitted_tokens         = 0;
    size_t admitted_max_seq_len    = 0;
    size_t admitted_sequence_count = 0;
    for (const auto& stream : streams) {
        admitted_tokens += prefillTokenCostWithCache(stream);
        admitted_max_seq_len = std::max(admitted_max_seq_len, prefillSeqLenWithCache(stream));
        admitted_sequence_count += static_cast<size_t>(stream->currentBatchSize());
    }
    return fitsPrefillTokenLimits(
        admitted_count, admitted_tokens, admitted_max_seq_len, admitted_sequence_count, new_stream);
}

bool FIFOScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                          const GenerateStreamPtr&       new_stream) {
    // FIFOScheduler's own scheduling loop uses the counter-based evaluateRunningBatch();
    // this list-based FIFOSchedulerBase entry point delegates to the list overload so
    // both paths agree.
    return evaluateRunningBatch(streams, new_stream);
}

bool FIFOScheduler::fitsPrefillTokenLimits(size_t                   admitted_stream_count,
                                           size_t                   admitted_tokens,
                                           size_t                   admitted_max_seq_len,
                                           size_t                   admitted_sequence_count,
                                           const GenerateStreamPtr& candidate) const {
    // Preserve the historical singleton boundary; checkInputLength() has already
    // validated the candidate's standalone token cost.
    if (admitted_stream_count == 0 && candidate->contextLength() + running_streams_.size() < int(max_seq_len_)) {
        return true;
    }

    // Preserve the existing one-token reserve for each already-running stream.
    // Prefill is not mixed with a running batch, so this is normally zero, but
    // keeping it here makes both callers retain the original boundary semantics.
    const auto running_token_reserve = running_streams_.size();
    if (running_token_reserve >= max_batch_tokens_size_) {
        return false;
    }
    const auto available_tokens = max_batch_tokens_size_ - running_token_reserve;

    // Full logical token cost (including reused prefix) must be strictly below
    // max_batch_tokens_size_. Subtraction avoids overflow in the sum.
    if (admitted_tokens >= available_tokens) {
        return false;
    }
    const auto candidate_tokens = prefillTokenCostWithCache(candidate);
    if (candidate_tokens >= available_tokens - admitted_tokens) {
        return false;
    }

    // The model executes prefill as a padded rectangle. Use the full logical
    // sequence length (prefix included) and the real sequence width represented
    // by currentBatchSize(). Division avoids overflow in max_seq_len * width.
    const auto candidate_sequence_count = static_cast<size_t>(candidate->currentBatchSize());
    const auto sequence_count           = admitted_sequence_count + candidate_sequence_count;
    const auto max_seq_len              = std::max(admitted_max_seq_len, prefillSeqLenWithCache(candidate));
    return max_seq_len == 0 || sequence_count <= (available_tokens - 1) / max_seq_len;
}

size_t FIFOScheduler::prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const {
    // Match the token count that CP presents to the model after per-sequence padding.
    auto token_count = static_cast<size_t>(std::max(stream->contextLength(), 0));
    if (prefill_cp_size_ > 1) {
        token_count = makeZigzagTokenLayout(token_count, prefill_cp_size_).padded_token_count;
    }
    return token_count * static_cast<size_t>(stream->currentBatchSize());
}

size_t FIFOScheduler::prefillSeqLenWithCache(const GenerateStreamPtr& stream) const {
    // max_batch_tokens_size bounds the full logical sequence. Unlike the compute-only
    // quota, this count includes the reused prefix and therefore does not apply CP padding.
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
                   + count_inited(running_streams_) + count_inited(new_streams_);
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
    return stop_ || schedule_trigger_ || !running_streams_.empty();
}

void FIFOScheduler::admitWaitingStreams(list<GenerateStreamPtr>&       waiting_streams,
                                        const list<GenerateStreamPtr>& already_admitted_streams) {
    RTP_LLM_PROFILE_FUNCTION();
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;
    if (!waiting_streams.empty()) {
        auto oldest_enqueue_time_us = (*std::min_element(waiting_streams.begin(),
                                                         waiting_streams.end(),
                                                         [](const auto& lhs, const auto& rhs) {
                                                             return lhs->schedulerEnqueueTimeUs()
                                                                    < rhs->schedulerEnqueueTimeUs();
                                                         }))
                                          ->schedulerEnqueueTimeUs();
        last_waiting_oldest_age_us_ =
            std::max<int64_t>(0, autil::TimeUtility::currentTimeInMicroSeconds() - oldest_enqueue_time_us);
    }
    const size_t inited_kv_streams = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;

    // Explicit groups are scheduled through enqueueGroup() and the dedicated group
    // queues. group_id/group_size on a stream in waiting_streams are status metadata;
    // they must not delay or isolate ordinary FIFO admission.

    ScheduleRuntime schedule_runtime;
    for (const auto& stream : already_admitted_streams) {
        if (!stream || stream->getStatus() != StreamState::RUNNING) {
            continue;
        }
        ++schedule_runtime.admitted_stream_count;
        schedule_runtime.admitted_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
        schedule_runtime.admitted_prefill_max_seq_len_with_cache =
            std::max(schedule_runtime.admitted_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
        schedule_runtime.admitted_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());
        if (stream->isContextStream()) {
            ++last_admitted_context_batch_size_;
            last_admitted_context_token_size_ += stream->contextLength();
            schedule_runtime.admitted_prefill_token_size_without_cache += prefillTokenCostWithoutCache(stream);
        }
    }

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto  current = it++;
        auto& stream  = *current;

        if (stream->hasError()) {
            auto state     = stream->getStatus();
            auto new_state = stream->moveToNext();
            if (new_state != state) {
                addStreamToNewState(stream, new_state);
                waiting_streams.erase(current);
            }
            continue;
        }

        // Stop admitting new work once this scheduling round reaches the uncached
        // token quota. Keep scanning so errored streams behind it can still be
        // finalized.
        if (max_batch_tokens_without_cache_ > 0
            && schedule_runtime.admitted_prefill_token_size_without_cache >= max_batch_tokens_without_cache_) {
            continue;
        }

        // Check admission capacity and memory constraints.
        //
        // Some PD decode streams already carry CanRun before entering FIFO: DecodeRpcServer uses
        // CanRun to drive the pre-enqueue KV allocation path. CanRun is a permanent event, so it
        // cannot be used as proof that FIFO has admitted this stream in the current scheduling
        // round. Always run FIFO capacity checks and only advance streams admitted here.
        const bool already_inited_kv = stream->curBlocksNum() > 0;
        if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv
            && inited_kv_streams + schedule_runtime.newly_inited_kv_streams >= max_inited_kv_cache_streams_) {
            continue;
        }

        if (!evaluateRunningBatch(schedule_runtime, stream)) {
            continue;
        }

        const auto state                 = stream->getStatus();
        const bool load_initiated_before = stream->hasEvent(StreamEvents::LoadInitiated);
        if (!stream->hasEvent(StreamEvents::CanRun)) {
            stream->reportEvent(StreamEvents::CanRun);
        }

        const auto new_state           = stream->moveToNext();
        const bool kv_initialized      = !already_inited_kv && stream->curBlocksNum() > 0;
        const bool load_initiated      = !load_initiated_before && stream->hasEvent(StreamEvents::LoadInitiated);
        const bool scheduling_progress = new_state == StreamState::RUNNING || new_state == StreamState::LOADING_CACHE
                                         || (new_state == StreamState::WAITING && (kv_initialized || load_initiated));

        if (scheduling_progress) {
            // Admission limits cover every stream that made progress in this
            // round, including an async cache load. Otherwise multiple pending
            // loads can oversubscribe max_generate_batch_size before any one of
            // them reaches RUNNING.
            ++schedule_runtime.admitted_stream_count;
            schedule_runtime.admitted_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
            schedule_runtime.admitted_prefill_max_seq_len_with_cache =
                std::max(schedule_runtime.admitted_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
            schedule_runtime.admitted_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());
            if (kv_initialized) {
                ++schedule_runtime.newly_inited_kv_streams;
            }
            if (stream->isContextStream()) {
                ++last_admitted_context_batch_size_;
                last_admitted_context_token_size_ += stream->contextLength();
                schedule_runtime.admitted_prefill_token_size_without_cache += prefillTokenCostWithoutCache(stream);
            }
        }

        if (new_state != state) {
            addStreamToNewState(stream, new_state);
            waiting_streams.erase(current);
        }
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
    if (!running_streams_.empty()) {
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
        if (!evaluateRunningBatch(admitted_streams, stream)) {
            continue;
        }

        if (!stream->hasEvent(StreamEvents::CanRun)) {
            stream->reportEvent(StreamEvents::CanRun);
        }
        const auto state = stream->moveToNext();
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

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule() {
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

    evaluateAndUpdateStreams(running_streams_);

    if (running_streams_.empty()) {
        active_admission_lane_ = AdmissionLane::NONE;
    }

    // Advance the group-loading lane first. It can publish to new_streams_ only
    // when the current execution batch is empty, so a ready explicit group
    // owns that boundary without being mixed with ordinary work.
    evaluateLoadingCacheGroupQueue();
    if (!new_streams_.empty()) {
        active_admission_lane_ = AdmissionLane::GROUP;
        prefer_group_next_     = false;
    } else if (!running_streams_.empty()) {
        if (active_admission_lane_ == AdmissionLane::NONE) {
            // Defensive fallback for schedulers restored with an already
            // populated running list.
            active_admission_lane_ = AdmissionLane::NORMAL;
        }
        const bool group_barrier = loadingGroupReady() || !waiting_group_queue_.empty();
        if (active_admission_lane_ == AdmissionLane::NORMAL) {
            // Ordinary streams already loading cache belong to the active
            // NORMAL lane. Always advance them; a group arriving later is the
            // barrier for new waiting streams, not for already-owned work.
            evaluateAndUpdateStreams(loading_cache_streams_);
            if (!group_barrier) {
                admitWaitingStreams(waiting_streams_, new_streams_);
            } else {
                // Do not admit another waiting stream after a group arrives.
                // The current normal lane (including completed cache loads)
                // drains to an isolated group boundary.
                prefer_group_next_ = true;
            }
        }
    } else {
        // Poll cache loads that were admitted by the NORMAL lane before choosing
        // the next empty-batch owner. A completed load must take this boundary by
        // itself; otherwise dispatching a group below would mix the two lanes in
        // one scheduler result. A still-pending load does not block an executable
        // group, so cache I/O cannot starve group traffic.
        evaluateAndUpdateStreams(loading_cache_streams_);
        const bool has_waiting_group = !waiting_group_queue_.empty();
        if (!new_streams_.empty()) {
            active_admission_lane_ = AdmissionLane::NORMAL;
            prefer_group_next_     = has_waiting_group;
            if (!has_waiting_group) {
                // With no group boundary to protect, ordinary cache completions
                // and ordinary waiters are the same lane and may batch together.
                admitWaitingStreams(waiting_streams_, new_streams_);
            }
        }

        const bool has_normal_waiting = !waiting_streams_.empty();

        if (!new_streams_.empty()) {
            // The completed ordinary loads above own this execution boundary.
        } else if (has_waiting_group && (prefer_group_next_ || !has_normal_waiting)) {
            evaluateWaitingGroupQueue();
            if (!new_streams_.empty()) {
                active_admission_lane_ = AdmissionLane::GROUP;
                prefer_group_next_     = false;
            } else {
                // The attempted group produced no executable work. Ordinary
                // loads were already polled above; admit an ordinary waiter so
                // it can release any inited-KV slot blocking the group.
                admitWaitingStreams(waiting_streams_, new_streams_);
                if (!new_streams_.empty()) {
                    active_admission_lane_ = AdmissionLane::NORMAL;
                    prefer_group_next_     = true;
                }
            }
        } else {
            // This round has explicitly selected the NORMAL lane. Advance its
            // waiters after polling cache completions above; a queued group is
            // the barrier for the *next* execution boundary.
            admitWaitingStreams(waiting_streams_, new_streams_);
            if (!new_streams_.empty()) {
                active_admission_lane_ = AdmissionLane::NORMAL;
                prefer_group_next_     = has_waiting_group;
            } else if (has_waiting_group) {
                // Normal streams may all be blocked on KV or cache loading. No
                // normal stream entered the execution batch, so trying one
                // explicit group here preserves isolation while avoiding
                // head-of-line deadlock between the two lanes.
                evaluateWaitingGroupQueue();
                if (!new_streams_.empty()) {
                    active_admission_lane_ = AdmissionLane::GROUP;
                    prefer_group_next_     = false;
                }
            }
        }
    }
    running_streams_.insert(running_streams_.end(), new_streams_.begin(), new_streams_.end());
    new_streams_.clear();

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

int64_t FIFOScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + groupQueueStreamsSize(waiting_group_queue_);
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::waitingTaskList() {
    std::lock_guard<mutex> lock(lock_);
    waiting_task_list_.clear();
    waiting_task_list_.reserve(waiting_streams_.size() + groupQueueStreamsSize(waiting_group_queue_));
    for (const auto& stream : waiting_streams_) {
        EngineScheduleInfo::TaskInfo task_info;
        task_info.request_id    = stream->streamId();
        task_info.prefix_length = stream->prefixLength();
        task_info.input_length  = stream->inputLength();
        task_info.batch_id      = stream->groupId();
        waiting_task_list_.push_back(task_info);
    }
    for (const auto& group : waiting_group_queue_) {
        for (const auto& stream : group) {
            EngineScheduleInfo::TaskInfo task_info;
            task_info.request_id    = stream->streamId();
            task_info.prefix_length = stream->prefixLength();
            task_info.input_length  = stream->inputLength();
            task_info.batch_id      = stream->groupId();
            waiting_task_list_.push_back(task_info);
        }
    }
    return waiting_task_list_;
}

}  // namespace rtp_llm
