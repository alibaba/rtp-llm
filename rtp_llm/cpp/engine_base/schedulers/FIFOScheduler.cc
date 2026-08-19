#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

using namespace std;
namespace rtp_llm {
namespace {

bool asyncCachePrepareEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_PREPARE_CACHE");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

constexpr auto kCachePrepareRetryInterval        = std::chrono::milliseconds(250);
constexpr auto kCachePrepareResourcePollInterval = std::chrono::milliseconds(10);

}  // namespace

FIFOScheduler::FIFOScheduler(const RuntimeConfig&                   runtime_config,
                             const ModelConfig&                     model_config,
                             const PDSepConfig&                     pd_sep_config,
                             const ParallelismConfig&               parallelism_config,
                             const ModelSpecificConfig&             model_specific_config,
                             const std::shared_ptr<KVCacheManager>& cache_manager,
                             const kmonitor::MetricsReporterPtr     metrics_reporter,
                             const int                              max_score_len):
    pd_sep_config_(pd_sep_config),
    model_specific_config_(model_specific_config),
    cache_manager_(cache_manager),
    max_seq_len_(model_config.max_seq_len),
    max_batch_tokens_size_(runtime_config.fifo_scheduler_config.max_batch_tokens_size),
    max_batch_tokens_without_cache_(
        std::max<int64_t>(runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache, 0)),
    max_generate_batch_size_(runtime_config.max_generate_batch_size),
    max_inited_kv_cache_streams_(
        std::max<int64_t>(runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams, 0)),
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    prefill_cp_size_(parallelism_config.prefill_cp_config.is_enabled() ?
                         static_cast<size_t>(std::max<int64_t>(parallelism_config.tp_size, 1)) :
                         1),
    metrics_reporter_(metrics_reporter) {
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%zu], max_batch_tokens_size is [%zu], "
                     "max_batch_tokens_without_cache is [%zu], prefill_cp_size is [%zu], "
                     "max_inited_kv_cache_streams is [%zu]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_,
                     max_batch_tokens_without_cache_,
                     prefill_cp_size_,
                     max_inited_kv_cache_streams_);
    if (asyncCachePrepareEnabled() && pd_sep_config_.role_type != RoleType::DECODE && parallelism_config.tp_rank == 0) {
        try {
            async_cache_prepare_enabled_ = true;
            cache_prepare_thread_        = std::thread([this]() { cachePrepareLoop(); });
            RTP_LLM_LOG_INFO("async cache prepare enabled");
        } catch (const std::exception& e) {
            async_cache_prepare_enabled_ = false;
            RTP_LLM_LOG_WARNING("cache prepare worker start failed, use scheduler fallback: %s", e.what());
        }
    }
}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory FIFOScheduler");
}

bool FIFOScheduler::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_streams_.empty() && loading_cache_streams_.empty() && running_streams_.empty()
           && waiting_group_queue_.empty() && loading_cache_group_queue_.empty();
}

void FIFOScheduler::cancelStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        stream->moveToNext();
    }
    streams.clear();
}

void FIFOScheduler::cancelGroups(StreamGroupQueue& group_queue) {
    for (auto& group : group_queue) {
        cancelStreams(group);
    }
    group_queue.clear();
}

bool FIFOScheduler::finalizeErroredStreams(std::list<GenerateStreamPtr>& streams) {
    bool removed = false;
    for (auto it = streams.begin(); it != streams.end();) {
        auto& stream = *it;
        stream->checkTimeout();
        if (stream == cache_prepare_inflight_stream_) {
            // prepareCache() mutates the stream's cache resource without
            // holding lock_. Defer state transition/resource release until the
            // worker publishes completion.
            ++it;
            continue;
        }
        if (!stream->hasEvent(StreamEvents::Error)) {
            ++it;
            continue;
        }
        const auto state     = stream->getStatus();
        const auto new_state = stream->moveToNext();
        if (new_state == state) {
            ++it;
            continue;
        }
        addStreamToNewState(stream, new_state);
        it      = streams.erase(it);
        removed = true;
    }
    return removed;
}

bool FIFOScheduler::finalizeErroredGroups(StreamGroupQueue& group_queue) {
    bool removed       = false;
    bool changed_front = false;
    for (auto it = group_queue.begin(); it != group_queue.end();) {
        const bool is_front      = it == group_queue.begin();
        const bool group_changed = finalizeErroredStreams(*it);
        removed                  = group_changed || removed;
        changed_front            = changed_front || (is_front && group_changed);
        if (it->empty()) {
            removed       = true;
            changed_front = changed_front || is_front;
            it            = group_queue.erase(it);
        } else {
            ++it;
        }
    }
    if (changed_front) {
        waiting_group_yields_cache_prepare_ = false;
    }
    return removed;
}

absl::Status FIFOScheduler::stop() {
    RTP_LLM_LOG_INFO("stop FIFOScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
        if (!cache_prepare_thread_.joinable()) {
            cancelStreams(waiting_streams_);
            cancelStreams(loading_cache_streams_);
            cancelStreams(running_streams_);
            cancelGroups(waiting_group_queue_);
            cancelGroups(loading_cache_group_queue_);
        }
    }
    cond_.notify_all();
    if (!cache_prepare_thread_.joinable()) {
        return absl::OkStatus();
    }
    cache_prepare_thread_.join();
    {
        lock_guard<mutex> lock(lock_);
        cancelStreams(waiting_streams_);
        cancelStreams(loading_cache_streams_);
        cancelStreams(running_streams_);
        cancelGroups(waiting_group_queue_);
        cancelGroups(loading_cache_group_queue_);
        clearCachePrepareBlocked();
    }
    return absl::OkStatus();
}

int64_t FIFOScheduler::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

bool FIFOScheduler::checkInputLength(const GenerateStreamPtr& stream) {
    const auto input_length = static_cast<size_t>(stream->inputLength());
    const auto reserve_step = stream->reserveStep();
    if (reserve_step > 0 && !(input_length <= max_seq_len_ && reserve_step <= max_seq_len_ - input_length)) {
        const auto allowed_input_length = reserve_step <= max_seq_len_ ? max_seq_len_ - reserve_step : 0;
        auto       error_info =
            autil::StringUtil::formatString("input len %zu with speculative reserve_step %zu exceeds max seq len %zu, "
                                            "allowed max input len for speculative decoding is %zu",
                                            input_length,
                                            reserve_step,
                                            max_seq_len_,
                                            allowed_input_length);
        stream->reportError(ErrorCode::LONG_PROMPT_ERROR, error_info);
        return false;
    }
    if (stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
        stream->reportError(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                            autil::StringUtil::formatString("input len " + std::to_string(stream->inputLength())
                                                            + " is greater than kv cache max available tokens num "
                                                            + std::to_string(cache_manager_->maxAvailableTokensNum())));
        return false;  // Input length exceeds max available tokens
    }

    const auto input_token_cost = input_length * static_cast<size_t>(stream->currentBatchSize());
    if (input_token_cost > max_batch_tokens_size_) {
        auto error_info =
            autil::StringUtil::formatString("input len [%d] * batch size [%d] > max_batch_tokens_size [%zu]",
                                            stream->inputLength(),
                                            stream->currentBatchSize(),
                                            max_batch_tokens_size_);
        stream->reportError(ErrorCode::MALLOC_FAILED, error_info);
        return false;
    }
    return true;
}

absl::Status FIFOScheduler::enqueue(const GenerateStreamPtr& stream) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!checkInputLength(stream)) {
        return absl::InvalidArgumentError("Check input length failed");
    }
    {
        std::lock_guard<std::mutex> lock(lock_);
        stream->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds());
        waiting_streams_.emplace_back(stream);
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return absl::OkStatus();
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
            if (waiting_group_queue_.empty()) {
                waiting_group_yields_cache_prepare_ = false;
            }
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
    const auto admitted_running_stream_count = schedule_runtime.admitted_running_stream_count;
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        // Decode-only scheduling can top up an existing running decode batch.
        // max_generate_batch_size_ is an inclusive cap; only requests above it
        // should be rejected.
        if (running_streams_.size() + admitted_running_stream_count + 1 <= max_generate_batch_size_) {
            return true;
        }
    }
    // prefill and decode not mixed together
    if (!running_streams_.empty()) {
        return false;
    }
    if (running_streams_.size() + admitted_running_stream_count + 1 > max_generate_batch_size_) {
        return false;
    }

    return fitsPrefillTokenLimits(admitted_running_stream_count,
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
    auto count_inited = [this](const list<GenerateStreamPtr>& streams) {
        size_t count = 0;
        for (const auto& stream : streams) {
            // The worker reserves this slot before releasing lock_. Do not
            // inspect its block table while prepareCache() may be mutating it.
            if (stream == cache_prepare_inflight_stream_) {
                continue;
            }
            if (stream && stream->curBlocksNum() > 0) {
                ++count;
            }
        }
        return count;
    };

    size_t count = cache_prepare_inflight_stream_ ? 1 : 0;
    count += count_inited(waiting_streams_) + count_inited(loading_cache_streams_) + count_inited(running_streams_)
             + count_inited(new_streams_);
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

bool FIFOScheduler::canPrepareCacheBeforeAdmission(const GenerateStreamPtr& stream) const {
    if (!stream || max_generate_batch_size_ == 0) {
        return false;
    }
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        return true;
    }

    // Evaluate the candidate against an otherwise-empty execution boundary.
    // Async preparation must not reserve KV for a request that can never pass
    // the scheduler's static singleton limits. The synchronous path does not
    // allocate until after this same admission check.
    const auto context_length = static_cast<size_t>(std::max(stream->contextLength(), 0));
    if (context_length < max_seq_len_) {
        return true;
    }
    const auto candidate_tokens = prefillTokenCostWithCache(stream);
    if (candidate_tokens >= max_batch_tokens_size_) {
        return false;
    }
    const auto sequence_count = static_cast<size_t>(stream->currentBatchSize());
    const auto max_seq_len    = prefillSeqLenWithCache(stream);
    return max_seq_len == 0 || sequence_count <= (max_batch_tokens_size_ - 1) / max_seq_len;
}

FIFOScheduler::AdmissionLane FIFOScheduler::cachePrepareLane() const {
    const bool has_normal = !waiting_streams_.empty();
    const bool has_group  = !waiting_group_queue_.empty();
    if (!has_group) {
        return has_normal ? AdmissionLane::NORMAL : AdmissionLane::NONE;
    }
    if (!has_normal) {
        return AdmissionLane::GROUP;
    }

    // Mirror schedule()'s lane ownership. While a NORMAL batch drains, the
    // head group owns the next boundary. While a GROUP batch runs (or loads),
    // ordinary FIFO work may prepare for the following boundary.
    if (!running_streams_.empty()) {
        if (active_admission_lane_ == AdmissionLane::NORMAL) {
            return AdmissionLane::GROUP;
        }
        if (active_admission_lane_ == AdmissionLane::GROUP) {
            return AdmissionLane::NORMAL;
        }
    }
    if (!loading_cache_group_queue_.empty()) {
        return AdmissionLane::NORMAL;
    }
    if (!loading_cache_streams_.empty()) {
        return AdmissionLane::GROUP;
    }
    if (prefer_group_next_ && !waiting_group_yields_cache_prepare_) {
        return AdmissionLane::GROUP;
    }

    // Inspect the strict FIFO prefix. A prepared member before the first
    // dynamic blocker can publish the NORMAL boundary. If there is no such
    // member, a stalled load/allocation (or a queue containing only statically
    // inadmissible requests) yields preparation to the group lane.
    bool normal_ready_prefix = false;
    for (const auto& stream : waiting_streams_) {
        if (!stream || stream->hasEvent(StreamEvents::Error)) {
            continue;
        }
        if (stream->hasEvent(StreamEvents::CachePrepared)) {
            normal_ready_prefix = true;
            continue;
        }
        if (!canPrepareCacheBeforeAdmission(stream) && !stream->hasEvent(StreamEvents::LoadInitiated)
            && stream->curBlocksNum() == 0) {
            continue;
        }
        if (stream == cache_prepare_inflight_stream_) {
            return AdmissionLane::NORMAL;
        }
        if (isCachePrepareBlocked(stream) || stream->hasEvent(StreamEvents::LoadInitiated)
            || stream->curBlocksNum() > 0) {
            return normal_ready_prefix ? AdmissionLane::NORMAL : AdmissionLane::GROUP;
        }
        return AdmissionLane::NORMAL;
    }
    return normal_ready_prefix ? AdmissionLane::NORMAL : AdmissionLane::GROUP;
}

bool FIFOScheduler::isCachePrepareBlocked(const GenerateStreamPtr& stream) const {
    return stream
           && std::find(cache_prepare_blocked_streams_.begin(), cache_prepare_blocked_streams_.end(), stream)
                  != cache_prepare_blocked_streams_.end();
}

bool FIFOScheduler::hasCachePrepareBlocker(const StreamGroup& streams) const {
    return std::any_of(
        streams.begin(), streams.end(), [this](const auto& stream) { return isCachePrepareBlocked(stream); });
}

bool FIFOScheduler::hasErroredCachePrepareBlocker() const {
    return std::any_of(cache_prepare_blocked_streams_.begin(),
                       cache_prepare_blocked_streams_.end(),
                       [](const auto& stream) { return !stream || stream->hasEvent(StreamEvents::Error); });
}

void FIFOScheduler::markCachePrepareBlocked(const GenerateStreamPtr& stream) {
    if (!stream || isCachePrepareBlocked(stream)) {
        return;
    }
    const auto available_blocks = cache_manager_ ? cache_manager_->availableBlocksNumPerPool() : std::vector<size_t>{};
    const auto inited_streams   = countInitedKVCacheStreams();
    if (cache_prepare_blocked_streams_.empty()) {
        cache_prepare_blocked_stream_           = stream;
        cache_prepare_blocked_available_blocks_ = available_blocks;
        cache_prepare_blocked_inited_streams_   = inited_streams;
        cache_prepare_retry_at_                 = std::chrono::steady_clock::now() + kCachePrepareRetryInterval;
    } else if (available_blocks.size() == cache_prepare_blocked_available_blocks_.size()) {
        // A stable group scan may successfully allocate a later small member
        // between two failures. Track the per-pool low-water mark across all
        // blockers so releasing that member wakes a still-runnable later
        // blocker even when availability only returns to the first blocker's
        // original snapshot.
        for (size_t i = 0; i < available_blocks.size(); ++i) {
            cache_prepare_blocked_available_blocks_[i] =
                std::min(cache_prepare_blocked_available_blocks_[i], available_blocks[i]);
        }
        cache_prepare_blocked_inited_streams_ = std::max(cache_prepare_blocked_inited_streams_, inited_streams);
    } else {
        // Allocator pool topology is expected to be stable. If it changes,
        // force the resource comparison to observe a different shape.
        cache_prepare_blocked_available_blocks_ = available_blocks;
        cache_prepare_blocked_inited_streams_   = inited_streams;
    }
    cache_prepare_blocked_streams_.push_back(stream);
}

void FIFOScheduler::clearCachePrepareBlocked() {
    cache_prepare_blocked_stream_.reset();
    cache_prepare_blocked_streams_.clear();
    cache_prepare_blocked_available_blocks_.clear();
    cache_prepare_blocked_inited_streams_ = 0;
    cache_prepare_retry_at_               = {};
}

bool FIFOScheduler::cachePrepareResourcesChanged() const {
    if (cache_prepare_blocked_streams_.empty()) {
        return false;
    }
    bool cache_availability_increased = false;
    if (cache_manager_) {
        const auto available_blocks  = cache_manager_->availableBlocksNumPerPool();
        cache_availability_increased = available_blocks.size() != cache_prepare_blocked_available_blocks_.size();
        for (size_t i = 0; !cache_availability_increased && i < available_blocks.size(); ++i) {
            cache_availability_increased = available_blocks[i] > cache_prepare_blocked_available_blocks_[i];
        }
    }
    const bool released_inited_slot = countInitedKVCacheStreams() < cache_prepare_blocked_inited_streams_;
    return cache_availability_increased || released_inited_slot;
}

bool FIFOScheduler::cachePrepareRetryDue() const {
    return !cache_prepare_blocked_streams_.empty() && std::chrono::steady_clock::now() >= cache_prepare_retry_at_;
}

void FIFOScheduler::accountBatchMetrics(const GenerateStreamPtr& new_stream) {
    for (auto& stream : running_streams_) {
        stream->incBatchWithPrefillTimes(1);
        stream->incBatchWithPrefillLen(new_stream->currentExecuteTokenSize());
    }
}

bool FIFOScheduler::waitPredicate() {
    if (!async_cache_prepare_enabled_) {
        return stop_ || schedule_trigger_ || !waiting_streams_.empty() || !loading_cache_streams_.empty()
               || !running_streams_.empty() || !waiting_group_queue_.empty() || !loading_cache_group_queue_.empty();
    }
    return stop_ || schedule_trigger_ || !loading_cache_streams_.empty() || !running_streams_.empty()
           || !loading_cache_group_queue_.empty();
}

void FIFOScheduler::cachePrepareLoop() {
    cudaPreRun(static_cast<int>(getDeviceId()));
    while (!stop_.load(std::memory_order_acquire)) {
        std::vector<GenerateStreamPtr> streams;
        bool                           normal_pass_unblocked = false;
        bool                           group_pass_unblocked  = false;
        bool                           scans_front_group     = false;
        {
            std::unique_lock<std::mutex> lock(lock_);
            const auto                   has_prepare_work = [this]() {
                if (stop_) {
                    return true;
                }
                if (!cache_prepare_blocked_streams_.empty()
                    && (hasErroredCachePrepareBlocker() || cachePrepareResourcesChanged() || cachePrepareRetryDue())) {
                    return true;
                }
                const auto needs_prepare = [this](const GenerateStreamPtr& stream, bool allow_fresh_allocation) {
                    return stream && !stream->hasEvent(StreamEvents::Error)
                           && !stream->hasEvent(StreamEvents::CachePrepared)
                           && (stream->hasEvent(StreamEvents::LoadInitiated) || stream->curBlocksNum() > 0
                               || (allow_fresh_allocation && canPrepareCacheBeforeAdmission(stream)));
                };
                const auto lane               = cachePrepareLane();
                const bool normal_has_blocker = hasCachePrepareBlocker(waiting_streams_);
                const bool allow_fresh_normal = lane == AdmissionLane::NORMAL && !normal_has_blocker;
                for (const auto& stream : waiting_streams_) {
                    if (!isCachePrepareBlocked(stream) && needs_prepare(stream, allow_fresh_normal)) {
                        return true;
                    }
                }
                // Only prepare the head group. Preparing later groups would let
                // them reserve KV blocks ahead of the scheduler's group order.
                if (!waiting_group_queue_.empty()) {
                    const auto& group             = waiting_group_queue_.front();
                    const bool  group_has_blocker = hasCachePrepareBlocker(group);
                    const bool  allow_fresh_group = lane == AdmissionLane::GROUP && !group_has_blocker;
                    for (const auto& stream : group) {
                        if (!isCachePrepareBlocked(stream) && needs_prepare(stream, allow_fresh_group)) {
                            return true;
                        }
                    }
                }
                return false;
            };
            if (!cache_prepare_blocked_streams_.empty()) {
                if (!cond_.wait_for(lock, kCachePrepareResourcePollInterval, has_prepare_work)) {
                    continue;
                }
            } else {
                cond_.wait(lock, has_prepare_work);
            }
            if (stop_) {
                return;
            }
            if (!cache_prepare_blocked_streams_.empty()
                && (hasErroredCachePrepareBlocker() || cachePrepareResourcesChanged() || cachePrepareRetryDue())) {
                clearCachePrepareBlocked();
                waiting_group_yields_cache_prepare_ = false;
            }
            normal_pass_unblocked = !hasCachePrepareBlocker(waiting_streams_);
            group_pass_unblocked =
                waiting_group_queue_.empty() || !hasCachePrepareBlocker(waiting_group_queue_.front());
            streams.assign(waiting_streams_.begin(), waiting_streams_.end());
            if (!waiting_group_queue_.empty()) {
                const auto& group = waiting_group_queue_.front();
                streams.insert(streams.end(), group.begin(), group.end());
            }
        }

        bool has_pending = false;
        for (const auto& stream : streams) {
            bool is_normal_waiter      = false;
            bool is_front_group_waiter = false;
            bool fresh_prepare_allowed = false;
            {
                std::lock_guard<std::mutex> lock(lock_);
                if (stop_) {
                    return;
                }
                is_normal_waiter =
                    std::find(waiting_streams_.begin(), waiting_streams_.end(), stream) != waiting_streams_.end();
                is_front_group_waiter =
                    !waiting_group_queue_.empty()
                    && std::find(waiting_group_queue_.front().begin(), waiting_group_queue_.front().end(), stream)
                           != waiting_group_queue_.front().end();
                if (is_front_group_waiter && !scans_front_group) {
                    // Mark only the portion of the pass that actually scans
                    // the front group. Processing a long NORMAL prefix must
                    // not make group evaluation look busy.
                    scans_front_group                     = true;
                    cache_prepare_group_scan_in_progress_ = true;
                }
                if (!is_normal_waiter && !is_front_group_waiter) {
                    continue;
                }
                if (!stream || stream->hasEvent(StreamEvents::Error) || stream->hasEvent(StreamEvents::CachePrepared)) {
                    continue;
                }
                if (isCachePrepareBlocked(stream)) {
                    continue;
                }
                const bool owns_prepare_resource =
                    stream->hasEvent(StreamEvents::LoadInitiated) || stream->curBlocksNum() > 0;
                // Lane ownership may change after the worker snapshots this
                // pass (for example, opposite-lane work can enqueue). Recheck
                // at the allocation linearization point so a stale permit
                // cannot reserve KV across the new boundary.
                const auto lane = cachePrepareLane();
                fresh_prepare_allowed =
                    (is_normal_waiter && lane == AdmissionLane::NORMAL && normal_pass_unblocked)
                    || (is_front_group_waiter && lane == AdmissionLane::GROUP && group_pass_unblocked);
                if (!owns_prepare_resource && (!fresh_prepare_allowed || !canPrepareCacheBeforeAdmission(stream))) {
                    continue;
                }
                const bool already_inited = stream->curBlocksNum() > 0;
                if (max_inited_kv_cache_streams_ > 0 && !already_inited
                    && countInitedKVCacheStreams() >= max_inited_kv_cache_streams_) {
                    markCachePrepareBlocked(stream);
                    schedule_trigger_ = true;
                    cond_.notify_all();
                    if (is_normal_waiter && fresh_prepare_allowed) {
                        break;
                    }
                    continue;
                }
                cache_prepare_inflight_stream_ = stream;
            }

            CachePrepareResult result = CachePrepareResult::DONE;
            try {
                result = stream->prepareCache();
            } catch (const std::exception& e) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, std::string("async cache prepare failed: ") + e.what());
            } catch (...) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, "async cache prepare failed with unknown exception");
            }
            if (stream->hasEvent(StreamEvents::Error) && !stream->hasEvent(StreamEvents::CachePrepared)) {
                stream->reportEvent(StreamEvents::CachePrepared);
            }

            {
                std::lock_guard<std::mutex> lock(lock_);
                RTP_LLM_CHECK_WITH_INFO(cache_prepare_inflight_stream_ == stream,
                                        "cache prepare inflight stream changed unexpectedly");
                cache_prepare_inflight_stream_.reset();
                if (result == CachePrepareResult::LACK_MEM) {
                    // Snapshot after prepareCache() rolls back any partial
                    // multi-pool allocation. This prevents the request's own
                    // rollback from looking like an external resource change.
                    // The bounded retry closes the release-before-snapshot
                    // lost-wakeup window without busy-spinning on a stable
                    // shortage.
                    markCachePrepareBlocked(stream);
                    schedule_trigger_ = true;
                }
                if (result == CachePrepareResult::DONE) {
                    schedule_trigger_ = true;
                }
                cond_.notify_all();
            }
            if (result == CachePrepareResult::LACK_MEM) {
                // Ordinary FIFO remains strict. An explicit group uses the
                // scheduler's existing stable-greedy semantics: continue this
                // pass so a later smaller member can still be prepared.
                if (is_normal_waiter && fresh_prepare_allowed) {
                    break;
                }
                continue;
            }
            has_pending = has_pending || result == CachePrepareResult::WAIT;
        }
        if (scans_front_group) {
            std::lock_guard<std::mutex> lock(lock_);
            cache_prepare_group_scan_in_progress_ = false;
            schedule_trigger_                     = true;
            cond_.notify_all();
        }
        if (has_pending) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void FIFOScheduler::evaluateAndUpdateStreams(list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
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

void FIFOScheduler::evaluateWaitingStreams(list<GenerateStreamPtr>&       waiting_streams,
                                           const list<GenerateStreamPtr>& already_admitted_streams) {
    RTP_LLM_PROFILE_FUNCTION();
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;
    if (!waiting_streams.empty()) {
        auto oldest_enqueue_time_us =
            (*std::min_element(waiting_streams.begin(), waiting_streams.end(), [](const auto& lhs, const auto& rhs) {
                return lhs->schedulerEnqueueTimeUs() < rhs->schedulerEnqueueTimeUs();
            }))->schedulerEnqueueTimeUs();
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
        ++schedule_runtime.admitted_running_stream_count;
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

    bool cache_prepare_admission_blocked = false;
    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto  current = it++;
        auto& stream  = *current;

        // Async preparation preserves strict FIFO admission: later streams must
        // not pass an earlier stream whose KV allocation or cache load is
        // pending. Continue scanning only to finalize later errors/timeouts so
        // their resources cannot deadlock the head stream. A statically
        // inadmissible stream does not own prepare resources and preserves the
        // synchronous scheduler's behavior of yielding to later valid work.
        if (async_cache_prepare_enabled_) {
            if (cache_prepare_admission_blocked) {
                continue;
            }
            if (!stream->hasEvent(StreamEvents::CachePrepared)) {
                const bool prepare_pending = stream == cache_prepare_inflight_stream_
                                             || stream->hasEvent(StreamEvents::LoadInitiated)
                                             || stream->curBlocksNum() > 0 || canPrepareCacheBeforeAdmission(stream);
                cache_prepare_admission_blocked = prepare_pending;
                continue;
            }
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
            if (new_state == StreamState::RUNNING) {
                ++schedule_runtime.admitted_running_stream_count;
                schedule_runtime.admitted_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
                schedule_runtime.admitted_prefill_max_seq_len_with_cache =
                    std::max(schedule_runtime.admitted_prefill_max_seq_len_with_cache, prefillSeqLenWithCache(stream));
                schedule_runtime.admitted_prefill_sequence_count += static_cast<size_t>(stream->currentBatchSize());
            }
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

void FIFOScheduler::addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
    switch (new_state) {
        case StreamState::WAITING:
            waiting_streams_.push_back(stream);
            break;
        case StreamState::LOADING_CACHE:
            loading_cache_streams_.push_back(stream);
            break;
        case StreamState::RUNNING:
            accountBatchMetrics(stream);
            new_streams_.push_back(stream);
            break;
        case StreamState::FINISHED:
            break;
        default:
            RTP_LLM_LOG_ERROR("Unknown state: %d for stream [%ld]", static_cast<int>(new_state), stream->streamId());
            break;
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
        || waiting_group_queue_.empty() || cache_prepare_inflight_stream_ || cache_prepare_group_scan_in_progress_) {
        return;
    }

    // Evaluate at most one waiting group per scheduling round.
    auto&        group                  = waiting_group_queue_.front();
    const size_t original_size          = group.size();
    const bool   blocked_by_kv_shortage = async_cache_prepare_enabled_ && hasCachePrepareBlocker(group);
    const bool   has_unprepared_candidate =
        async_cache_prepare_enabled_ && std::any_of(group.begin(), group.end(), [this](const auto& stream) {
            return !stream->hasEvent(StreamEvents::CachePrepared) && !isCachePrepareBlocked(stream)
                   && (stream->hasEvent(StreamEvents::LoadInitiated) || stream->curBlocksNum() > 0
                       || canPrepareCacheBeforeAdmission(stream));
        });
    if (has_unprepared_candidate && !blocked_by_kv_shortage) {
        // Cache preparation may overlap the current GPU round. Normally the
        // group waits for the worker to finish its stable scan. Static-only
        // residual members do not participate in this barrier, and a completed
        // KV-shortage scan may dispatch its prepared subset below.
        const bool waiting_on_owned_resource = std::any_of(group.begin(), group.end(), [](const auto& stream) {
            return !stream->hasEvent(StreamEvents::CachePrepared)
                   && (stream->hasEvent(StreamEvents::LoadInitiated) || stream->curBlocksNum() > 0);
        });
        if (waiting_on_owned_resource && !waiting_group_yields_cache_prepare_) {
            // A group waiting on remote cache I/O has no executable batch yet.
            // Let the other lane prepare while continuing to poll the group's
            // already-owned resource.
            waiting_group_yields_cache_prepare_ = true;
            cond_.notify_all();
        }
        return;
    }

    StreamGroup  admitted_streams;
    const size_t inited_kv_streams       = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;
    size_t       newly_inited_kv_streams = 0;
    for (auto it = group.begin(); it != group.end();) {
        auto  current = it++;
        auto& stream  = *current;
        if (async_cache_prepare_enabled_ && !stream->hasEvent(StreamEvents::CachePrepared)) {
            continue;
        }
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
        waiting_group_yields_cache_prepare_ = false;
        cond_.notify_all();
        if (admitted_streams.empty()) {
            return;
        }
        dispatchPreparedGroup(admitted_streams);
        return;
    }

    if (!admitted_streams.empty()) {
        waiting_group_yields_cache_prepare_ = false;
        RTP_LLM_LOG_DEBUG("group partially admitted; keeping residual group at queue head: "
                          "original=%zu admitted=%zu deferred=%zu",
                          original_size,
                          admitted_streams.size(),
                          group.size());
        dispatchPreparedGroup(admitted_streams);
        pending_group_fallback_count_.fetch_add(1, std::memory_order_relaxed);
    } else if (async_cache_prepare_enabled_) {
        // schedule() falls back to the other lane when the selected group made
        // no progress. Mirror that decision in the prepare worker so ordinary
        // FIFO work can make the next boundary ready without mixing lanes.
        const bool yield_state_changed      = !waiting_group_yields_cache_prepare_;
        waiting_group_yields_cache_prepare_ = true;
        if (yield_state_changed) {
            cond_.notify_all();
        }
    }
}

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    const bool         async_cache_prepare = async_cache_prepare_enabled_;
    const bool         needs_periodic_poll =
        async_cache_prepare
        && (!waiting_streams_.empty() || !waiting_group_queue_.empty() || cache_prepare_blocked_stream_);
    if (need_fill_fake_stream_ || needs_periodic_poll) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }
    if (stop_) {
        // stop() joins the async prepare worker before cancelling queues. A
        // schedule thread awakened in that window must not dispatch newly
        // prepared work that stop() is about to cancel and release.
        schedule_trigger_ = false;
        return std::list<GenerateStreamPtr>{};
    }
    const size_t prepare_waiting_size_before     = waiting_streams_.size();
    const size_t prepare_group_queue_size_before = waiting_group_queue_.size();
    const size_t prepare_group_front_size_before =
        waiting_group_queue_.empty() ? 0 : waiting_group_queue_.front().size();
    const size_t prepare_loading_size_before             = loading_cache_streams_.size();
    const size_t prepare_loading_group_queue_size_before = loading_cache_group_queue_.size();
    const size_t prepare_loading_group_front_size_before =
        loading_cache_group_queue_.empty() ? 0 : loading_cache_group_queue_.front().size();
    const size_t prepare_running_size_before  = running_streams_.size();
    const auto*  prepare_waiting_front_before = waiting_streams_.empty() ? nullptr : waiting_streams_.front().get();
    const auto*  prepare_group_front_before   = waiting_group_queue_.empty() || waiting_group_queue_.front().empty() ?
                                                    nullptr :
                                                    waiting_group_queue_.front().front().get();
    const auto*  prepare_loading_group_front_before =
        loading_cache_group_queue_.empty() || loading_cache_group_queue_.front().empty() ?
             nullptr :
             loading_cache_group_queue_.front().front().get();
    const auto* prepare_running_front_before = running_streams_.empty() ? nullptr : running_streams_.front().get();
    const auto  prepare_active_lane_before   = active_admission_lane_;
    const bool  prepare_prefer_group_before  = prefer_group_next_;
    const bool  prepare_group_yield_before   = waiting_group_yields_cache_prepare_;
    schedule_trigger_                        = false;
    last_admitted_context_batch_size_        = 0;
    last_admitted_context_token_size_        = 0;
    last_waiting_oldest_age_us_              = 0;

    const auto previous_active_admission_lane = active_admission_lane_;
    evaluateAndUpdateStreams(running_streams_);

    bool finalized_waiting = finalizeErroredStreams(waiting_streams_);
    finalized_waiting      = finalizeErroredStreams(loading_cache_streams_) || finalized_waiting;
    finalized_waiting      = finalizeErroredGroups(waiting_group_queue_) || finalized_waiting;
    if (waiting_group_queue_.empty() && loading_cache_group_queue_.empty()) {
        // Do not carry a turn owned by a cancelled/timed-out group across an
        // idle boundary. A group arriving while NORMAL is still active will
        // establish a fresh barrier below.
        prefer_group_next_ = false;
    }
    if (finalized_waiting) {
        cond_.notify_all();
    }

    if (running_streams_.empty()) {
        // A group that arrived while a NORMAL batch was running owns the next
        // empty execution boundary. Preserve that barrier when the last normal
        // stream finishes; otherwise an ordinary tail can prepare first and
        // consume the KV blocks that the already-blocked group is waiting for.
        if (previous_active_admission_lane == AdmissionLane::NORMAL && !waiting_group_queue_.empty()) {
            prefer_group_next_ = true;
        }
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
                evaluateWaitingStreams(waiting_streams_, new_streams_);
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
                evaluateWaitingStreams(waiting_streams_, new_streams_);
            }
        }

        const bool has_normal_waiting = !waiting_streams_.empty();

        if (!new_streams_.empty()) {
            // The completed ordinary loads above own this execution boundary.
        } else if (has_waiting_group && (prefer_group_next_ || !has_normal_waiting)) {
            const bool cache_prepare_busy = cache_prepare_inflight_stream_ || cache_prepare_group_scan_in_progress_;
            evaluateWaitingGroupQueue();
            if (!new_streams_.empty()) {
                active_admission_lane_ = AdmissionLane::GROUP;
                prefer_group_next_     = false;
            } else if (!async_cache_prepare || (!cache_prepare_busy && waiting_group_yields_cache_prepare_)) {
                // The attempted group produced no executable work. Ordinary
                // loads were already polled above; admit an ordinary waiter so
                // it can release any inited-KV slot blocking the group. An
                // async group may yield only after its evaluator explicitly
                // classifies the wait as owned remote I/O, resource shortage,
                // or static no-progress. Fresh work waiting for the worker to
                // start, an in-flight prepare, and a stable group scan all keep
                // the boundary empty; otherwise a prepared ordinary tail could
                // cross the group barrier.
                evaluateWaitingStreams(waiting_streams_, new_streams_);
                if (!new_streams_.empty()) {
                    active_admission_lane_ = AdmissionLane::NORMAL;
                    prefer_group_next_     = true;
                }
            }
        } else {
            // This round has explicitly selected the NORMAL lane. Advance its
            // waiters after polling cache completions above; a queued group is
            // the barrier for the *next* execution boundary.
            evaluateWaitingStreams(waiting_streams_, new_streams_);
            if (!new_streams_.empty()) {
                active_admission_lane_ = AdmissionLane::NORMAL;
                prefer_group_next_     = has_waiting_group;
            } else if (has_waiting_group && (!async_cache_prepare || cachePrepareLane() == AdmissionLane::GROUP)) {
                // Normal streams may all be blocked on KV or cache loading. No
                // normal stream entered the execution batch. An async NORMAL
                // lane with fresh or in-flight preparation still owns this
                // boundary; only a lane that has explicitly yielded on cache
                // I/O, KV shortage, or static rejection may fall back to the
                // group here.
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

    const bool cache_prepare_resources_changed = async_cache_prepare && cachePrepareResourcesChanged();
    if (cache_prepare_resources_changed) {
        clearCachePrepareBlocked();
        waiting_group_yields_cache_prepare_ = false;
        cond_.notify_all();
    }

    if (async_cache_prepare) {
        const auto* prepare_waiting_front_after = waiting_streams_.empty() ? nullptr : waiting_streams_.front().get();
        const auto* prepare_group_front_after   = waiting_group_queue_.empty() || waiting_group_queue_.front().empty() ?
                                                      nullptr :
                                                      waiting_group_queue_.front().front().get();
        const auto* prepare_loading_group_front_after =
            loading_cache_group_queue_.empty() || loading_cache_group_queue_.front().empty() ?
                nullptr :
                loading_cache_group_queue_.front().front().get();
        const auto* prepare_running_front_after = running_streams_.empty() ? nullptr : running_streams_.front().get();
        const bool  prepare_state_changed =
            prepare_waiting_size_before != waiting_streams_.size()
            || prepare_group_queue_size_before != waiting_group_queue_.size()
            || prepare_group_front_size_before
                   != (waiting_group_queue_.empty() ? 0 : waiting_group_queue_.front().size())
            || prepare_loading_size_before != loading_cache_streams_.size()
            || prepare_loading_group_queue_size_before != loading_cache_group_queue_.size()
            || prepare_loading_group_front_size_before
                   != (loading_cache_group_queue_.empty() ? 0 : loading_cache_group_queue_.front().size())
            || prepare_running_size_before != running_streams_.size()
            || prepare_waiting_front_before != prepare_waiting_front_after
            || prepare_group_front_before != prepare_group_front_after
            || prepare_loading_group_front_before != prepare_loading_group_front_after
            || prepare_running_front_before != prepare_running_front_after
            || prepare_active_lane_before != active_admission_lane_ || prepare_prefer_group_before != prefer_group_next_
            || prepare_group_yield_before != waiting_group_yields_cache_prepare_;
        if (prepare_state_changed || finalized_waiting || cache_prepare_resources_changed) {
            // Admission can expose a new queue head or change the preferred
            // prepare lane without a new enqueue. Avoid waking the worker on
            // steady-state decode steps where none of that state changed.
            cond_.notify_all();
        }
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

int64_t FIFOScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + groupQueueStreamsSize(waiting_group_queue_);
}

int64_t FIFOScheduler::runningStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return running_streams_.size();
}

int64_t FIFOScheduler::onflightStreams() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size()
           + groupQueueStreamsSize(waiting_group_queue_) + groupQueueStreamsSize(loading_cache_group_queue_);
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

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::runningTaskList() {
    std::lock_guard<mutex> lock(lock_);
    running_task_list_.clear();
    running_task_list_.reserve(running_streams_.size());
    for (const auto& stream : running_streams_) {
        EngineScheduleInfo::TaskInfo task_info;
        task_info.request_id    = stream->streamId();
        task_info.prefix_length = stream->prefixLength();
        task_info.input_length  = stream->inputLength();
        task_info.batch_id      = stream->groupId();
        running_task_list_.push_back(task_info);
    }
    return running_task_list_;
}

void FIFOScheduler::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size    = waiting_streams_.size() + groupQueueStreamsSize(waiting_group_queue_);
        collector.running_stream_size = running_streams_.size();
        collector.loading_cache_stream_size =
            loading_cache_streams_.size() + groupQueueStreamsSize(loading_cache_group_queue_);
        collector.admitted_context_batch_size = last_admitted_context_batch_size_;
        collector.admitted_context_token_size = last_admitted_context_token_size_;
        collector.waiting_oldest_age_us       = last_waiting_oldest_age_us_;
        collector.group_fallback_count        = pending_group_fallback_count_.exchange(0, std::memory_order_relaxed);
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
    return;
}

}  // namespace rtp_llm
