#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

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
    pd_sep_config_(pd_sep_config),
    model_specific_config_(model_specific_config),
    cache_manager_(cache_manager),
    max_seq_len_(model_config.max_seq_len),
    max_batch_tokens_size_(runtime_config.fifo_scheduler_config.max_batch_tokens_size),
    max_generate_batch_size_(runtime_config.max_generate_batch_size),
    max_inited_kv_cache_streams_(
        std::max<int64_t>(runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams, 0)),
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    cp_force_single_prefill_(parallelism_config.prefill_cp_config.is_enabled()
                             && runtime_config.fifo_scheduler_config.cp_force_single_prefill),
    metrics_reporter_(metrics_reporter) {
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%zu], max_batch_tokens_size is [%zu], "
                     "cp_force_single_prefill is [%d], max_inited_kv_cache_streams is [%zu]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_,
                     cp_force_single_prefill_,
                     max_inited_kv_cache_streams_);
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

absl::Status FIFOScheduler::stop() {
    RTP_LLM_LOG_INFO("stop FIFOScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
        cancelStreams(waiting_streams_);
        cancelStreams(loading_cache_streams_);
        cancelStreams(running_streams_);
        cancelGroups(waiting_group_queue_);
        cancelGroups(loading_cache_group_queue_);
    }
    cond_.notify_all();
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
    } else if ((size_t)stream->inputLength() * stream->currentBatchSize() > max_batch_tokens_size_) {
        auto error_info =
            autil::StringUtil::formatString("input len [%d] * batch size [%d] > max_batch_tokens_size [%d]",
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
        RTP_LLM_LOG_WARNING("enqueue group exceeds scheduler limits; fallback to individual streams: "
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
    }
    cond_.notify_all();
    return {std::move(enqueue_successes), streams};
}

bool FIFOScheduler::evaluateRunningBatch(const list<GenerateStreamPtr>& streams,
                                         const GenerateStreamPtr&       new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        // Decode-only scheduling can top up an existing running decode batch.
        // max_generate_batch_size_ is an inclusive cap; only requests above it
        // should be rejected.
        if (running_streams_.size() + streams.size() + 1 <= max_generate_batch_size_) {
            return true;
        }
    }
    // prefill and decode not mixed together
    if (!running_streams_.empty()) {
        return false;
    }
    // Conservative CP prefill mode: cap at one stream per round unless runtime
    // config explicitly allows CP prefill batching.
    if (cp_force_single_prefill_ && !streams.empty()) {
        return false;
    }
    if (running_streams_.size() + streams.size() + 1 > max_generate_batch_size_) {
        return false;
    }

    int max_token_size = new_stream->contextLength();
    if (streams.empty() && max_token_size + running_streams_.size() < int(max_seq_len_)) {
        return true;
    }
    for (auto& stream : streams) {
        max_token_size = std::max(max_token_size, stream->contextLength());
    }
    // 这里的判断是要求当前调度轮所有请求参与计算的 token 数之和小于 max_batch_tokens_size_，loading_cache_streams
    // 这一轮实际不参与计算，不需要计入。
    return max_token_size * (streams.size() + 1) + running_streams_.size() < int(max_batch_tokens_size_);
}

size_t FIFOScheduler::countInitedKVCacheStreams() const {
    auto count_inited = [](const list<GenerateStreamPtr>& streams) {
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

bool FIFOScheduler::waitPredicate() {
    return stop_ || !waiting_streams_.empty() || !loading_cache_streams_.empty() || !running_streams_.empty()
           || !waiting_group_queue_.empty() || !loading_cache_group_queue_.empty();
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
    // A prepared front group forms the beginning of this round's batch. Ordinary
    // waiting streams may fill the remaining capacity, while legacy groups in
    // waiting_streams_ keep their existing isolation behavior.
    list<GenerateStreamPtr>             admitted_streams = already_admitted_streams;
    std::unordered_set<GenerateStream*> admitted_stream_ptrs;
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
    size_t inited_kv_streams         = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;
    size_t admitted_new_init_streams = 0;

    // Preserve the legacy grouping contract for streams submitted individually through enqueue().
    // Explicit enqueueGroup() requests use the dedicated group queues and do not carry group metadata here.

    struct GroupInfo {
        int64_t first_arrival_time = 0;
        int     count              = 0;
    };
    std::unordered_map<int64_t, GroupInfo> request_group_info;

    int64_t now = autil::TimeUtility::currentTimeInMilliSeconds();

    // Build group info statistics for group streams
    for (const auto& stream : waiting_streams) {
        if (stream->isGroup()) {
            auto& info = request_group_info[stream->groupId()];
            if (info.count == 0) {
                info.first_arrival_time = stream->enqueueTime() / 1000;
            }
            info.count++;
        }
    }

    int64_t group_id = -1;

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto& stream = *it;
        bool  group  = stream->isGroup();

        // Check if this stream can be scheduled based on group rules
        if (group) {
            auto& info = request_group_info[stream->groupId()];
            // Check timeout: if expired, treat as normal stream
            if (now - info.first_arrival_time > stream->groupTimeout()) {
                group = false;
            } else if (info.count < stream->groupSize()) {
                // Group incomplete, skip this stream
                it++;
                continue;
            }
        }

        // Batch isolation: group streams and normal streams cannot mix in the same round.
        // The first stream that passes checks determines the batch type for this round.
        if (!admitted_streams.empty()) {
            if (group_id != -1) {
                // Already in group mode, only accept same group
                if (!group || stream->groupId() != group_id) {
                    it++;
                    continue;
                }
            } else {
                // Already in normal mode, skip group streams
                if (group) {
                    it++;
                    continue;
                }
            }
        }

        // Check for errors and memory constraints
        //
        // Some PD decode streams already carry CanRun before entering FIFO: DecodeRpcServer uses
        // CanRun to drive the pre-enqueue KV allocation path. CanRun is a permanent event, so it
        // cannot be used as proof that FIFO has admitted this stream in the current scheduling
        // round. Always run FIFO capacity checks and only advance streams admitted here.
        const bool already_inited_kv = stream->curBlocksNum() > 0;
        if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv
            && inited_kv_streams + admitted_new_init_streams >= max_inited_kv_cache_streams_) {
            it++;
            continue;
        }

        if (!stream->hasError() && evaluateRunningBatch(admitted_streams, stream)) {
            if (!stream->hasEvent(StreamEvents::CanRun)) {
                stream->reportEvent(StreamEvents::CanRun);
            }
            admitted_streams.push_back(stream);
            admitted_stream_ptrs.insert(stream.get());
            if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv) {
                ++admitted_new_init_streams;
            }

            // Lock batch type based on first scheduled stream
            if (admitted_streams.size() == 1 && group) {
                group_id = stream->groupId();
            }
        }
        it++;
    }

    for (const auto& stream : admitted_streams) {
        if (stream->isContextStream()) {
            ++last_admitted_context_batch_size_;
            last_admitted_context_token_size_ += stream->contextLength();
        }
    }

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto& stream = *it;
        if (!stream->hasError() && admitted_stream_ptrs.find(stream.get()) == admitted_stream_ptrs.end()) {
            it++;
            continue;
        }
        auto state     = stream->getStatus();
        auto new_state = stream->moveToNext();
        if (new_state != state) {
            addStreamToNewState(stream, new_state);
            it = waiting_streams.erase(it);
        } else {
            it++;
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

void FIFOScheduler::dissolveGroup(StreamGroup& group) {
    for (auto it = group.begin(); it != group.end();) {
        const auto state = (*it)->getStatus();
        if (state == StreamState::RUNNING) {
            accountBatchMetrics(*it);
            new_streams_.splice(new_streams_.end(), group, it++);
        } else if (state == StreamState::LOADING_CACHE) {
            loading_cache_streams_.splice(loading_cache_streams_.end(), group, it++);
        } else if (state == StreamState::WAITING) {
            waiting_streams_.splice(waiting_streams_.end(), group, it++);
        } else {
            RTP_LLM_CHECK_WITH_INFO(state == StreamState::FINISHED, "unexpected stream state while dissolving group");
            it = group.erase(it);
        }
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

void FIFOScheduler::evaluateWaitingGroupQueue() {
    if (!running_streams_.empty() || !new_streams_.empty() || !waiting_streams_.empty()
        || !loading_cache_streams_.empty() || !loading_cache_group_queue_.empty() || waiting_group_queue_.empty()) {
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

    size_t      deferred_count = 0;
    StreamGroup admitted_streams;
    for (auto it = group.begin(); it != group.end();) {
        auto& stream = *it;
        if (!evaluateRunningBatch(admitted_streams, stream)) {
            ++deferred_count;
            ++it;
            continue;
        }

        if (!stream->hasEvent(StreamEvents::CanRun)) {
            stream->reportEvent(StreamEvents::CanRun);
        }
        const auto state = stream->moveToNext();
        if (state == StreamState::FINISHED) {
            it = group.erase(it);
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(state == StreamState::RUNNING || state == StreamState::LOADING_CACHE,
                                "group stream must be RUNNING or LOADING_CACHE after scheduler admission");
        admitted_streams.push_back(stream);
        ++it;
    }

    if (deferred_count > 0) {
        RTP_LLM_LOG_WARNING("group partially admitted and will be dissolved: original=%zu admitted=%zu deferred=%zu",
                            original_size,
                            admitted_streams.size(),
                            deferred_count);
        dissolveGroup(group);
        pending_group_fallback_count_.fetch_add(1, std::memory_order_relaxed);
        waiting_group_queue_.pop_front();
        return;
    }

    const bool all_running = std::all_of(
        group.begin(), group.end(), [](const auto& stream) { return stream->getStatus() == StreamState::RUNNING; });
    if (all_running) {
        moveGroupToNewStreams(group);
        waiting_group_queue_.pop_front();
    } else {
        moveGroupToAllocatingGroup(group);
        waiting_group_queue_.pop_front();
    }
}

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }
    last_admitted_context_batch_size_ = 0;
    last_admitted_context_token_size_ = 0;
    last_waiting_oldest_age_us_       = 0;

    evaluateAndUpdateStreams(loading_cache_streams_);
    evaluateAndUpdateStreams(running_streams_);

    evaluateLoadingCacheGroupQueue();
    evaluateWaitingGroupQueue();
    evaluateWaitingStreams(waiting_streams_, new_streams_);
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
