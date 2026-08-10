#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

using namespace std;
namespace rtp_llm {
namespace {
constexpr int64_t kMaxContextBatchCoalescingWindowMs = 60'000;
constexpr auto    kContextBatchCoalescingCheckInterval = std::chrono::milliseconds(10);

struct ForceBatchGroupInfo {
    int64_t first_arrival_time = 0;
    int     count              = 0;
};

using ForceBatchGroupInfoMap = std::unordered_map<int64_t, ForceBatchGroupInfo>;

template<typename IncludeStream>
ForceBatchGroupInfoMap collectForceBatchGroupInfo(const std::list<GenerateStreamPtr>& streams,
                                                  IncludeStream&&                    include_stream) {
    ForceBatchGroupInfoMap group_info;
    for (const auto& stream : streams) {
        if (!include_stream(stream) || !stream->forceBatch() || stream->batchGroupId() == -1) {
            continue;
        }
        auto& info = group_info[stream->batchGroupId()];
        if (info.count == 0) {
            info.first_arrival_time = stream->enqueueTime() / 1000;
        }
        ++info.count;
    }
    return group_info;
}

ForceBatchMode forceBatchMode(const GenerateStreamPtr&       stream,
                              const ForceBatchGroupInfoMap& group_info,
                              int64_t                       now_ms) {
    if (!stream->forceBatch()) {
        return ForceBatchMode::NORMAL;
    }
    if (stream->batchGroupId() == -1) {
        return ForceBatchMode::ATOMIC;
    }
    const auto info_it = group_info.find(stream->batchGroupId());
    if (info_it == group_info.end()) {
        return ForceBatchMode::WAITING;
    }
    const auto& info = info_it->second;
    if (info.count >= stream->batchGroupSize()) {
        return ForceBatchMode::ATOMIC;
    }
    if (now_ms - info.first_arrival_time > stream->batchGroupTimeout()) {
        return ForceBatchMode::NORMAL;
    }
    return ForceBatchMode::WAITING;
}
}

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
    max_context_batch_size_(std::max<int64_t>(1, runtime_config.fifo_scheduler_config.max_context_batch_size)),
    context_batch_coalescing_window_ms_(runtime_config.fifo_scheduler_config.context_batch_coalescing_window_ms),
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    metrics_reporter_(metrics_reporter) {
    if (context_batch_coalescing_window_ms_ < 0
        || context_batch_coalescing_window_ms_ > kMaxContextBatchCoalescingWindowMs) {
        throw std::invalid_argument("context_batch_coalescing_window_ms must be in [0, "
                                    + std::to_string(kMaxContextBatchCoalescingWindowMs) + "]");
    }
    RTP_LLM_LOG_INFO(
        "max_generate_batch_size is [%zu], max_context_batch_size is [%zu], max_batch_tokens_size is [%zu], "
        "context_batch_coalescing_window_ms is [%ld]",
        max_generate_batch_size_,
        max_context_batch_size_,
        max_batch_tokens_size_,
        context_batch_coalescing_window_ms_);
}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory FIFOScheduler");
}

bool FIFOScheduler::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_streams_.empty() && loading_cache_streams_.empty() && running_streams_.empty();
}

void FIFOScheduler::cancelStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        stream->moveToNext();  // Stream should be finished after moveToNext
    }
    streams.clear();
}

absl::Status FIFOScheduler::stop() {
    RTP_LLM_LOG_INFO("stop FIFOScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
        cancelStreams(waiting_streams_);
        cancelStreams(loading_cache_streams_);
        cancelStreams(running_streams_);
        context_load_cohort_.clear();
        context_load_cohort_deadline_.reset();
        force_batch_admission_by_member_.clear();
        active_force_batch_group_ids_.clear();
    }
    cond_.notify_all();
    return absl::OkStatus();
}

int64_t FIFOScheduler::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

// 在入队前校验输入长度，避免无效请求进入等待队列
// 仅检查输入长度不超过 KV Cache 最大可用 token 数；max_batch_tokens_size 的约束在调度时由
// evaluateRunningMemory 基于 contextLength 判断，不应在 enqueue 阶段乘以 batch_size 拒绝请求。
bool FIFOScheduler::checkInputLength(const GenerateStreamPtr& stream) {
    if (stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
        stream->reportError(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                            autil::StringUtil::formatString("input len " + std::to_string(stream->inputLength())
                                                            + " is greater than kv cache max available tokens num "
                                                            + std::to_string(cache_manager_->maxAvailableTokensNum())));
        return false;  // Input length exceeds max available tokens
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
        waiting_streams_.emplace_back(stream);
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return absl::OkStatus();
}

std::vector<std::shared_ptr<GenerateStream>> FIFOScheduler::batchEnqueue(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
    // Preserve 1:1 correspondence with the caller's input vector: failing streams are still
    // returned (already marked errored by checkInputLength via reportError) but only valid ones
    // enter the waiting queue.
    std::vector<std::shared_ptr<GenerateStream>> stream_enqueued;
    stream_enqueued.reserve(streams.size());
    for (const auto& stream : streams) {
        if (checkInputLength(stream)) {
            stream_enqueued.emplace_back(stream);
        }
    }
    {
        std::lock_guard<std::mutex> lock(lock_);
        waiting_streams_.insert(waiting_streams_.end(), stream_enqueued.begin(), stream_enqueued.end());
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return streams;
}

bool FIFOScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                          const GenerateStreamPtr&       new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        if (running_streams_.size() + streams.size() + 1 < max_generate_batch_size_) {
            return true;
        }
    }
    // prefill and decode not mixed together
    if (!running_streams_.empty()) {
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

void FIFOScheduler::accountBatchMetrics(const GenerateStreamPtr& new_stream) {
    for (auto& stream : running_streams_) {
        stream->incBatchWithPrefillTimes(1);
        stream->incBatchWithPrefillLen(new_stream->currentExecuteTokenSize());
    }
}

bool FIFOScheduler::waitPredicate() {
    // Check streams directly without calling empty() which acquires lock_ (already held by schedule())
    return stop_ || schedule_trigger_ || !waiting_streams_.empty() || !loading_cache_streams_.empty()
           || !running_streams_.empty();
}

size_t FIFOScheduler::coalescibleContextBatchSize() const {
    const auto group_info = collectForceBatchGroupInfo(waiting_streams_, [this](const auto& stream) {
        return force_batch_admission_by_member_.count(stream.get()) == 0;
    });
    const auto now_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
    size_t batch_size = 0;
    for (const auto& stream : waiting_streams_) {
        if (!stream->hasEvent(StreamEvents::Error) && !stream->hasEvent(StreamEvents::CanRun)
            && stream->isContextStream()
            && effectiveForceBatchMode(stream, forceBatchMode(stream, group_info, now_ms), now_ms)
                   == ForceBatchMode::NORMAL) {
            const size_t stream_batch_size = static_cast<size_t>(std::max(0, stream->currentBatchSize()));
            if (batch_size >= max_context_batch_size_
                || stream_batch_size >= max_context_batch_size_ - batch_size) {
                return max_context_batch_size_;
            }
            batch_size += stream_batch_size;
        }
    }
    return batch_size;
}

bool FIFOScheduler::shouldCoalesceContextBatch() const {
    if (pd_sep_config_.role_type != RoleType::PDFUSION || context_batch_coalescing_window_ms_ <= 0
        || max_context_batch_size_ <= 1 || !running_streams_.empty() || !loading_cache_streams_.empty()
        || !new_streams_.empty()) {
        return false;
    }
    if (std::any_of(waiting_streams_.begin(), waiting_streams_.end(), [](const auto& stream) {
            return stream->hasEvent(StreamEvents::Error);
        })) {
        return false;
    }
    const auto first_runnable = std::find_if(waiting_streams_.begin(), waiting_streams_.end(), [](const auto& stream) {
        return !stream->hasEvent(StreamEvents::Error) && !stream->hasEvent(StreamEvents::CanRun);
    });
    if (first_runnable == waiting_streams_.end() || !(*first_runnable)->isContextStream()) {
        return false;
    }
    const auto group_info = collectForceBatchGroupInfo(waiting_streams_, [this](const auto& stream) {
        return force_batch_admission_by_member_.count(stream.get()) == 0;
    });
    const auto now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    if (effectiveForceBatchMode(*first_runnable, forceBatchMode(*first_runnable, group_info, now_ms), now_ms)
        != ForceBatchMode::NORMAL) {
        return false;
    }
    return coalescibleContextBatchSize() < max_context_batch_size_;
}

void FIFOScheduler::waitForContextBatch(std::unique_lock<std::mutex>& lock) {
    if (!shouldCoalesceContextBatch()) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(context_batch_coalescing_window_ms_);
    onContextBatchCoalescingWait();
    while (!stop_.load(std::memory_order_acquire) && shouldCoalesceContextBatch()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            break;
        }
        cond_.wait_until(lock, std::min(deadline, now + kContextBatchCoalescingCheckInterval));
    }
}

bool FIFOScheduler::contextLoadCohortEnabled() const {
    return pd_sep_config_.role_type == RoleType::PDFUSION && context_batch_coalescing_window_ms_ > 0
           && max_context_batch_size_ > 1;
}

void FIFOScheduler::removeFromContextLoadCohort(const GenerateStreamPtr& stream) {
    if (context_load_cohort_.erase(stream.get()) > 0 && context_load_cohort_.empty()) {
        context_load_cohort_deadline_.reset();
    }
}

ForceBatchMode FIFOScheduler::effectiveForceBatchMode(const GenerateStreamPtr& stream,
                                                      ForceBatchMode             base_mode,
                                                      int64_t                    now_ms) const {
    const auto member_it = force_batch_admission_by_member_.find(stream.get());
    if (member_it != force_batch_admission_by_member_.end()) {
        const auto& identity = member_it->second;
        if (!identity->broken
            && std::any_of(identity->members.begin(), identity->members.end(), [](const auto* member) {
                   return member->hasEvent(StreamEvents::Error) || member->hasEvent(StreamEvents::GenerateDone);
               })) {
            identity->broken = true;
        }
        bool                all_waiting   = !identity->members.empty();
        bool                has_loading   = false;
        bool                phase_mismatch = false;
        std::optional<bool> context_phase;
        for (const auto* member : identity->members) {
            const auto state = member->getStatus();
            if (state == StreamState::LOADING_CACHE) {
                has_loading = true;
                all_waiting = false;
                continue;
            }
            if (state != StreamState::WAITING) {
                all_waiting = false;
                continue;
            }
            if (!context_phase.has_value()) {
                context_phase = member->isContextStream();
            } else if (context_phase.value() != member->isContextStream()) {
                phase_mismatch = true;
            }
        }
        if (identity->broken) {
            return now_ms > identity->fallback_deadline_ms ? ForceBatchMode::NORMAL : ForceBatchMode::WAITING;
        }
        if (phase_mismatch) {
            return ForceBatchMode::WAITING;
        }
        if (has_loading) {
            return ForceBatchMode::WAITING;
        }
        return all_waiting ? ForceBatchMode::ATOMIC : ForceBatchMode::WAITING;
    }

    if (stream->forceBatch() && stream->batchGroupId() != -1
        && active_force_batch_group_ids_.count(stream->batchGroupId()) > 0) {
        return ForceBatchMode::WAITING;
    }
    return base_mode;
}

void FIFOScheduler::trackCompleteForceBatchAdmissions(const std::list<GenerateStreamPtr>& admitted_streams,
                                                      int64_t                             now_ms) {
    const auto group_info = collectForceBatchGroupInfo(waiting_streams_, [this](const auto& stream) {
        return force_batch_admission_by_member_.count(stream.get()) == 0;
    });
    std::unordered_map<int64_t, std::vector<GenerateStreamPtr>> admitted_groups;
    for (const auto& stream : admitted_streams) {
        if (!stream->hasEvent(StreamEvents::CanRun) || !stream->forceBatch() || stream->batchGroupId() == -1
            || force_batch_admission_by_member_.count(stream.get()) > 0) {
            continue;
        }
        const auto base_mode = forceBatchMode(stream, group_info, now_ms);
        if (effectiveForceBatchMode(stream, base_mode, now_ms) == ForceBatchMode::ATOMIC) {
            admitted_groups[stream->batchGroupId()].push_back(stream);
        }
    }

    for (auto& [group_id, members] : admitted_groups) {
        if (members.empty() || active_force_batch_group_ids_.count(group_id) > 0) {
            continue;
        }
        const int expected_size = members.front()->batchGroupSize();
        if (expected_size <= 0 || members.size() != static_cast<size_t>(expected_size)
            || std::any_of(members.begin(), members.end(), [expected_size](const auto& member) {
                   return member->batchGroupSize() != expected_size;
               })) {
            continue;
        }

        const auto oldest = *std::min_element(members.begin(), members.end(), [](const auto& lhs, const auto& rhs) {
            return lhs->enqueueTime() < rhs->enqueueTime();
        });
        auto identity = std::make_shared<ForceBatchAdmissionIdentity>();
        identity->group_id             = group_id;
        identity->expected_size        = expected_size;
        identity->fallback_deadline_ms = oldest->enqueueTime() / 1000 + oldest->batchGroupTimeout();
        for (const auto& member : members) {
            identity->members.insert(member.get());
            force_batch_admission_by_member_[member.get()] = identity;
        }
        active_force_batch_group_ids_.insert(group_id);
    }
}

void FIFOScheduler::removeFromForceBatchAdmission(const GenerateStreamPtr& stream, bool broken_before_run) {
    const auto member_it = force_batch_admission_by_member_.find(stream.get());
    if (member_it == force_batch_admission_by_member_.end()) {
        return;
    }
    const auto identity = member_it->second;
    if (broken_before_run) {
        identity->broken = true;
    }
    identity->members.erase(stream.get());
    force_batch_admission_by_member_.erase(member_it);
    if (identity->members.empty()) {
        active_force_batch_group_ids_.erase(identity->group_id);
    }
}

bool FIFOScheduler::waitForContextLoadCohort(std::unique_lock<std::mutex>& lock) {
    if (!contextLoadCohortEnabled() || context_load_cohort_.empty() || !running_streams_.empty()
        || !new_streams_.empty()) {
        return false;
    }

    const auto inspect_cohort = [this] {
        size_t ready_batch_size = 0;
        bool   has_loading_peer = false;
        bool   has_waiting_error = false;
        for (const auto& stream : loading_cache_streams_) {
            if (context_load_cohort_.count(stream.get()) > 0 && !stream->hasEvent(StreamEvents::Error)) {
                has_loading_peer = true;
            }
        }
        for (const auto& stream : waiting_streams_) {
            if (stream->hasEvent(StreamEvents::Error)) {
                has_waiting_error = true;
            }
            if (context_load_cohort_.count(stream.get()) == 0 || stream->hasEvent(StreamEvents::Error)) {
                continue;
            }
            const size_t stream_batch_size = static_cast<size_t>(std::max(0, stream->currentBatchSize()));
            if (ready_batch_size >= max_context_batch_size_
                || stream_batch_size >= max_context_batch_size_ - ready_batch_size) {
                ready_batch_size = max_context_batch_size_;
                break;
            }
            ready_batch_size += stream_batch_size;
        }
        return std::make_tuple(ready_batch_size, has_loading_peer, has_waiting_error);
    };

    auto [ready_batch_size, has_loading_peer, has_waiting_error] = inspect_cohort();
    if (ready_batch_size == 0) {
        return false;
    }

    const auto group_info = collectForceBatchGroupInfo(waiting_streams_, [this](const auto& stream) {
        return force_batch_admission_by_member_.count(stream.get()) == 0;
    });
    const auto now_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
    for (const auto& stream : waiting_streams_) {
        if (stream->hasEvent(StreamEvents::Error) || stream->hasEvent(StreamEvents::CanRun)) {
            continue;
        }
        const auto mode = effectiveForceBatchMode(stream, forceBatchMode(stream, group_info, now_ms), now_ms);
        if (mode == ForceBatchMode::WAITING) {
            continue;
        }
        if (!stream->isContextStream()
            || (mode == ForceBatchMode::ATOMIC && context_load_cohort_.count(stream.get()) == 0)) {
            return false;
        }
        break;
    }

    if (has_waiting_error || !has_loading_peer || ready_batch_size >= max_context_batch_size_) {
        return true;
    }

    if (!context_load_cohort_deadline_.has_value()) {
        context_load_cohort_deadline_ = std::chrono::steady_clock::now()
                                        + std::chrono::milliseconds(context_batch_coalescing_window_ms_);
    }
    onContextBatchCoalescingWait();
    while (!stop_.load(std::memory_order_acquire) && has_loading_peer
           && ready_batch_size < max_context_batch_size_) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= *context_load_cohort_deadline_) {
            break;
        }
        cond_.wait_until(lock, std::min(*context_load_cohort_deadline_, now + kContextBatchCoalescingCheckInterval));
        evaluateAndUpdateStreams(loading_cache_streams_);
        std::tie(ready_batch_size, has_loading_peer, has_waiting_error) = inspect_cohort();
        if (has_waiting_error || ready_batch_size == 0) {
            break;
        }
    }
    return true;
}

// 通过 GenerateStateMachine 驱动每个 stream 的状态转移，状态变化的 stream 移入对应队列
void FIFOScheduler::evaluateAndUpdateStreams(list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<std::pair<GenerateStreamPtr, bool>> force_batch_removals;
    for (auto it = streams.begin(); it != streams.end();) {
        auto state     = (*it)->getStatus();
        auto new_state = (*it)->moveToNext();
        if (pd_sep_config_.role_type == RoleType::PDFUSION && state == StreamState::LOADING_CACHE
            && new_state == StreamState::WAITING) {
            (*it)->consumeCanRunAdmission();
        }
        if (contextLoadCohortEnabled()) {
            if ((*it)->isContextStream() && state == StreamState::WAITING && new_state == StreamState::LOADING_CACHE
                && (context_load_cohort_.empty() || !context_load_cohort_deadline_.has_value())) {
                context_load_cohort_.insert(it->get());
            }
            if ((state == StreamState::LOADING_CACHE && new_state == StreamState::WAITING
                 && !(*it)->isContextStream())
                || new_state == StreamState::RUNNING || new_state == StreamState::FINISHED) {
                removeFromContextLoadCohort(*it);
            }
        }
        if (new_state == StreamState::RUNNING || new_state == StreamState::FINISHED) {
            force_batch_removals.emplace_back(*it, new_state == StreamState::FINISHED && state != StreamState::RUNNING);
        }
        if (new_state != state) {
            addStreamToNewState(*it, new_state);
            it = streams.erase(it);
        } else {
            it++;
        }
    }
    for (const auto& [stream, broken_before_run] : force_batch_removals) {
        removeFromForceBatchAdmission(stream, broken_before_run);
    }
}

void FIFOScheduler::evaluateWaitingStreams(list<GenerateStreamPtr>& waiting_streams) {
    RTP_LLM_PROFILE_FUNCTION();
    list<GenerateStreamPtr> new_streams;

    // Batch group scheduling support:
    // 1. Group completeness: force_batch streams with same batch_group_id are scheduled together
    //    only when group size reaches batch_group_size
    // 2. Timeout fallback: if batch_group_timeout expires, incomplete group is scheduled as normal
    // 3. Batch isolation: each scheduling round handles only one type:
    //    - normal streams, OR
    //    - streams from a single force_batch group

    const auto request_group_info = collectForceBatchGroupInfo(waiting_streams, [this](const auto& stream) {
        return force_batch_admission_by_member_.count(stream.get()) == 0;
    });
    const auto now                = autil::TimeUtility::currentTimeInMilliSeconds();

    int64_t force_batch_group_id = -1;
    std::optional<bool> selected_context_phase;
    size_t              selected_context_batch_size = 0;
    const bool          enforce_pure_phase = pd_sep_config_.role_type == RoleType::PDFUSION;
    const bool limit_context_batch = pd_sep_config_.role_type == RoleType::PDFUSION
                                     && context_batch_coalescing_window_ms_ > 0
                                     && max_context_batch_size_ > 1;
    if (enforce_pure_phase) {
        // A scheduling round has one execution phase even when coalescing is disabled. The
        // executor cannot consume context and decode streams in the same model invocation.
        if (!running_streams_.empty()) {
            selected_context_phase = running_streams_.front()->isContextStream();
        } else if (admit_context_load_cohort_only_) {
            selected_context_phase = true;
        }
    }

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto& stream      = *it;
        const auto force_batch_mode =
            effectiveForceBatchMode(stream, forceBatchMode(stream, request_group_info, now), now);
        bool       force_batch      = force_batch_mode == ForceBatchMode::ATOMIC;

        if (admit_context_load_cohort_only_ && stream->isContextStream()
            && context_load_cohort_.count(stream.get()) == 0) {
            it++;
            continue;
        }

        if (enforce_pure_phase && selected_context_phase.has_value()
            && stream->isContextStream() != selected_context_phase.value()) {
            it++;
            continue;
        }
        const size_t stream_context_batch_size =
            static_cast<size_t>(std::max(0, stream->currentBatchSize()));

        // Check if this stream can be scheduled based on batch group rules
        if (force_batch_mode == ForceBatchMode::WAITING) {
            it++;
            continue;
        }
        // An expired force-batch group falls back to normal scheduling and must obey the
        // same context row cap. Complete force-batch groups retain their atomic semantics.
        if (limit_context_batch && stream->isContextStream() && !force_batch && !new_streams.empty()) {
            if (selected_context_batch_size >= max_context_batch_size_
                || stream_context_batch_size > max_context_batch_size_ - selected_context_batch_size) {
                it++;
                continue;
            }
        }

        // Batch isolation: force_batch streams and normal streams cannot mix in the same round.
        // The first stream that passes checks determines the batch type for this round.
        if (!new_streams.empty()) {
            if (force_batch_group_id != -1) {
                // Already in force_batch mode, only accept same group
                if (!force_batch || stream->batchGroupId() != force_batch_group_id) {
                    it++;
                    continue;
                }
            } else {
                // Already in normal mode, skip force_batch streams
                if (force_batch) {
                    it++;
                    continue;
                }
            }
        }

        // Check for errors and memory constraints
        if (!stream->hasEvent(StreamEvents::Error) && !stream->hasEvent(StreamEvents::CanRun)
            && evaluateRunningMemory(new_streams, stream)) {
            stream->reportEvent(StreamEvents::CanRun);
            new_streams.push_back(stream);
            if (!selected_context_phase.has_value()) {
                selected_context_phase = stream->isContextStream();
            }
            if (stream->isContextStream()) {
                if (selected_context_batch_size >= max_context_batch_size_
                    || stream_context_batch_size >= max_context_batch_size_ - selected_context_batch_size) {
                    selected_context_batch_size = max_context_batch_size_;
                } else {
                    selected_context_batch_size += stream_context_batch_size;
                }
            }

            // Lock batch type based on first scheduled stream
            if (new_streams.size() == 1 && force_batch && stream->batchGroupId() != -1) {
                force_batch_group_id = stream->batchGroupId();
            }
        }
        it++;
    }
    trackCompleteForceBatchAdmissions(new_streams, now);
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

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    schedule_trigger_ = false;

    // LOADING_CACHE -> DONE/WAITING: error / load cache done
    evaluateAndUpdateStreams(loading_cache_streams_);
    // RUNNING -> DONE: error / finished
    evaluateAndUpdateStreams(running_streams_);

    admit_context_load_cohort_only_ = waitForContextLoadCohort(lock);
    if (!admit_context_load_cohort_only_) {
        waitForContextBatch(lock);
    }
    // WAITING -> RUNNING: can run
    // WAITING -> LOADING_CACHE: load cache ok
    //
    // Two-phase state transition for WAITING streams:
    //   Phase 1 (evaluateWaitingStreams): Streams that pass memory check get CanRun event,
    //       but are NOT removed from waiting_streams_ yet. This is because evaluateWaitingStreams
    //       iterates over waiting_streams_ and removing elements during iteration would be unsafe.
    //   Phase 2 (evaluateAndUpdateStreams): Actually moves streams from waiting_streams_ to
    //       their new state (RUNNING or LOADING_CACHE) based on the events set in Phase 1.
    // This separation ensures safe iteration while deferring structural modifications.
    size_t prev_waiting_size = waiting_streams_.size();
    evaluateWaitingStreams(waiting_streams_);
    evaluateAndUpdateStreams(waiting_streams_);
    admit_context_load_cohort_only_ = false;
    running_streams_.insert(running_streams_.end(), new_streams_.begin(), new_streams_.end());
    new_streams_.clear();

    // If streams were scheduled, trigger next scheduling round
    if (waiting_streams_.size() < prev_waiting_size) {
        schedule_trigger_ = true;
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

int64_t FIFOScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size();
}

int64_t FIFOScheduler::runningStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return running_streams_.size();
}

int64_t FIFOScheduler::onflightStreams() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size();
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::waitingTaskList() {
    std::lock_guard<mutex> lock(lock_);
    waiting_task_list_.clear();
    waiting_task_list_.reserve(waiting_streams_.size());
    for (const auto& stream : waiting_streams_) {
        EngineScheduleInfo::TaskInfo task_info;
        task_info.request_id    = stream->streamId();
        task_info.prefix_length = stream->prefixLength();
        task_info.input_length  = stream->inputLength();
        waiting_task_list_.emplace_back(task_info);
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
        running_task_list_.emplace_back(task_info);
    }
    return running_task_list_;
}

void FIFOScheduler::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size          = waiting_streams_.size();
        collector.running_stream_size       = running_streams_.size();
        collector.loading_cache_stream_size = loading_cache_streams_.size();
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
    return;
}

}  // namespace rtp_llm
