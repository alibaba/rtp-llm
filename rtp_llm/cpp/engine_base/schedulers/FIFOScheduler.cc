#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <algorithm>
#include <chrono>
#include <memory>
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
    }
    cond_.notify_all();
    return absl::OkStatus();
}

int64_t FIFOScheduler::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

// 在入队前校验输入长度，避免无效请求进入等待队列
// 检查输入长度、投机解码预留空间和 batch token 上限。
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

std::vector<std::shared_ptr<GenerateStream>> FIFOScheduler::batchEnqueue(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<std::shared_ptr<GenerateStream>> stream_enqueued;
    for (auto it = streams.begin(); it != streams.end(); ++it) {
        if (checkInputLength((*it))) {
            stream_enqueued.emplace_back((*it));
        }
    }
    {
        std::lock_guard<std::mutex> lock(lock_);
        const auto                  enqueue_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        for (auto& stream : stream_enqueued) {
            stream->recordSchedulerEnqueueTime(enqueue_time_us);
        }
        waiting_streams_.insert(waiting_streams_.end(), stream_enqueued.begin(), stream_enqueued.end());
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return stream_enqueued;
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

    if (admitted_running_stream_count == 0
        && new_stream->contextLength() + running_streams_.size() < int(max_seq_len_)) {
        return true;
    }
    return schedule_runtime.admitted_prefill_token_size_with_cache + prefillTokenCostWithCache(new_stream)
               + running_streams_.size()
           < max_batch_tokens_size_;
}

size_t FIFOScheduler::prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const {
    // Match the token count that CP presents to the model after per-sequence padding.
    auto token_count = static_cast<size_t>(std::max(stream->contextLength(), 0));
    if (prefill_cp_size_ > 1) {
        token_count = makeZigzagTokenLayout(token_count, prefill_cp_size_).padded_token_count;
    }
    return token_count * static_cast<size_t>(stream->currentBatchSize());
}

size_t FIFOScheduler::prefillTokenCostWithCache(const GenerateStreamPtr& stream) const {
    // max_batch_tokens_size bounds the full logical sequence. Unlike the compute-only
    // quota, this count includes the reused prefix and therefore does not apply CP padding.
    const auto input_length  = static_cast<size_t>(std::max(stream->contextLength(), 0));
    const auto prefix_length = static_cast<size_t>(std::max(stream->prefixLength(), 0));
    return (input_length + prefix_length) * static_cast<size_t>(stream->currentBatchSize());
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
    return count_inited(waiting_streams_) + count_inited(loading_cache_streams_) + count_inited(running_streams_);
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

// 通过 GenerateStateMachine 驱动每个 stream 的状态转移，状态变化的 stream 移入对应队列
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

void FIFOScheduler::evaluateWaitingStreams(list<GenerateStreamPtr>& waiting_streams) {
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

    // Batch group scheduling support:
    // 1. Group completeness: force_batch streams with same batch_group_id are scheduled together
    //    only when group size reaches batch_group_size
    // 2. Timeout fallback: if batch_group_timeout expires, incomplete group is scheduled as normal
    // 3. Batch isolation: each scheduling round handles only one type:
    //    - normal streams, OR
    //    - streams from a single force_batch group

    struct GroupInfo {
        int64_t first_arrival_time = 0;
        int     count              = 0;
    };
    std::unordered_map<int64_t, GroupInfo> request_group_info;

    int64_t now = autil::TimeUtility::currentTimeInMilliSeconds();

    // Build group info statistics for force_batch streams
    for (const auto& stream : waiting_streams) {
        if (stream->forceBatch() && stream->batchGroupId() != -1) {
            auto& info = request_group_info[stream->batchGroupId()];
            if (info.count == 0) {
                info.first_arrival_time = stream->enqueueTime() / 1000;
            }
            info.count++;
        }
    }

    ScheduleRuntime schedule_runtime;

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto  current     = it++;
        auto& stream      = *current;
        bool  force_batch = stream->forceBatch();

        if (stream->hasError()) {
            auto state     = stream->getStatus();
            auto new_state = stream->moveToNext();
            if (new_state != state) {
                addStreamToNewState(stream, new_state);
                waiting_streams.erase(current);
            }
            continue;
        }

        // A stream that crosses the soft quota is already admitted. Keep scanning so
        // errored streams behind it can be finalized, but skip further admission.
        if (max_batch_tokens_without_cache_ > 0
            && schedule_runtime.admitted_prefill_token_size_without_cache >= max_batch_tokens_without_cache_) {
            continue;
        }

        // Check if this stream can be scheduled based on batch group rules
        if (force_batch && stream->batchGroupId() != -1) {
            const auto group_id                 = stream->batchGroupId();
            const bool group_partially_admitted = partially_admitted_force_batch_group_ids_.find(group_id)
                                                  != partially_admitted_force_batch_group_ids_.end();
            auto& info = request_group_info[group_id];
            // A partially admitted group remains complete even though its residual
            // member count is smaller than the original batch_group_size.
            if (!group_partially_admitted) {
                if (now - info.first_arrival_time > stream->batchGroupTimeout()) {
                    force_batch = false;
                } else if (info.count < stream->batchGroupSize()) {
                    // Group incomplete, skip this stream.
                    continue;
                }
            }
        }

        // Batch isolation: force_batch streams and normal streams cannot mix in the same round.
        // The first stream that makes scheduling progress determines the batch type for this round.
        if (schedule_runtime.batch_type_selected) {
            if (schedule_runtime.force_batch_group_id != -1) {
                // Already in force_batch mode, only accept same group
                if (!force_batch || stream->batchGroupId() != schedule_runtime.force_batch_group_id) {
                    continue;
                }
            } else {
                // Already in normal mode, skip force_batch streams
                if (force_batch) {
                    continue;
                }
            }
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
            if (!schedule_runtime.batch_type_selected) {
                schedule_runtime.batch_type_selected = true;
            }
            if (schedule_runtime.force_batch_group_id == -1 && force_batch && stream->batchGroupId() != -1) {
                schedule_runtime.force_batch_group_id = stream->batchGroupId();
                partially_admitted_force_batch_group_ids_.insert(schedule_runtime.force_batch_group_id);
            }

            if (new_state == StreamState::RUNNING) {
                ++schedule_runtime.admitted_running_stream_count;
                schedule_runtime.admitted_prefill_token_size_with_cache += prefillTokenCostWithCache(stream);
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

    for (auto group_it = partially_admitted_force_batch_group_ids_.begin();
         group_it != partially_admitted_force_batch_group_ids_.end();) {
        const auto group_id = *group_it;
        const bool still_waiting =
            std::any_of(waiting_streams.begin(), waiting_streams.end(), [group_id](const auto& stream) {
                return stream->forceBatch() && stream->batchGroupId() == group_id;
            });
        if (still_waiting) {
            ++group_it;
        } else {
            group_it = partially_admitted_force_batch_group_ids_.erase(group_it);
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

    // WAITING -> RUNNING: can run
    // WAITING -> LOADING_CACHE: load cache ok
    //
    // WAITING streams are advanced only after FIFO admits them in this scheduling round.
    // This matters for PD decode: DecodeRpcServer may pre-set CanRun before enqueue to
    // allocate KV blocks, so a permanent CanRun bit alone must not bypass capacity checks.
    size_t prev_waiting_size = waiting_streams_.size();
    evaluateWaitingStreams(waiting_streams_);
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
        collector.wait_stream_size            = waiting_streams_.size();
        collector.running_stream_size         = running_streams_.size();
        collector.loading_cache_stream_size   = loading_cache_streams_.size();
        collector.admitted_context_batch_size = last_admitted_context_batch_size_;
        collector.admitted_context_token_size = last_admitted_context_token_size_;
        collector.waiting_oldest_age_us       = last_waiting_oldest_age_us_;
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
    return;
}

}  // namespace rtp_llm
