#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_set>
#include <sstream>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;
namespace rtp_llm {

int64_t computeChunkGrant(int64_t budget, int64_t rows, int64_t remaining, int64_t block_size) {
    if (budget <= 0 || rows <= 0 || remaining <= 0 || block_size <= 0) {
        return 0;
    }

    const int64_t max_len = budget / rows;
    return remaining <= max_len ? remaining : (max_len / block_size) * block_size;
}

FIFOSchedulerBase::FIFOSchedulerBase(const RuntimeConfig&                   runtime_config,
                                     const ModelConfig&                     model_config,
                                     const PDSepConfig&                     pd_sep_config,
                                     const ParallelismConfig&               parallelism_config,
                                     const ModelSpecificConfig&             model_specific_config,
                                     const std::shared_ptr<KVCacheManager>& cache_manager,
                                     const kmonitor::MetricsReporterPtr     metrics_reporter):
    pd_sep_config_(pd_sep_config),
    model_specific_config_(model_specific_config),
    cache_manager_(cache_manager),
    max_seq_len_(model_config.max_seq_len),
    max_batch_tokens_size_(runtime_config.fifo_scheduler_config.max_batch_tokens_size),
    max_generate_batch_size_(runtime_config.max_generate_batch_size),
    max_inited_kv_cache_streams_(
        std::max<int64_t>(runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams, 0)),
    prefill_chunk_size_(runtime_config.fifo_scheduler_config.prefill_chunk_size),
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    metrics_reporter_(metrics_reporter) {}

bool FIFOSchedulerBase::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_streams_.empty() && loading_cache_streams_.empty() && running_streams_.empty()
           && extraOnflightStreams() == 0;
}

void FIFOSchedulerBase::cancelStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        stream->moveToNext();
    }
    streams.clear();
}

bool FIFOSchedulerBase::refreshAndReapTerminalStreams(std::list<GenerateStreamPtr>& streams) {
    bool reaped = false;
    for (auto it = streams.begin(); it != streams.end();) {
        auto& stream = *it;
        if (!stream->isFinished()) {
            stream->checkTimeout();
            if (!stream->hasError() && !stream->hasEvent(StreamEvents::GenerateDone)) {
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
        it     = streams.erase(it);
        reaped = true;
    }
    return reaped;
}

absl::Status FIFOSchedulerBase::stop() {
    RTP_LLM_LOG_INFO("stop %s", schedulerName());
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
        cancelStreams(waiting_streams_);
        cancelStreams(loading_cache_streams_);
        cancelStreams(running_streams_);
        cancelExtraStreams();
    }
    cond_.notify_all();
    return absl::OkStatus();
}

int64_t FIFOSchedulerBase::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

bool FIFOSchedulerBase::checkInputLength(const GenerateStreamPtr& stream) {
    const auto input_length = static_cast<size_t>(stream->inputLength());
    const auto reserve_step = stream->reserveStep();
    if (reserve_step > 0 && !(input_length <= max_seq_len_ && reserve_step <= max_seq_len_ - input_length)) {
        const auto allowed_input_length = reserve_step <= max_seq_len_ ? max_seq_len_ - reserve_step : 0;
        auto       error_info           = autil::StringUtil::formatString(
            "input len %zu with speculative reserve_step %zu exceeds max seq len %zu, "
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
        return false;
    }
    return true;
}

absl::Status FIFOSchedulerBase::checkChunkedPrefillRequest(const GenerateStreamPtr& stream) {
    if (!stream->chunkedPrefillEnabled()) {
        return absl::OkStatus();
    }

    const int64_t seq_length = stream->seqLength();
    const int64_t rows       = stream->currentBatchSize();
    const int64_t block_size = stream->seqSizePerBlock();
    if (seq_length > 0 && rows > 0 && block_size > 0) {
        return absl::OkStatus();
    }

    std::ostringstream error_msg;
    error_msg << "[chunked_prefill] invalid request shape: seq_length=" << seq_length << ", rows=" << rows
              << ", block_size=" << block_size;
    stream->reportError(ErrorCode::ERROR_GENERATE_CONFIG_FORMAT, error_msg.str());
    return absl::InvalidArgumentError(error_msg.str());
}

absl::Status FIFOSchedulerBase::enqueue(const GenerateStreamPtr& stream) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!checkInputLength(stream)) {
        return absl::InvalidArgumentError("Check input length failed");
    }
    auto chunk_status = checkChunkedPrefillRequest(stream);
    if (!chunk_status.ok()) {
        return chunk_status;
    }
    stream->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds());
    {
        std::lock_guard<std::mutex> lock(lock_);
        waiting_streams_.emplace_back(stream);
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return absl::OkStatus();
}

std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
FIFOSchedulerBase::enqueueGroup(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
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
    if (!valid_streams.empty()) {
        const auto enqueue_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        for (auto& stream : valid_streams) {
            stream->recordSchedulerEnqueueTime(enqueue_time_us);
        }
        std::lock_guard<std::mutex> lock(lock_);
        waiting_streams_.insert(waiting_streams_.end(), valid_streams.begin(), valid_streams.end());
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return {std::move(enqueue_successes), streams};
}

std::list<GenerateStreamPtr> FIFOSchedulerBase::selectPrefillPrefix(std::list<GenerateStreamPtr>& active_streams) {
    if (prefill_chunk_size_ <= 0) {
        return active_streams;
    }

    int64_t                      budget_left = prefill_chunk_size_;
    std::list<GenerateStreamPtr> selected;

    const auto finish_invalid_stream = [&active_streams](auto it, const std::string& reason) {
        const auto error_msg = "[chunked_prefill] scheduler rejects stream[" + std::to_string((*it)->streamId())
                               + "]: " + reason;
        (*it)->reportError(ErrorCode::UNKNOWN_ERROR, error_msg);
        (*it)->moveToNext();
        return active_streams.erase(it);
    };

    for (auto it = active_streams.begin(); it != active_streams.end();) {
        const auto& stream = *it;
        stream->checkTimeout();

        if (stream->hasError() || stream->hasEvent(StreamEvents::GenerateDone)) {
            stream->moveToNext();
            it = active_streams.erase(it);
            continue;
        }

        const int64_t rows       = stream->currentBatchSize();
        const int64_t reuse      = stream->reuseLength();
        const int64_t remaining  = static_cast<int64_t>(stream->seqLength()) - reuse;
        const int64_t block_size = stream->seqSizePerBlock();

        if (remaining <= 0 || block_size <= 0 || reuse < 0 || reuse % block_size != 0) {
            it = finish_invalid_stream(it,
                                       "invalid chunk window (rows=" + std::to_string(rows)
                                           + ", reuse=" + std::to_string(reuse) + ", remaining=" + std::to_string(remaining)
                                           + ", block_size=" + std::to_string(block_size) + ")");
            continue;
        }

        const int64_t grant = computeChunkGrant(budget_left, rows, remaining, block_size);

        if (grant <= 0) {
            if (!selected.empty()) {
                break;
            }
            it = finish_invalid_stream(it, "full prefill budget cannot produce a positive aligned grant");
            continue;
        }

        stream->setChunkSize(static_cast<int>(grant));
        selected.push_back(stream);
        budget_left -= grant * rows;
        ++it;

        if (grant < remaining) {
            break;
        }
    }

    return selected;
}

size_t FIFOSchedulerBase::evaluateAndUpdateStreams(list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
    size_t moved_count = 0;
    for (auto it = streams.begin(); it != streams.end();) {
        auto state     = (*it)->getStatus();
        auto new_state = (*it)->moveToNext();
        if (new_state != state) {
            addStreamToNewState(*it, new_state);
            it = streams.erase(it);
            ++moved_count;
        } else {
            it++;
        }
    }
    return moved_count;
}

void FIFOSchedulerBase::evaluateWaitingStreams(list<GenerateStreamPtr>& waiting_streams) {
    RTP_LLM_PROFILE_FUNCTION();
    list<GenerateStreamPtr>             admitted_streams;
    std::unordered_set<GenerateStream*> admitted_stream_ptrs;
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
    size_t       admitted_new_init_streams = 0;

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto& stream = *it;

        const bool already_inited_kv = stream->curBlocksNum() > 0;
        if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv
            && inited_kv_streams + admitted_new_init_streams >= max_inited_kv_cache_streams_) {
            ++it;
            continue;
        }

        // PD decode streams may already carry CanRun before FIFO admission.
        // Capacity checks must still run for the current scheduling round.
        if (!stream->hasError() && evaluateRunningMemory(admitted_streams, stream)) {
            if (!stream->hasEvent(StreamEvents::CanRun)) {
                stream->reportEvent(StreamEvents::CanRun);
            }
            admitted_streams.push_back(stream);
            admitted_stream_ptrs.insert(stream.get());
            if (max_inited_kv_cache_streams_ > 0 && !already_inited_kv) {
                ++admitted_new_init_streams;
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
            ++it;
            continue;
        }
        const auto state     = stream->getStatus();
        const auto new_state = stream->moveToNext();
        if (new_state != state) {
            addStreamToNewState(stream, new_state);
            it = waiting_streams.erase(it);
        } else {
            ++it;
        }
    }
}

size_t FIFOSchedulerBase::countInitedKVCacheStreams() const {
    const auto count_inited = [](const list<GenerateStreamPtr>& streams) {
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

void FIFOSchedulerBase::addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
    switch (new_state) {
        case StreamState::WAITING:
            waiting_streams_.push_back(stream);
            break;
        case StreamState::LOADING_CACHE:
            loading_cache_streams_.push_back(stream);
            break;
        case StreamState::RUNNING:
            onRunningStream(stream);
            new_streams_.push_back(stream);
            break;
        case StreamState::FINISHED:
            break;
        default:
            RTP_LLM_LOG_ERROR("Unknown state: %d for stream [%ld]", static_cast<int>(new_state), stream->streamId());
            break;
    }
}

int64_t FIFOSchedulerBase::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size();
}

int64_t FIFOSchedulerBase::runningStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return running_streams_.size();
}

int64_t FIFOSchedulerBase::onflightStreams() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size() + extraOnflightStreams();
}

void FIFOSchedulerBase::appendTaskInfos(std::vector<EngineScheduleInfo::TaskInfo>& task_list,
                                       const std::list<GenerateStreamPtr>&         streams) const {
    for (const auto& stream : streams) {
        EngineScheduleInfo::TaskInfo task_info{};
        task_info.request_id    = stream->streamId();
        task_info.prefix_length = stream->initialReuseLength();
        task_info.input_length  = stream->inputLength();
        task_info.iterate_count = stream->iterCount();
        task_info.batch_id      = stream->groupId();
        task_list.emplace_back(task_info);
    }
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOSchedulerBase::waitingTaskList() {
    std::lock_guard<mutex> lock(lock_);
    waiting_task_list_.clear();
    waiting_task_list_.reserve(waiting_streams_.size());
    appendTaskInfos(waiting_task_list_, waiting_streams_);
    return waiting_task_list_;
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOSchedulerBase::runningTaskList() {
    std::lock_guard<mutex> lock(lock_);
    running_task_list_.clear();
    running_task_list_.reserve(running_streams_.size());
    appendTaskInfos(running_task_list_, running_streams_);
    appendExtraRunningTaskList(running_task_list_);
    return running_task_list_;
}

void FIFOSchedulerBase::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size            = waiting_streams_.size();
        collector.running_stream_size         = running_streams_.size();
        collector.loading_cache_stream_size   = loading_cache_streams_.size();
        collector.admitted_context_batch_size = last_admitted_context_batch_size_;
        collector.admitted_context_token_size = last_admitted_context_token_size_;
        collector.waiting_oldest_age_us       = last_waiting_oldest_age_us_;
        fillExtraMetrics(collector);
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
