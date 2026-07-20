#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;
namespace rtp_llm {

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
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    metrics_reporter_(metrics_reporter) {}

bool FIFOSchedulerBase::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_streams_.empty() && loading_cache_streams_.empty() && running_streams_.empty() && !hasExtraStreams();
}

void FIFOSchedulerBase::cancelStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        stream->moveToNext();
    }
    streams.clear();
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

absl::Status FIFOSchedulerBase::enqueue(const GenerateStreamPtr& stream) {
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

std::vector<std::shared_ptr<GenerateStream>> FIFOSchedulerBase::batchEnqueue(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
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
    return stream_enqueued;
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
    const size_t inited_kv_streams = max_inited_kv_cache_streams_ > 0 ? countInitedKVCacheStreams() : 0;
    size_t       admitted_new_init_streams = 0;

    struct GroupInfo {
        int64_t first_arrival_time = 0;
        int     count              = 0;
    };
    std::unordered_map<int64_t, GroupInfo> request_group_info;

    int64_t now = autil::TimeUtility::currentTimeInMilliSeconds();
    for (const auto& stream : waiting_streams) {
        if (stream->forceBatch() && stream->batchGroupId() != -1) {
            auto& info = request_group_info[stream->batchGroupId()];
            if (info.count == 0) {
                info.first_arrival_time = stream->enqueueTime() / 1000;
            }
            info.count++;
        }
    }

    int64_t force_batch_group_id = -1;

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto& stream      = *it;
        bool  force_batch = stream->forceBatch();

        if (force_batch && stream->batchGroupId() != -1) {
            auto& info = request_group_info[stream->batchGroupId()];
            if (now - info.first_arrival_time > stream->batchGroupTimeout()) {
                force_batch = false;
            } else if (info.count < stream->batchGroupSize()) {
                it++;
                continue;
            }
        }

        if (!admitted_streams.empty()) {
            if (force_batch_group_id != -1) {
                if (!force_batch || stream->batchGroupId() != force_batch_group_id) {
                    it++;
                    continue;
                }
            } else if (force_batch) {
                it++;
                continue;
            }
        }

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
            if (admitted_streams.size() == 1 && force_batch && stream->batchGroupId() != -1) {
                force_batch_group_id = stream->batchGroupId();
            }
        }
        it++;
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

std::vector<EngineScheduleInfo::TaskInfo> FIFOSchedulerBase::waitingTaskList() {
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

std::vector<EngineScheduleInfo::TaskInfo> FIFOSchedulerBase::runningTaskList() {
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

void FIFOSchedulerBase::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size          = waiting_streams_.size();
        collector.running_stream_size       = running_streams_.size();
        collector.loading_cache_stream_size = loading_cache_streams_.size();
        fillExtraMetrics(collector);
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
