#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"

#include <chrono>
#include <mutex>
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
    if (computeChunkGrant(prefill_chunk_size_, rows, seq_length, block_size) > 0) {
        return absl::OkStatus();
    }

    std::ostringstream error_msg;
    error_msg << "[chunked_prefill] request cannot make progress within the global prefill budget: budget="
              << prefill_chunk_size_ << ", seq_length=" << seq_length << ", rows=" << rows
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
        if (checkInputLength(stream) && checkChunkedPrefillRequest(stream).ok()) {
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
    list<GenerateStreamPtr> new_streams;

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

        if (!new_streams.empty()) {
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

        if (!stream->hasError() && !stream->hasEvent(StreamEvents::CanRun)
            && evaluateRunningMemory(new_streams, stream)) {
            stream->reportEvent(StreamEvents::CanRun);
            new_streams.push_back(stream);
            if (new_streams.size() == 1 && force_batch && stream->batchGroupId() != -1) {
                force_batch_group_id = stream->batchGroupId();
            }
        }
        it++;
    }
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
        collector.wait_stream_size          = waiting_streams_.size();
        collector.running_stream_size       = running_streams_.size();
        collector.loading_cache_stream_size = loading_cache_streams_.size();
        fillExtraMetrics(collector);
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
