#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;
namespace rtp_llm {

namespace {
constexpr auto kNoProgressScheduleGap = std::chrono::milliseconds(1);

// Parse the decode_prefill_ratio string into the internal signed cadence step.
//   "N"   (N>=1) -> N    (1 prefill : N decode)
//   "1/X" (X>=1) -> -X   (X prefill : 1 decode)
//   invalid      -> 1    (alternation), with a warning
int64_t parseDecodePrefillRatio(const std::string& ratio) {
    try {
        auto slash = ratio.find('/');
        if (slash == std::string::npos) {
            size_t  pos = 0;
            int64_t n   = std::stoll(ratio, &pos);
            if (pos == ratio.size() && n >= 1) {
                return n;
            }
        } else if (ratio.substr(0, slash) == "1") {
            const std::string den = ratio.substr(slash + 1);
            size_t            pos = 0;
            int64_t           x   = std::stoll(den, &pos);
            if (pos == den.size() && x >= 1) {
                return -x;
            }
        }
    } catch (const std::exception&) {
        // fall through to warning + default
    }
    RTP_LLM_LOG_WARNING("invalid decode_prefill_ratio '%s', falling back to '1' (alternation)", ratio.c_str());
    return 1;
}
}  // namespace

PDFusionRatioScheduler::PDFusionRatioScheduler(const RuntimeConfig&                   runtime_config,
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
    decode_prefill_step_(parseDecodePrefillRatio(runtime_config.fifo_scheduler_config.decode_prefill_ratio)),
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    metrics_reporter_(metrics_reporter) {
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%zu], max_batch_tokens_size is [%zu]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_);
    RTP_LLM_LOG_INFO("pdfusion ratio scheduler role_type [%d], decode_prefill_ratio [%s], parsed step [%ld]",
                     static_cast<int>(pd_sep_config_.role_type),
                     runtime_config.fifo_scheduler_config.decode_prefill_ratio.c_str(),
                     decode_prefill_step_);
}

PDFusionRatioScheduler::~PDFusionRatioScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory PDFusionRatioScheduler");
}

bool PDFusionRatioScheduler::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_streams_.empty() && loading_cache_streams_.empty() && running_streams_.empty()
           && pending_decode_streams_.empty() && new_streams_.empty();
}

void PDFusionRatioScheduler::cancelStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        stream->moveToNext();
    }
    streams.clear();
}

absl::Status PDFusionRatioScheduler::stop() {
    RTP_LLM_LOG_INFO("stop PDFusionRatioScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
        cancelStreams(waiting_streams_);
        cancelStreams(loading_cache_streams_);
        cancelStreams(running_streams_);
        cancelStreams(pending_decode_streams_);
        cancelStreams(new_streams_);
    }
    cond_.notify_all();
    return absl::OkStatus();
}

int64_t PDFusionRatioScheduler::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

bool PDFusionRatioScheduler::checkInputLength(const GenerateStreamPtr& stream) {
    if (stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
        stream->reportError(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                            autil::StringUtil::formatString("input len " + std::to_string(stream->inputLength())
                                                            + " is greater than kv cache max available tokens num "
                                                            + std::to_string(cache_manager_->maxAvailableTokensNum())));
        return false;
    }
    return true;
}

absl::Status PDFusionRatioScheduler::enqueue(const GenerateStreamPtr& stream) {
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

std::vector<std::shared_ptr<GenerateStream>>
PDFusionRatioScheduler::batchEnqueue(const vector<GenerateStreamPtr>& streams) {
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
    return streams;
}

bool PDFusionRatioScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                                   const GenerateStreamPtr&       new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    const auto in_flight_streams =
        loading_cache_streams_.size() + running_streams_.size() + pending_decode_streams_.size() + streams.size();
    if (in_flight_streams + 1 > max_generate_batch_size_) {
        return false;
    }

    size_t max_token_size = static_cast<size_t>(new_stream->contextLength());
    if (streams.empty() && max_token_size < max_seq_len_) {
        return true;
    }
    for (auto& stream : streams) {
        max_token_size = std::max(max_token_size, static_cast<size_t>(stream->contextLength()));
    }
    return max_token_size * (streams.size() + 1) < max_batch_tokens_size_;
}

bool PDFusionRatioScheduler::waitPredicate() {
    return stop_ || schedule_trigger_ || !waiting_streams_.empty() || !loading_cache_streams_.empty()
           || !running_streams_.empty() || !pending_decode_streams_.empty();
}

size_t PDFusionRatioScheduler::evaluateAndUpdateStreams(list<GenerateStreamPtr>& streams) {
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

void PDFusionRatioScheduler::evaluateWaitingStreams(list<GenerateStreamPtr>& waiting_streams) {
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

void PDFusionRatioScheduler::addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
    switch (new_state) {
        case StreamState::WAITING:
            waiting_streams_.push_back(stream);
            break;
        case StreamState::LOADING_CACHE:
            loading_cache_streams_.push_back(stream);
            break;
        case StreamState::RUNNING:
            new_streams_.push_back(stream);
            break;
        case StreamState::FINISHED:
            break;
        default:
            RTP_LLM_LOG_ERROR("Unknown state: %d for stream [%ld]", static_cast<int>(new_state), stream->streamId());
            break;
    }
}

absl::StatusOr<list<GenerateStreamPtr>> PDFusionRatioScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    schedule_trigger_ = false;

    bool made_progress = false;
    made_progress |= evaluateAndUpdateStreams(loading_cache_streams_) > 0;
    made_progress |= reapErroredWaitingStreams() > 0;
    made_progress |= reapFinished(running_streams_) > 0;
    made_progress |= reapFinished(pending_decode_streams_) > 0;

    const RoundType round = chooseRound();

    if (round == RoundType::PREFILL) {
        const size_t prev_waiting_size = waiting_streams_.size();
        evaluateWaitingStreams(waiting_streams_);
        made_progress |= evaluateAndUpdateStreams(waiting_streams_) > 0;
        if (!new_streams_.empty()) {
            list<GenerateStreamPtr> prefill_batch(new_streams_.begin(), new_streams_.end());
            pending_decode_streams_.insert(pending_decode_streams_.end(), new_streams_.begin(), new_streams_.end());
            new_streams_.clear();
            decode_since_prefill_ = 0;
            prefill_since_decode_ += 1;
            if (waiting_streams_.size() < prev_waiting_size) {
                schedule_trigger_ = true;
            }
            reportMetrics();
            last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
            return prefill_batch;
        }
    }

    made_progress |= evaluateAndUpdateStreams(running_streams_) > 0;
    made_progress |= promotePendingDecodeStreams() > 0;
    if (!running_streams_.empty()) {
        decode_since_prefill_ += 1;
        prefill_since_decode_ = 0;
    }
    if (!pending_decode_streams_.empty() || (made_progress && !waiting_streams_.empty())) {
        schedule_trigger_ = true;
    }
    if (running_streams_.empty() && !made_progress && !waiting_streams_.empty()) {
        cond_.wait_for(lock, kNoProgressScheduleGap, [this] {
            return stop_ || schedule_trigger_ || !loading_cache_streams_.empty() || !running_streams_.empty()
                   || !pending_decode_streams_.empty();
        });
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

PDFusionRatioScheduler::RoundType PDFusionRatioScheduler::chooseRound() {
    if (waiting_streams_.empty()) {
        return RoundType::DECODE;
    }
    if (running_streams_.empty() && pending_decode_streams_.empty()) {
        return RoundType::PREFILL;
    }
    if (decode_prefill_step_ >= 1) {
        return decode_since_prefill_ >= decode_prefill_step_ ? RoundType::PREFILL : RoundType::DECODE;
    }
    const int64_t m = -decode_prefill_step_;
    return prefill_since_decode_ < m ? RoundType::PREFILL : RoundType::DECODE;
}

size_t PDFusionRatioScheduler::reapErroredWaitingStreams() {
    size_t reaped_count = 0;
    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        auto& stream = *it;
        if (stream->hasError()) {
            auto new_state = stream->moveToNext();
            if (new_state != StreamState::FINISHED) {
                RTP_LLM_LOG_ERROR("Unexpected state %d when reaping errored waiting stream [%ld]",
                                  static_cast<int>(new_state),
                                  stream->streamId());
            }
            it = waiting_streams_.erase(it);
            ++reaped_count;
        } else {
            ++it;
        }
    }
    return reaped_count;
}

size_t PDFusionRatioScheduler::reapFinished(std::list<GenerateStreamPtr>& streams) {
    size_t reaped_count = 0;
    for (auto it = streams.begin(); it != streams.end();) {
        auto& stream = *it;
        if (stream->hasError() || stream->hasEvent(StreamEvents::GenerateDone)) {
            auto new_state = stream->moveToNext();
            if (new_state != StreamState::FINISHED) {
                RTP_LLM_LOG_ERROR("Unexpected state %d when reaping finished stream [%ld]",
                                  static_cast<int>(new_state),
                                  stream->streamId());
            }
            it = streams.erase(it);
            ++reaped_count;
        } else {
            ++it;
        }
    }
    return reaped_count;
}

size_t PDFusionRatioScheduler::promotePendingDecodeStreams() {
    size_t promoted_count = 0;
    for (auto it = pending_decode_streams_.begin(); it != pending_decode_streams_.end();) {
        auto& stream    = *it;
        auto  new_state = stream->moveToNext();
        if (new_state == StreamState::RUNNING) {
            running_streams_.push_back(stream);
        } else if (new_state != StreamState::FINISHED) {
            RTP_LLM_LOG_ERROR("Unexpected state %d when promoting pending decode stream [%ld]",
                              static_cast<int>(new_state),
                              stream->streamId());
            addStreamToNewState(stream, new_state);
        }
        it = pending_decode_streams_.erase(it);
        ++promoted_count;
    }
    return promoted_count;
}

int64_t PDFusionRatioScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size();
}

int64_t PDFusionRatioScheduler::runningStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return running_streams_.size();
}

int64_t PDFusionRatioScheduler::pendingDecodeStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return pending_decode_streams_.size();
}

int64_t PDFusionRatioScheduler::decodeSincePrefillForTest() {
    std::lock_guard<mutex> lock(lock_);
    return decode_since_prefill_;
}

int64_t PDFusionRatioScheduler::onflightStreams() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size()
           + pending_decode_streams_.size();
}

std::vector<EngineScheduleInfo::TaskInfo> PDFusionRatioScheduler::waitingTaskList() {
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

std::vector<EngineScheduleInfo::TaskInfo> PDFusionRatioScheduler::runningTaskList() {
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

void PDFusionRatioScheduler::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size           = waiting_streams_.size();
        collector.running_stream_size        = running_streams_.size();
        collector.loading_cache_stream_size  = loading_cache_streams_.size();
        collector.pending_decode_stream_size = pending_decode_streams_.size();
        collector.decode_since_prefill       = decode_since_prefill_;
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
