#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
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
    max_generate_batch_size_(runtime_config.max_generate_batch_size),
    need_fill_fake_stream_(parallelism_config.dp_size > 1 && parallelism_config.tp_rank == 0),
    metrics_reporter_(metrics_reporter) {
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%d], max_batch_tokens_size is [%d]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_);
}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory FIFOScheduler");
}

bool FIFOScheduler::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_streams_.empty() && running_streams_.empty();
}

absl::Status FIFOScheduler::stop() {
    RTP_LLM_LOG_INFO("stop FIFOScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
    }
    cond_.notify_all();
    return absl::OkStatus();
}

void FIFOScheduler::evaluateRunningRemote() {
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        if ((*it)->needRemoteGenerate() && (*it)->setRemoteGenerate()) {
            remote_running_streams_.emplace_back(*it);
            RTP_LLM_LOG_DEBUG("stream [%ld] move to remote running streams", (*it)->streamId());
            it = running_streams_.erase(it);
        } else {
            ++it;
        }
    }
}

int64_t FIFOScheduler::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

void FIFOScheduler::evictDoneStreams(list<GenerateStreamPtr>& streams) {
    for (auto it = streams.begin(); it != streams.end();) {
        (*it)->checkTimeout();
        if ((*it)->stopped() || (*it)->finished()) {
            // Immediately free resources to run more streams
            (*it)->releaseResource();
            RTP_LLM_LOG_DEBUG("evict stream [%ld]", (*it)->streamId());
            it = streams.erase(it);
        } else {
            ++it;
        }
    }
}

absl::Status FIFOScheduler::enqueue(const GenerateStreamPtr& stream) {
    {
        std::lock_guard<std::mutex> lock(lock_);
        waiting_streams_.emplace_back(stream);
    }
    cond_.notify_all();
    return absl::OkStatus();
}

absl::Status FIFOScheduler::batchEnqueue(const vector<GenerateStreamPtr>& streams) {
    {
        std::lock_guard<std::mutex> lock(lock_);
        waiting_streams_.insert(waiting_streams_.end(), streams.begin(), streams.end());
    }
    cond_.notify_all();
    return absl::OkStatus();
}

int FIFOScheduler::runningNextBlockNum(size_t reserve_step) const {
    int total_need_block_nums = 0;
    for (auto& stream : running_streams_) {
        total_need_block_nums += stream->nextNeedBlockNums(reserve_step);
    }
    return total_need_block_nums;
}

// TODO(xinfei.sxf) Is there any situation where the request cannot be ended?
int FIFOScheduler::evaluateRunningNext(size_t reserve_step) {
    int error_streams = 0;
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        auto result = (*it)->incrKVBlock(reserve_step);
        if (!result.ok()) {
            (*it)->stopAndRelease(ErrorCode::MALLOC_FAILED, "incrKVBlock failed: LACK MEM");
            RTP_LLM_LOG_WARNING("stream [%ld] incr block failed", (*it)->streamId());
            it = running_streams_.erase(it);
            error_streams++;
        } else {
            it++;
        }
    }
    return error_streams;
}

bool FIFOScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                          const GenerateStreamPtr&       new_stream) const {
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        if (running_streams_.size() + streams.size() + 1 < max_generate_batch_size_) {
            return true;
        }
    }
    if (model_specific_config_.load_python_model) {
        // new model py not support prefill and decode togather now
        if (!running_streams_.empty()) {
            return false;
        }
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
    return max_token_size * (streams.size() + 1) + running_streams_.size() < int(max_batch_tokens_size_);
}

bool FIFOScheduler::evaluateNewStream(const list<GenerateStreamPtr>& streams,
                                      const GenerateStreamPtr&       new_stream,
                                      size_t                         reserve_step) {
    if (!evaluateRunningMemory(streams, new_stream)) {
        return false;
    }

    auto result = new_stream->initKVBlock(reserve_step);
    return result.ok();
}

list<GenerateStreamPtr> FIFOScheduler::scheduleNew(size_t reserve_step) {
    list<GenerateStreamPtr> new_streams;
    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        auto& stream = *it;
        if (evaluateNewStream(new_streams, *it, reserve_step)) {
            RTP_LLM_LOG_DEBUG("stream [%ld] add to new queue", stream->streamId());
            // if setRunning fails, it must be in stopped state; release KV blocks and erase immediately
            if (stream->setRunning()) {
                new_streams.emplace_back(stream);
                it = waiting_streams_.erase(it);
            } else {
                RTP_LLM_LOG_WARNING("stream [%ld] set running failed", stream->streamId());
                stream->releaseResource();
                it = waiting_streams_.erase(it);
            }
        } else if (running_streams_.empty() && new_streams.empty() && remote_running_streams_.empty()) {
            // TODO(xinfei.sxf) At this time, we can also release the blocks held by other waiting streams
            RTP_LLM_LOG_WARNING("stream [%ld] can not add to new queue", stream->streamId());
            if (stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
                stream->setStop(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                                "input len " + std::to_string(stream->inputLength())
                                    + " is greater than kv cache max available tokens num "
                                    + std::to_string(cache_manager_->maxAvailableTokensNum()));
            } else if ((size_t)stream->inputLength() * stream->currentBatchSize() > max_batch_tokens_size_) {
                auto error_info =
                    autil::StringUtil::formatString("input len [%d] * batch size [%d] > max_batch_tokens_size [%d]",
                                                    stream->inputLength(),
                                                    stream->currentBatchSize(),
                                                    max_batch_tokens_size_);
                stream->setStop(ErrorCode::MALLOC_FAILED, error_info);
            } else {
                stream->setStop(ErrorCode::MALLOC_FAILED, "LACK MEM");
            }
            stream->releaseResource();
            it = waiting_streams_.erase(it);
        } else {
            break;
        }
    }
    return new_streams;
}

void FIFOScheduler::accountBatchMetrics(const list<GenerateStreamPtr>& new_streams,
                                        const list<GenerateStreamPtr>& running_streams) {
    size_t total_prefill_len = 0;
    for (auto& stream : new_streams) {
        total_prefill_len += stream->currentExecuteTokenSize();
    }
    for (auto& stream : running_streams) {
        stream->incBatchWithPrefillTimes(new_streams.size());
        stream->incBatchWithPrefillLen(total_prefill_len);
    }
}

bool FIFOScheduler::waitPredicate() {
    return stop_ || !waiting_streams_.empty() || !running_streams_.empty() || !remote_running_streams_.empty();
}

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule(size_t reserve_step) {
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }
    evaluateRunningRemote();
    evictDoneStreams(waiting_streams_);
    evictDoneStreams(running_streams_);
    evictDoneStreams(remote_running_streams_);

    // TODO(xinfei.sxf) Those who just kicked out of running may join running again immediately.
    evaluateRunningNext(reserve_step);
    auto new_streams = scheduleNew(reserve_step);
    accountBatchMetrics(new_streams, running_streams_);
    running_streams_.insert(running_streams_.end(), new_streams.begin(), new_streams.end());
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
    return waiting_streams_.size() + running_streams_.size();
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::waitingTaskList() {
    std::lock_guard<mutex> lock(lock_);
    waiting_task_list_.clear();
    waiting_task_list_.reserve(waiting_streams_.size());
    for (const auto& stream : waiting_streams_) {
        EngineScheduleInfo::TaskInfo task_info;
        task_info.inter_request_id = stream->interRequestId();
        task_info.prefix_length    = stream->prefixLength();
        task_info.input_length     = stream->inputLength();
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
        task_info.inter_request_id = stream->interRequestId();
        task_info.prefix_length    = stream->prefixLength();
        task_info.input_length     = stream->inputLength();
        running_task_list_.emplace_back(task_info);
    }
    return running_task_list_;
}

void FIFOScheduler::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size           = waiting_streams_.size();
        collector.running_stream_size        = running_streams_.size();
        collector.remote_running_stream_size = remote_running_streams_.size();
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
    return;
}

}  // namespace rtp_llm
