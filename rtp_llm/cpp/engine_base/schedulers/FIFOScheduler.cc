#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

using namespace std;
namespace rtp_llm {

namespace {
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
    decode_prefill_step_(parseDecodePrefillRatio(runtime_config.fifo_scheduler_config.decode_prefill_ratio)),
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
    // new_streams_ is schedule-local; include it defensively to catch unfinished transitions.
    return waiting_streams_.empty() && loading_cache_streams_.empty() && running_streams_.empty()
           && pending_decode_streams_.empty() && new_streams_.empty();
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
        cancelStreams(pending_decode_streams_);
        cancelStreams(new_streams_);
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
    if (pd_sep_config_.role_type == RoleType::PDFUSION) {
        // Concurrency cap over the whole PDFUSION in-flight pipeline:
        // running decode-ready + pending-decode + admitted-this-round + 1.
        if (running_streams_.size() + pending_decode_streams_.size() + streams.size() + 1 > max_generate_batch_size_) {
            return false;
        }
        // KV-availability gate: prefilling new_stream must leave the in-flight pipeline able to step.
        // Reserve 1 block of near-term decode headroom per running/pending-decode stream.
        size_t need = static_cast<size_t>(new_stream->nextNeedBlockNums(0));
        for (auto& s : streams) {
            need += static_cast<size_t>(s->nextNeedBlockNums(0));
        }
        const size_t headroom = running_streams_.size() + pending_decode_streams_.size();
        if (cache_manager_->freeBlocksNum() < need + headroom) {
            return false;
        }
        // Token-batch cap for the pure prefill batch. running_streams_ are held-back decode streams
        // in PDFUSION prefill rounds, so they are covered by KV headroom above but not counted here.
        size_t max_token_size = static_cast<size_t>(new_stream->contextLength());
        // A single stream that fits within max_seq_len is always admissible (mirrors the legacy
        // path); this also prevents a hang when max_batch_tokens_size_ is 0.
        if (streams.empty() && max_token_size < max_seq_len_) {
            return true;
        }
        for (auto& stream : streams) {
            max_token_size = std::max(max_token_size, static_cast<size_t>(stream->contextLength()));
        }
        // Keep the legacy strict '< max_batch_tokens_size_' boundary semantics.
        return max_token_size * (streams.size() + 1) < max_batch_tokens_size_;
    }

    // ---- legacy path (DECODE / PREFILL roles), unchanged ----
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
           || !running_streams_.empty() || !pending_decode_streams_.empty();
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
    list<GenerateStreamPtr> new_streams;

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

    int64_t force_batch_group_id = -1;

    for (auto it = waiting_streams.begin(); it != waiting_streams.end();) {
        auto& stream      = *it;
        bool  force_batch = stream->forceBatch();

        // Check if this stream can be scheduled based on batch group rules
        if (force_batch && stream->batchGroupId() != -1) {
            auto& info = request_group_info[stream->batchGroupId()];
            // Check timeout: if expired, treat as normal stream
            if (now - info.first_arrival_time > stream->batchGroupTimeout()) {
                force_batch = false;
            } else if (info.count < stream->batchGroupSize()) {
                // Group incomplete, skip this stream
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
        if (!stream->hasError() && !stream->hasEvent(StreamEvents::CanRun)
            && evaluateRunningMemory(new_streams, stream)) {
            stream->reportEvent(StreamEvents::CanRun);
            new_streams.push_back(stream);

            // Lock batch type based on first scheduled stream
            if (new_streams.size() == 1 && force_batch && stream->batchGroupId() != -1) {
                force_batch_group_id = stream->batchGroupId();
            }
        }
        it++;
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
            if (pd_sep_config_.role_type != RoleType::PDFUSION) {
                accountBatchMetrics(stream);
            }
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

    list<GenerateStreamPtr> batch;
    if (pd_sep_config_.role_type == RoleType::PDFUSION) {
        batch = schedulePrefillFirst();
    } else {
        batch = scheduleLegacy();
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return batch;
}

list<GenerateStreamPtr> FIFOScheduler::scheduleLegacy() {
    // LOADING_CACHE -> DONE/WAITING: error / load cache done
    evaluateAndUpdateStreams(loading_cache_streams_);
    // RUNNING -> DONE: error / finished
    evaluateAndUpdateStreams(running_streams_);

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
    running_streams_.insert(running_streams_.end(), new_streams_.begin(), new_streams_.end());
    new_streams_.clear();

    // If streams were scheduled, trigger next scheduling round
    if (waiting_streams_.size() < prev_waiting_size) {
        schedule_trigger_ = true;
    }
    return running_streams_;
}

list<GenerateStreamPtr> FIFOScheduler::schedulePrefillFirst() {
    // 1. advance async cache loads: a completed LOADING_CACHE stream returns to waiting_streams_
    //    (handleLoading -> WAITING) and is admitted on a later prefill round; errored ones are dropped.
    evaluateAndUpdateStreams(loading_cache_streams_);

    // 2. retire finished streams every round (no incrKVBlock) to release KV promptly
    reapFinished(running_streams_);
    reapFinished(pending_decode_streams_);

    // 3. decide round type (pure cadence + seed)
    const RoundType round = chooseRound();

    if (round == RoundType::PREFILL) {
        const size_t prev_waiting_size = waiting_streams_.size();
        evaluateWaitingStreams(waiting_streams_);    // mark CanRun (gated by evaluateRunningMemory)
        evaluateAndUpdateStreams(waiting_streams_);  // CanRun -> RUNNING into new_streams_
        if (!new_streams_.empty()) {
            list<GenerateStreamPtr> prefill_batch(new_streams_.begin(), new_streams_.end());
            pending_decode_streams_.insert(pending_decode_streams_.end(), new_streams_.begin(), new_streams_.end());
            new_streams_.clear();
            decode_since_prefill_ = 0;
            prefill_since_decode_ += 1;
            if (waiting_streams_.size() < prev_waiting_size) {
                schedule_trigger_ = true;
            }
            return prefill_batch;  // PURE-CONTEXT batch
        }
        // A failed prefill admission is not counted as a prefill round. Fall through to decode and
        // keep cadence pressure so waiting prefill work can be retried promptly next round.
    }

    // DECODE path (cadence decode, or degraded prefill)
    evaluateAndUpdateStreams(running_streams_);  // incrKVBlock-prep survivors for THIS decode (only here)
    promotePendingDecodeStreams();
    if (!running_streams_.empty()) {
        decode_since_prefill_ += 1;
        prefill_since_decode_ = 0;
    }
    if (!pending_decode_streams_.empty() || !waiting_streams_.empty()) {
        schedule_trigger_ = true;  // more work pending; keep the loop moving
    }
    return running_streams_;  // PURE-DECODE batch
}

FIFOScheduler::RoundType FIFOScheduler::chooseRound() {
    if (waiting_streams_.empty()) {
        return RoundType::DECODE;  // nothing to prefill
    }
    if (running_streams_.empty() && pending_decode_streams_.empty()) {
        return RoundType::PREFILL;  // nothing to decode -> must seed
    }
    if (decode_prefill_step_ >= 1) {  // 1 prefill : N decode (N==1 => strict alternation)
        return decode_since_prefill_ >= decode_prefill_step_ ? RoundType::PREFILL : RoundType::DECODE;
    }
    const int64_t m = -decode_prefill_step_;  // prefill-heavy: M prefill : 1 decode
    return prefill_since_decode_ < m ? RoundType::PREFILL : RoundType::DECODE;
}

// Retire streams that finished/errored on their last forward. moveToNext() short-circuits to
// FINISHED before incrKVBlock for these, so this allocates nothing and frees their KV promptly.
void FIFOScheduler::reapFinished(std::list<GenerateStreamPtr>& streams) {
    for (auto it = streams.begin(); it != streams.end();) {
        auto& stream = *it;
        if (stream->hasError() || stream->hasEvent(StreamEvents::GenerateDone)) {
            auto new_state = stream->moveToNext();  // -> FINISHED, releases resources
            if (new_state != StreamState::FINISHED) {
                RTP_LLM_LOG_ERROR("Unexpected state %d when reaping finished stream [%ld]",
                                  static_cast<int>(new_state),
                                  stream->streamId());
            }
            it = streams.erase(it);
        } else {
            ++it;
        }
    }
}

// Decode-path only: move pending-decode streams into running_streams_ as KV allows. A stream that
// can't get its next decode block stays pending and retries on a later decode round (never errored).
// NOTE: FIFOScheduler has no reserve_step_ member (reserve is set per-stream via setReserveStep and
// applied inside incrKVBlock). Pass 0 for the scheduler-side estimate.
void FIFOScheduler::promotePendingDecodeStreams() {
    for (auto it = pending_decode_streams_.begin(); it != pending_decode_streams_.end();) {
        auto&        stream = *it;
        const size_t need   = static_cast<size_t>(stream->nextNeedBlockNums(0));
        if (cache_manager_->freeBlocksNum() >= need) {
            auto new_state = stream->moveToNext();  // incrKVBlock -> RUNNING (decode); or FINISHED on error
            if (new_state == StreamState::RUNNING) {
                running_streams_.push_back(stream);
            } else if (new_state != StreamState::FINISHED) {
                RTP_LLM_LOG_ERROR("Unexpected state %d when promoting pending decode stream [%ld]",
                                  static_cast<int>(new_state),
                                  stream->streamId());
                addStreamToNewState(stream, new_state);
            }
            it = pending_decode_streams_.erase(it);
        } else {
            ++it;  // KV tight: keep pending
        }
    }
}

int64_t FIFOScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size();
}

int64_t FIFOScheduler::runningStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return running_streams_.size();
}

int64_t FIFOScheduler::pendingDecodeStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return pending_decode_streams_.size();
}

int64_t FIFOScheduler::decodeSincePrefillForTest() {
    std::lock_guard<mutex> lock(lock_);
    return decode_since_prefill_;
}

int64_t FIFOScheduler::onflightStreams() {
    std::lock_guard<mutex> lock(lock_);
    return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size()
           + pending_decode_streams_.size();
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
        collector.wait_stream_size           = waiting_streams_.size();
        collector.running_stream_size        = running_streams_.size();
        collector.loading_cache_stream_size  = loading_cache_streams_.size();
        collector.pending_decode_stream_size = pending_decode_streams_.size();
        collector.decode_since_prefill        = decode_since_prefill_;
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
    return;
}

}  // namespace rtp_llm
