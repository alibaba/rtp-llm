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
#include <unordered_set>

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
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%zu], max_batch_tokens_size is [%zu]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_);
}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory FIFOScheduler");
}

bool FIFOScheduler::empty() {
    lock_guard<mutex> lock(lock_);
    return waiting_.empty() && loading_.empty() && running_.empty();
}

void FIFOScheduler::cancelUnits(std::list<ScheduleUnit>& units) {
    for (auto& unit : units) {
        for (auto& stream : unit.streams) {
            stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
            stream->finish();
        }
    }
    units.clear();
}

absl::Status FIFOScheduler::stop() {
    RTP_LLM_LOG_INFO("stop FIFOScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
        cancelUnits(waiting_);
        cancelUnits(loading_);
        cancelUnits(running_);
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
        return false;
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
        ScheduleUnit                unit;
        unit.group_id = stream->isGroup() ? stream->groupId() : -1;
        unit.streams.push_back(stream);
        waiting_.push_back(std::move(unit));
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return absl::OkStatus();
}

std::vector<std::shared_ptr<GenerateStream>> FIFOScheduler::enqueueGroup(const vector<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<std::shared_ptr<GenerateStream>> valid_streams;
    valid_streams.reserve(streams.size());
    for (const auto& stream : streams) {
        if (checkInputLength(stream)) {
            valid_streams.emplace_back(stream);
        }
    }
    if (!valid_streams.empty()) {
        std::lock_guard<std::mutex> lock(lock_);
        bool                        is_group = !valid_streams.empty() && valid_streams[0]->isGroup();
        if (is_group) {
            ScheduleUnit unit;
            unit.group_id = valid_streams[0]->groupId();
            unit.streams  = valid_streams;
            waiting_.push_back(std::move(unit));
        } else {
            for (auto& stream : valid_streams) {
                ScheduleUnit unit;
                unit.group_id = -1;
                unit.streams.push_back(stream);
                waiting_.push_back(std::move(unit));
            }
        }
        schedule_trigger_ = true;
    }
    cond_.notify_all();
    return valid_streams;
}

void FIFOScheduler::accountBatchMetrics(const GenerateStreamPtr& new_stream) {
    for (auto& unit : running_) {
        for (auto& stream : unit.streams) {
            stream->incBatchWithPrefillTimes(1);
            stream->incBatchWithPrefillLen(new_stream->currentExecuteTokenSize());
        }
    }
}

bool FIFOScheduler::waitPredicate() {
    return stop_ || schedule_trigger_ || !waiting_.empty() || !loading_.empty() || !running_.empty();
}

size_t FIFOScheduler::countStreams(const std::list<ScheduleUnit>& queue) const {
    size_t total = 0;
    for (const auto& unit : queue) {
        total += unit.size();
    }
    return total;
}

std::list<GenerateStreamPtr> FIFOScheduler::flattenRunning() const {
    std::list<GenerateStreamPtr> result;
    for (const auto& unit : running_) {
        for (const auto& stream : unit.streams) {
            result.push_back(stream);
        }
    }
    return result;
}

bool FIFOScheduler::canAdmitUnit(size_t              admitted_count,
                                 size_t              admitted_total_tokens,
                                 size_t              running_count,
                                 const ScheduleUnit& unit) const {
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        return running_count + admitted_count + unit.size() <= max_generate_batch_size_;
    }
    if (running_count > 0) {
        return false;
    }
    if (admitted_count + unit.size() > max_generate_batch_size_) {
        return false;
    }
    size_t unit_tokens = 0;
    for (const auto& s : unit.streams) {
        unit_tokens += s->contextLength();
    }
    return admitted_total_tokens + unit_tokens < max_batch_tokens_size_;
}

bool FIFOScheduler::isGroupTimeoutExpired(const ScheduleUnit& unit) const {
    if (!unit.isGroup() || unit.streams.empty()) {
        return false;
    }
    const auto& s          = unit.streams[0];
    int64_t     elapsed_us = autil::TimeUtility::currentTimeInMicroSeconds() - s->enqueueTime();
    return elapsed_us > static_cast<int64_t>(s->groupTimeout()) * 1000;
}

void FIFOScheduler::processGroupUnits() {
    // Group waiting units by group_id
    std::unordered_map<int64_t, std::vector<std::list<ScheduleUnit>::iterator>> group_units;
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
        if (it->isGroup()) {
            group_units[it->group_id].push_back(it);
        }
    }

    for (auto& kv : group_units) {
        auto& units = kv.second;
        if (units.empty() || units[0]->streams.empty()) {
            continue;
        }

        int    group_size    = units[0]->streams[0]->groupSize();
        size_t total_streams = 0;
        for (const auto& it : units) {
            total_streams += it->size();
        }

        if (total_streams >= static_cast<size_t>(group_size)) {
            // Complete group: merge all streams into the first unit
            auto& first_unit = *units[0];
            for (size_t i = 1; i < units.size(); ++i) {
                for (auto& s : units[i]->streams) {
                    first_unit.streams.push_back(std::move(s));
                }
            }
            // Erase the extra units (reverse order; list iterators are stable)
            for (size_t i = units.size() - 1; i >= 1; --i) {
                waiting_.erase(units[i]);
            }
        } else if (isGroupTimeoutExpired(*units[0])) {
            // Incomplete group, timeout expired: dissolve so streams are admitted individually
            for (auto& it : units) {
                it->group_id = -1;
            }
        }
        // Incomplete group, timeout not expired: leave as-is (skipped in admitWaitingUnits)
    }
}

void FIFOScheduler::admitWaitingUnits() {
    size_t  admitted_count        = 0;
    size_t  admitted_total_tokens = 0;
    size_t  running_count         = countStreams(running_);
    int64_t admitted_group_id     = -1;

    // Remove units with pre-existing errors to avoid zombie entries
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        if (it->hasError()) {
            it = waiting_.erase(it);
        } else {
            ++it;
        }
    }

    // Aggregate group streams: merge complete groups, dissolve timed-out incomplete groups
    processGroupUnits();

    for (auto it = waiting_.begin(); it != waiting_.end();) {
        auto& unit = *it;
        if (admitted_count > 0) {
            if (admitted_group_id != -1) {
                break;
            }
            if (unit.isGroup()) {
                ++it;
                continue;
            }
        }
        // Skip incomplete groups that haven't timed out (dissolved groups have group_id = -1)
        if (unit.isGroup() && !unit.streams.empty()) {
            int group_size = unit.streams[0]->groupSize();
            if (unit.size() < static_cast<size_t>(group_size)) {
                ++it;
                continue;
            }
        }
        if (!canAdmitUnit(admitted_count, admitted_total_tokens, running_count, unit)) {
            // Fast-fail: group whose own tokens exceed the cap can never be admitted
            if (unit.isGroup()) {
                size_t group_tokens = 0;
                for (const auto& s : unit.streams) {
                    group_tokens += s->contextLength();
                }
                if (group_tokens >= max_batch_tokens_size_) {
                    for (auto& s : unit.streams) {
                        s->reportError(ErrorCode::GENERATE_TIMEOUT, "group total tokens exceed max_batch_tokens_size");
                    }
                    it = waiting_.erase(it);
                    continue;
                }
            }
            ++it;
            continue;
        }
        if (admitted_count == 0 && unit.isGroup()) {
            admitted_group_id = unit.group_id;
        }
        bool needs_loading = unit.prepare();
        if (!unit.alive()) {
            it = waiting_.erase(it);
            continue;
        }
        admitted_count += unit.size();
        for (const auto& s : unit.streams) {
            admitted_total_tokens += s->contextLength();
        }
        if (needs_loading) {
            loading_.splice(loading_.end(), waiting_, it++);
        } else {
            unit.activate();
            if (unit.alive()) {
                for (auto& s : unit.streams) {
                    if (s->isContextStream()) {
                        RTP_LLM_ACCESS_LOG_INFO("request_activated: %s role=prefill input_len=%d",
                                                s->streamLogTag().c_str(),
                                                s->inputLength());
                    } else {
                        RTP_LLM_ACCESS_LOG_INFO(
                            "request_activated: %s role=decode seq_len=%d", s->streamLogTag().c_str(), s->seqLength());
                    }
                    accountBatchMetrics(s);
                }
                running_.splice(running_.end(), waiting_, it++);
            } else {
                it = waiting_.erase(it);
            }
        }
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

    // 1. running: advance + cleanup
    for (auto it = running_.begin(); it != running_.end();) {
        it->advance();
        if (!it->alive()) {
            for (auto& s : it->streams) {
                RTP_LLM_ACCESS_LOG_INFO("request_finished: %s output_len=%ld iter_count=%ld",
                                        s->streamLogTag().c_str(),
                                        s->outputTokenLen(),
                                        s->iterCount());
            }
            it = running_.erase(it);
        } else {
            ++it;
        }
    }

    // 2. loading: check ready -> activate -> move to running
    //    DECODE streams always enter loading_ (async KV cache loading from prefill).
    //    PREFILL streams may also enter loading_ when prefix caching is enabled
    //    (asyncLoadCache returns true based on cache connector configuration).
    //    Use activated_count to track streams moved to running in this cycle,
    //    so the batch size check accounts for previously activated units.
    //    Similarly, accumulate admitted_total_tokens for token budget checking
    //    (relevant for PREFILL units with prefix caching).
    size_t running_at_step2      = countStreams(running_);
    size_t activated_count       = 0;
    size_t admitted_total_tokens = 0;
    for (auto it = loading_.begin(); it != loading_.end();) {
        if (it->isReady()) {
            if (!canAdmitUnit(activated_count, admitted_total_tokens, running_at_step2, *it)) {
                ++it;
                continue;
            }
            it->activate();
            if (it->alive()) {
                activated_count += it->size();
                for (auto& s : it->streams) {
                    if (s->isContextStream()) {
                        RTP_LLM_ACCESS_LOG_INFO("request_activated: %s role=prefill input_len=%d",
                                                s->streamLogTag().c_str(),
                                                s->inputLength());
                    } else {
                        RTP_LLM_ACCESS_LOG_INFO(
                            "request_activated: %s role=decode seq_len=%d", s->streamLogTag().c_str(), s->seqLength());
                    }
                    admitted_total_tokens += s->contextLength();
                    accountBatchMetrics(s);
                }
                running_.splice(running_.end(), loading_, it++);
            } else {
                it = loading_.erase(it);
            }
        } else {
            ++it;
        }
    }

    // 3. waiting: admit -> prepare -> loading or running
    size_t prev_waiting_size = countStreams(waiting_);
    admitWaitingUnits();
    if (countStreams(waiting_) < prev_waiting_size) {
        schedule_trigger_ = true;
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return flattenRunning();
}

int64_t FIFOScheduler::waitingStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return countStreams(waiting_);
}

int64_t FIFOScheduler::runningStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return countStreams(running_);
}

int64_t FIFOScheduler::onflightStreams() {
    std::lock_guard<mutex> lock(lock_);
    return countStreams(waiting_) + countStreams(loading_) + countStreams(running_);
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::waitingTaskList() {
    std::lock_guard<mutex> lock(lock_);
    waiting_task_list_.clear();
    for (const auto& unit : waiting_) {
        for (const auto& stream : unit.streams) {
            EngineScheduleInfo::TaskInfo task_info;
            task_info.request_id    = stream->streamId();
            task_info.prefix_length = stream->prefixLength();
            task_info.input_length  = stream->inputLength();
            task_info.batch_id      = unit.group_id;
            waiting_task_list_.emplace_back(task_info);
        }
    }
    return waiting_task_list_;
}

std::vector<EngineScheduleInfo::TaskInfo> FIFOScheduler::runningTaskList() {
    std::lock_guard<mutex> lock(lock_);
    running_task_list_.clear();
    for (const auto& unit : running_) {
        for (const auto& stream : unit.streams) {
            EngineScheduleInfo::TaskInfo task_info;
            task_info.request_id    = stream->streamId();
            task_info.prefix_length = stream->prefixLength();
            task_info.input_length  = stream->inputLength();
            task_info.batch_id      = unit.group_id;
            running_task_list_.emplace_back(task_info);
        }
    }
    return running_task_list_;
}

void FIFOScheduler::reportMetrics() {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size          = countStreams(waiting_);
        collector.running_stream_size       = countStreams(running_);
        collector.loading_cache_stream_size = countStreams(loading_);
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
    return;
}

}  // namespace rtp_llm
