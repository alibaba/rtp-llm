#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
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
    FIFOSchedulerBase(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      model_specific_config,
                      cache_manager,
                      metrics_reporter),
    decode_prefill_step_(parseDecodePrefillRatio(runtime_config.fifo_scheduler_config.decode_prefill_ratio)),
    decode_since_prefill_(0),
    prefill_since_decode_(0) {
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

bool PDFusionRatioScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                                   const GenerateStreamPtr&       new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    const auto in_flight_streams = loading_cache_streams_.size() + running_streams_.size()
                                   + pending_decode_streams_.size() + new_streams_.size() + streams.size();
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
           || !running_streams_.empty() || !pending_decode_streams_.empty() || !new_streams_.empty();
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
    made_progress |= refreshAndReapTerminalStreams(waiting_streams_);
    if (has_unclassified_prefill_batch_) {
        made_progress |= classifyActivePrefillBatch();
    } else {
        made_progress |= refreshAndReapTerminalStreams(new_streams_);
    }
    made_progress |= refreshAndReapTerminalStreams(running_streams_);
    made_progress |= refreshAndReapTerminalStreams(pending_decode_streams_);

    const RoundType round = chooseRound();

    if (round == RoundType::PREFILL) {
        if (new_streams_.empty()) {
            const size_t prev_waiting_size = waiting_streams_.size();
            evaluateWaitingStreams(waiting_streams_);
            made_progress |= evaluateAndUpdateStreams(waiting_streams_) > 0;
            if (waiting_streams_.size() < prev_waiting_size) {
                schedule_trigger_ = true;
            }
        }
        if (!new_streams_.empty()) {
            const size_t active_size_before = new_streams_.size();
            auto         prefill_batch      = selectPrefillPrefix(new_streams_);
            made_progress |= new_streams_.size() < active_size_before;
            if (!prefill_batch.empty()) {
                has_unclassified_prefill_batch_ = true;
                decode_since_prefill_          = 0;
                prefill_since_decode_ += 1;
                reportMetrics();
                last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
                return prefill_batch;
            }
        }
    }

    made_progress |= evaluateAndUpdateStreams(running_streams_) > 0;
    made_progress |= promotePendingDecodeStreams();
    if (!running_streams_.empty()) {
        decode_since_prefill_ += 1;
        prefill_since_decode_ = 0;
    }
    if (!pending_decode_streams_.empty() || !new_streams_.empty()
        || (made_progress && !waiting_streams_.empty())) {
        schedule_trigger_ = true;
    }
    if (running_streams_.empty() && !made_progress && !waiting_streams_.empty()) {
        cond_.wait_for(lock, kNoProgressScheduleGap, [this] {
            return stop_ || schedule_trigger_ || !loading_cache_streams_.empty() || !running_streams_.empty()
                   || !pending_decode_streams_.empty() || !new_streams_.empty();
        });
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

PDFusionRatioScheduler::RoundType PDFusionRatioScheduler::chooseRound() {
    if (new_streams_.empty() && waiting_streams_.empty()) {
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

bool PDFusionRatioScheduler::classifyActivePrefillBatch() {
    const bool classified = !new_streams_.empty();
    for (auto it = new_streams_.begin(); it != new_streams_.end();) {
        const auto& stream = *it;
        stream->checkTimeout();

        const auto state    = stream->getStatus();
        const bool terminal = stream->hasError() || stream->hasEvent(StreamEvents::GenerateDone);
        if (state != StreamState::RUNNING || terminal) {
            if (state != StreamState::RUNNING && state != StreamState::FINISHED && !stream->hasError()) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, "invalid active-prefill stream state");
            }
            stream->moveToNext();
            it = new_streams_.erase(it);
        } else if (stream->chunkedPrefillEnabled() && stream->isContextStream()) {
            ++it;
        } else {
            // A completed non-chunk/final prefill is decode-ready, but must not grow decode KV
            // until a DECODE round actually promotes it. Production dispatch already flips this
            // flag; setting it here also keeps fake/test prefills out of later context batches.
            stream->setIsContextStream(false);
            auto current = it++;
            pending_decode_streams_.splice(pending_decode_streams_.end(), new_streams_, current);
        }
    }
    has_unclassified_prefill_batch_ = false;
    return classified;
}

bool PDFusionRatioScheduler::promotePendingDecodeStreams() {
    const bool processed = !pending_decode_streams_.empty();
    for (auto it = pending_decode_streams_.begin(); it != pending_decode_streams_.end();) {
        const auto& stream = *it;
        const auto state = stream->getStatus();
        if ((state != StreamState::RUNNING || stream->isContextStream()) && state != StreamState::FINISHED
            && !stream->hasError()) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "invalid pending-decode stream state");
        }

        const auto new_state = stream->moveToNext();
        if (new_state == StreamState::RUNNING) {
            auto current = it++;
            running_streams_.splice(running_streams_.end(), pending_decode_streams_, current);
        } else {
            if (new_state != StreamState::FINISHED) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, "unexpected pending-decode promotion state");
                stream->moveToNext();
            }
            it = pending_decode_streams_.erase(it);
        }
    }
    return processed;
}

int64_t PDFusionRatioScheduler::pendingDecodeStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    // Preserve the historical test/diagnostic meaning: streams returned by a prefill round but
    // not yet part of running decode, regardless of which classification stage they are in.
    return pending_decode_streams_.size() + new_streams_.size();
}

int64_t PDFusionRatioScheduler::decodeSincePrefillForTest() {
    std::lock_guard<mutex> lock(lock_);
    return decode_since_prefill_;
}

void PDFusionRatioScheduler::cancelExtraStreams() {
    cancelStreams(pending_decode_streams_);
    has_unclassified_prefill_batch_ = false;
    cancelStreams(new_streams_);
}

int64_t PDFusionRatioScheduler::extraOnflightStreams() const {
    return pending_decode_streams_.size() + new_streams_.size();
}

void PDFusionRatioScheduler::fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {
    collector.pending_decode_stream_size = pending_decode_streams_.size();
    collector.decode_since_prefill       = decode_since_prefill_;
}

void PDFusionRatioScheduler::appendExtraRunningTaskList(std::vector<EngineScheduleInfo::TaskInfo>& task_list) const {
    appendTaskInfos(task_list, pending_decode_streams_);
    appendTaskInfos(task_list, new_streams_);
}

}  // namespace rtp_llm
