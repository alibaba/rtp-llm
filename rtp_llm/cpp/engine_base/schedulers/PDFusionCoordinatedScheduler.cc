#include "rtp_llm/cpp/engine_base/schedulers/PDFusionCoordinatedScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "autil/TimeUtility.h"
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <unistd.h>
namespace rtp_llm {
namespace {
int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
RuntimeConfig schedulerRuntime(const RuntimeConfig& runtime, const PDFusionCadenceConfig& cadence_config) {
    auto effective = runtime;
    if (cadence_config.adaptive) {
        // The inherited admission machinery must not parse an unused legacy ratio.
        effective.fifo_scheduler_config.decode_prefill_ratio = "1";
    }
    return effective;
}
}  // namespace
PDFusionCoordinatedScheduler::PDFusionCoordinatedScheduler(const RuntimeConfig&                   runtime,
                                                           const ModelConfig&                     model,
                                                           const PDSepConfig&                     pd,
                                                           const ParallelismConfig&               parallel,
                                                           const ModelSpecificConfig&             specific,
                                                           const std::shared_ptr<KVCacheManager>& cache,
                                                           const kmonitor::MetricsReporterPtr     reporter,
                                                           int                                    mtp_tokens,
                                                           PDFusionCadenceConfig                  cadence_config):
    PDFusionRatioScheduler(schedulerRuntime(runtime, cadence_config), model, pd, parallel, specific, cache, reporter),
    run_id_(runtime.fifo_scheduler_config.pdfusion_trace_run_id),
    rank_(parallel.dp_rank),
    rank_count_(parallel.dp_size),
    mtp_tokens_(mtp_tokens),
    timeout_ms_(runtime.fifo_scheduler_config.pdfusion_coord_timeout_ms),
    decode_steps_(cadence_config.decodeSteps(runtime.fifo_scheduler_config.decode_prefill_ratio)),
    cadence_config_(cadence_config),
    cadence_(decode_steps_, rank_count_, cadence_config_) {
    if (runtime.fifo_scheduler_config.pdfusion_coord_timeout_ms < 1
        || runtime.fifo_scheduler_config.pdfusion_coord_timeout_ms > 600000) {
        throw std::invalid_argument("global control timeout outside [1,600000] ms");
    }
    if (!runtime.fifo_scheduler_config.pdfusion_schedule_trace
        || runtime.fifo_scheduler_config.max_context_batch_size != 1) {
        throw std::invalid_argument("global cadence requires schedule trace and context batch limit 1");
    }
    if (cadence_config_.adaptive) {
        const auto* path = std::getenv("PDFUSION_CADENCE_DRAIN_FILE");
        if (path && *path) {
            drain_file_ = path;
            if (drain_file_.front() != '/') {
                throw std::invalid_argument("cadence drain marker path must be absolute");
            }
        }
    }
    RTP_LLM_LOG_INFO("PDFUSION_CADENCE adaptive=%ld decode_window=%ld low_water_pct=%ld prefill_max_wait_ms=%ld "
                     "min_decode_share_pct=%ld",
                     cadence_config_.adaptive,
                     cadence_config_.decode_window,
                     cadence_config_.low_water_pct,
                     cadence_config_.prefill_max_wait_ms,
                     cadence_config_.min_decode_share_pct);
}
bool PDFusionCoordinatedScheduler::observeDrainMarker() {
    if (draining_ || drain_file_.empty()) {
        return draining_;
    }
    if (::access(drain_file_.c_str(), F_OK) != 0) {
        if (errno == ENOENT) {
            return false;
        }
        throw std::runtime_error("cannot inspect cadence drain marker");
    }
    // The producer publishes by atomic rename after stopping new submissions.
    // Bind the marker to this control group, so a stale/foreign run cannot drain it.
    std::ifstream marker(drain_file_);
    std::string   token;
    if (!std::getline(marker, token) || token != run_id_ || marker.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid cadence drain marker run identity");
    }
    draining_ = true;
    return true;
}
bool PDFusionCoordinatedScheduler::evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                                                         const GenerateStreamPtr&            candidate) {
    if (!streams.empty()) {
        ++observation_.rejected_batch;
        return false;
    }
    // A cache-loading reservation already passed admission and owns its initial KV.
    // Re-admitting it would count itself twice in the inherited peak snapshot.
    if (candidate->hasEvent(StreamEvents::CanRun) && candidate->hasEvent(StreamEvents::LoadInitiated)
        && candidate->curBlocksNum() > 0 && candidate->isContextStream()) {
        return true;
    }
    if (!prefill_admit_allowed_) {
        return false;
    }
    return PDFusionRatioScheduler::evaluateRunningMemory(streams, candidate);
}
int64_t PDFusionCoordinatedScheduler::extraOnflightStreams() const {
    return pending_decode_streams_.size() + new_streams_.size();
}
void PDFusionCoordinatedScheduler::cancelExtraStreams() {
    PDFusionRatioScheduler::cancelExtraStreams();
    prepared_decode_.clear();
    committed_.clear();
    decode_last_served_us_.clear();
}
PDFusionPreparedState PDFusionCoordinatedScheduler::prepareLocal(int64_t epoch) {
    std::lock_guard<std::mutex> guard(lock_);
    if (epoch != prepared_epoch_ + 1 || committed_epoch_ != prepared_epoch_ || !model_completed_) {
        throw std::logic_error("prepareLocal duplicate or unfinished epoch");
    }
    prepared_epoch_ = epoch;
    PDFusionPreparedState state;
    state.last_model_us = last_model_us_;
    if (stop_) {
        state.stopped = 1;
        return state;
    }
    state.draining     = observeDrainMarker();
    observation_       = {};
    observation_.valid = true;
    evaluateAndUpdateStreams(loading_cache_streams_);
    reapErroredWaitingStreams();
    reapFinished(running_streams_);
    reapFinished(pending_decode_streams_);
    reapFinished(new_streams_);
    if (new_streams_.empty()) {
        reserved_since_us_ = 0;
    }
    evaluateAndUpdateStreams(running_streams_);
    promotePendingDecodeStreams();
    observation_.waiting = waiting_streams_.size();
    observation_.loading = loading_cache_streams_.size();
    observation_.running = running_streams_.size();
    observation_.pending = pending_decode_streams_.size();
    // The global time budget throttles only fresh admission, never revokes an
    // already admitted async load or ready reservation. An empty local Decode
    // batch can always refill, including after cancellation and at the tail.
    prefill_admit_allowed_ = cadence_.prefillAdmissionAllowed() || running_streams_.empty();
    // At most one actual context reservation, including an outstanding async load.
    // evaluateWaitingStreams owns the CanRun/initKVBlock/LoadInitiated transitions.
    if (new_streams_.empty() && loading_cache_streams_.empty()) {
        admitWaitingForCoordination();
    }
    const auto now = nowUs();
    if (new_streams_.empty()) {
        reserved_since_us_ = 0;
    } else if (!reserved_since_us_) {
        reserved_since_us_ = now;
    }
    for (const auto& stream : new_streams_) {
        if (stream->getStatus() != StreamState::RUNNING || !stream->isContextStream()
            || !stream->hasEvent(StreamEvents::LoadInitiated)) {
            throw std::logic_error("prefill reservation is not executable");
        }
        ++state.ready_prefill;
        state.input_tokens += stream->contextLength();
    }
    state.oldest_ready_us = reserved_since_us_ ? now - reserved_since_us_ : 0;
    prepared_decode_      = running_streams_;
    std::unordered_set<int64_t> active;
    for (const auto& stream : prepared_decode_) {
        if (stream->isContextStream() || stream->getStatus() != StreamState::RUNNING) {
            throw std::logic_error("decode reservation is not executable");
        }
        active.insert(stream->streamId());
        const auto entry         = decode_last_served_us_.emplace(stream->streamId(), now).first;
        state.decode_unserved_us = std::max(state.decode_unserved_us, now - entry->second);
        ++state.ready_decode;
    }
    for (auto it = decode_last_served_us_.begin(); it != decode_last_served_us_.end();) {
        if (!active.count(it->first)) {
            it = decode_last_served_us_.erase(it);
        } else {
            ++it;
        }
    }
    state.waiting             = waiting_streams_.size();
    state.loading             = loading_cache_streams_.size();
    state.kv_available        = cache_manager_->availableBlocksNum();
    observation_.kv_available = state.kv_available;
    observation_.kv_reserved  = cache_manager_->reserveBlocksNum();
    for (const auto& stream : waiting_streams_) {
        observation_.oldest_waiting_us = std::max(
            observation_.oldest_waiting_us,
            std::max<int64_t>(0, autil::TimeUtility::currentTimeInMicroSeconds() - stream->schedulerEnqueueTimeUs()));
    }
    return state;
}
std::list<GenerateStreamPtr> PDFusionCoordinatedScheduler::commitLocal(int64_t epoch, PDFusionPlan plan) {
    std::lock_guard<std::mutex> guard(lock_);
    if (epoch != prepared_epoch_ || committed_epoch_ == epoch || stop_) {
        throw std::logic_error("commitLocal duplicate, stale epoch or stop");
    }
    committed_epoch_ = epoch;
    plan_            = plan;
    reapFinished(new_streams_);
    reapFinished(running_streams_);
    committed_.clear();
    if (plan == PDFusionPlan::PREFILL) {
        // No enqueue or admission here: only the prepared, resource-owning handles.
        committed_.splice(committed_.end(), new_streams_);
        pending_decode_streams_.insert(pending_decode_streams_.end(), committed_.begin(), committed_.end());
        reserved_since_us_ = 0;
    } else if (plan == PDFusionPlan::DECODE) {
        for (const auto& stream : prepared_decode_) {
            if (!stream->hasError() && !stream->hasEvent(StreamEvents::GenerateDone)
                && stream->getStatus() == StreamState::RUNNING) {
                committed_.push_back(stream);
            }
        }
    } else if (plan != PDFusionPlan::IDLE) {
        throw std::logic_error("invalid global plan");
    }
    prepared_decode_.clear();
    observation_.intent_prefill = plan == PDFusionPlan::PREFILL;
    observeCommitted(committed_, plan == PDFusionPlan::PREFILL);
    model_completed_ = false;
    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return committed_;
}
absl::Status PDFusionCoordinatedScheduler::stop() {
    std::lock_guard<std::mutex> guard(lock_);
    stop_ = true;
    // Do not release reserved KV while the engine may be executing or waiting
    // for commit acknowledgements. Resource release follows engine-loop join.
    for (const auto* streams :
         {&waiting_streams_, &loading_cache_streams_, &running_streams_, &pending_decode_streams_, &new_streams_}) {
        for (const auto& stream : *streams) {
            stream->reportError(ErrorCode::CANCELLED, "coordinated scheduler stopped");
        }
    }
    cond_.notify_all();
    return absl::OkStatus();
}
void PDFusionCoordinatedScheduler::releaseStoppedStreams() {
    if (coordinator_) {
        coordinator_->abort();
    }
    (void)FIFOSchedulerBase::stop();
}
void PDFusionCoordinatedScheduler::completeModelStep(int64_t model_elapsed_us) {
    std::lock_guard<std::mutex> guard(lock_);
    if (model_completed_) {
        throw std::logic_error("duplicate model completion");
    }
    const auto now = nowUs();
    for (const auto& stream : committed_) {
        decode_last_served_us_[stream->streamId()] = now;
    }
    committed_.clear();
    last_model_us_    = model_elapsed_us >= 0 ? model_elapsed_us : (model_started_us_ ? now - model_started_us_ : 0);
    model_started_us_ = 0;
    cadence_.finish(plan_, final_commits_);
    model_completed_ = true;
}
absl::StatusOr<std::list<GenerateStreamPtr>> PDFusionCoordinatedScheduler::schedule() {
    try {
        // First connection happens on the engine loop, after normal warmup/capture.
        if (!coordinator_) {
            coordinator_ = std::make_unique<PDFusionScheduleCoordinator>(
                run_id_, rank_, decode_steps_, timeout_ms_, 300000, rank_count_, mtp_tokens_, cadence_config_);
        }
        ++epoch_;
        const auto begin    = nowUs();
        const auto local    = prepareLocal(epoch_);
        last_prepared_      = local;
        const auto prepared = nowUs();
        const auto states   = coordinator_->exchange(epoch_, PDFusionScheduleCoordinator::Phase::PREPARE, local);
        plan_               = cadence_.choose(states);
        const auto            planned      = nowUs();
        auto                  streams      = commitLocal(epoch_, plan_);
        const auto            committed_at = nowUs();
        PDFusionPreparedState status;
        if (plan_ == PDFusionPlan::PREFILL) {
            status.ready_prefill = streams.size();
        }
        if (plan_ == PDFusionPlan::DECODE) {
            status.ready_decode = streams.size();
        }
        const auto commits    = coordinator_->exchange(epoch_, PDFusionScheduleCoordinator::Phase::COMMIT, status);
        final_commits_        = commits;
        const auto reconciled = nowUs();
        final_real_count_     = 0;
        ready_mask_           = 0;
        commit_mask_          = 0;
        for (int rank = 0; rank < rank_count_; ++rank) {
            if (states[rank].ready_prefill) {
                ready_mask_ |= 1 << rank;
            }
            const auto count = commits[rank].ready_prefill + commits[rank].ready_decode;
            if (commits[rank].ready_prefill > states[rank].ready_prefill
                || commits[rank].ready_decode > states[rank].ready_decode
                || (plan_ != PDFusionPlan::PREFILL && commits[rank].ready_prefill)
                || (plan_ != PDFusionPlan::DECODE && commits[rank].ready_decode)) {
                throw std::logic_error("commit exceeds prepared plan");
            }
            if (count) {
                commit_mask_ |= 1 << rank;
            }
            final_real_count_ += count;
        }
        prepare_us_       = prepared - begin;
        control_us_       = planned - prepared + reconciled - committed_at;
        commit_us_        = committed_at - planned;
        model_started_us_ = final_real_count_ ? nowUs() : 0;
        if (!final_real_count_) {
            completeModelStep();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return streams;
    } catch (const std::exception& error) {
        if (coordinator_) {
            coordinator_->abort();
        }
        return absl::InternalError(error.what());
    }
}
}  // namespace rtp_llm
