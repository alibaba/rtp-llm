#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;
namespace rtp_llm {

namespace {
constexpr auto kNoProgressScheduleGap = std::chrono::milliseconds(1);

int remainingKVAllocationSteps(const GenerateStreamPtr& stream) {
    // The final generated token is never used as input to another forward, so it does not get a KV entry.
    return std::max(0, static_cast<int>(stream->maxTokenNum()) - stream->seqLength() - 1);
}

int64_t estimateNeedBlocks(const GenerateStreamPtr& stream, int decode_step) {
    return std::max(stream->estimatePeakNeedBlocks(decode_step), 0);
}

int64_t estimateInitialNeedBlocks(const GenerateStreamPtr& stream) {
    return std::max(stream->estimateInitialNeedBlocks(), 0);
}

bool isAdmittedWaitingStream(const GenerateStreamPtr& stream) {
    return stream->hasEvent(StreamEvents::CanRun) && stream->hasEvent(StreamEvents::LoadInitiated);
}

// Parse the decode_prefill_ratio string into the internal signed cadence step.
//   "0"   (prefill-first) -> 0
//   "N"   (N>=1) -> N    (1 prefill : N decode)
//   "1/X" (X>=1) -> -X   (X prefill : 1 decode)
//   invalid      -> 1    (alternation), with a warning
int64_t parseDecodePrefillRatio(const std::string& ratio) {
    try {
        auto slash = ratio.find('/');
        if (slash == std::string::npos) {
            size_t  pos = 0;
            int64_t n   = std::stoll(ratio, &pos);
            if (pos == ratio.size() && n >= 0) {
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

namespace {

struct AdmissionStreamInfo {
    GenerateStreamPtr stream;
    int               remaining_kv_steps;
};

struct AdmissionPeak {
    int     decode_step = 0;
    int64_t need_blocks = 0;
};

bool longerLifetimeFirst(const AdmissionStreamInfo& lhs, const AdmissionStreamInfo& rhs) {
    return lhs.remaining_kv_steps > rhs.remaining_kv_steps;
}

AdmissionPeak estimateApproximatePeak(const std::vector<AdmissionStreamInfo>& streams) {
    AdmissionPeak peak;
    int64_t       peak_tokens        = -1;
    int64_t       active_base_tokens = 0;
    int64_t       active_batch_size  = 0;

    // Streams are ordered by lifetime. At each distinct lifetime, the prefix visited so far is exactly
    // the set still alive at that decode step, so the aggregate logical-token peak is found in one sweep.
    for (size_t begin = 0; begin < streams.size();) {
        const int decode_step = streams[begin].remaining_kv_steps;
        size_t    end         = begin;
        while (end < streams.size() && streams[end].remaining_kv_steps == decode_step) {
            const auto&   stream     = streams[end].stream;
            const int64_t batch_size = std::max(stream->maxBatchSize(), 1);
            active_base_tokens += batch_size * stream->seqLength();
            active_batch_size += batch_size;
            ++end;
        }
        const int64_t active_tokens = active_base_tokens + active_batch_size * decode_step;
        if (active_tokens > peak_tokens) {
            peak_tokens      = active_tokens;
            peak.decode_step = decode_step;
        }
        begin = end;
    }

    // This aggregate estimate is a best-effort KV admission filter, not a lifecycle capacity guarantee. Before this
    // optimization, PDFusion admitted every candidate that passed the existing batch/token limits, and an
    // allocator-side MALLOC_FAILED was an allowed product outcome. This heuristic only rejects a subset of those
    // candidates, reducing allocation failures without changing the existing failure semantics; it may still
    // underestimate the actual physical-block peak and over-admit work.
    //
    // Summing each stream's independent peak is O(N) to compute, but treats non-coincident peaks as simultaneous and
    // substantially under-admits work, reducing batching concurrency and throughput. Tracking every stream lifetime
    // and KV-group allocation/release boundary would avoid that conservatism, but adds disproportionate implementation
    // complexity and scheduler-lock latency. The aggregate estimate therefore evaluates physical blocks only at the
    // logical-token peak. Independent pools additionally enforce separate physical-peak bounds during admission.
    for (const auto& stream : streams) {
        if (stream.remaining_kv_steps < peak.decode_step) {
            break;
        }
        peak.need_blocks += estimateNeedBlocks(stream.stream, peak.decode_step);
    }
    return peak;
}

}  // namespace

struct PDFusionRatioScheduler::AdmissionPeakState {
    std::vector<AdmissionStreamInfo>          streams;
    AdmissionPeak                             peak;
    int64_t                                   initial_need_blocks = 0;
    std::vector<KVCachePoolAdmissionEstimate> independent_pools;
    std::vector<int64_t>                      max_pool_growth;
    std::vector<int64_t>                      max_pool_peak;
};

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
                                                   const GenerateStreamPtr&       new_stream) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!new_stream->isContextStream()) {
        return false;
    }
    // The peak state is built once per prefill round and updated after every successful admission,
    // so its stream count covers both existing in-flight work and candidates admitted in this round.
    if (!admission_peak_state_) {
        buildAdmissionPeakState();
    }
    const auto in_flight_streams = admission_peak_state_->streams.size();
    if (in_flight_streams + 1 > max_generate_batch_size_) {
        return false;
    }

    size_t max_token_size = static_cast<size_t>(new_stream->contextLength());
    if (streams.empty() && max_token_size < max_seq_len_) {
        return tryAdmitKVForPrefill(new_stream);
    }
    for (auto& stream : streams) {
        max_token_size = std::max(max_token_size, static_cast<size_t>(stream->contextLength()));
    }
    return max_token_size * (streams.size() + 1) < max_batch_tokens_size_
           && tryAdmitKVForPrefill(new_stream);
}

bool PDFusionRatioScheduler::waitPredicate() {
    return stop_ || schedule_trigger_ || !waiting_streams_.empty() || !loading_cache_streams_.empty()
           || !running_streams_.empty() || !pending_decode_streams_.empty();
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
        // Build once for this in-flight snapshot, then update for accepted candidates.
        admission_peak_state_.reset();
        evaluateWaitingStreams(waiting_streams_);
        admission_peak_state_.reset();
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

    const bool                   independent_decode = useIndependentPoolDecode();
    std::list<GenerateStreamPtr> decode_batch;
    if (independent_decode) {
        decode_batch = scheduleIndependentPoolDecode(made_progress);
    } else {
        protected_decode_stream_.reset();
        made_progress |= evaluateAndUpdateStreams(running_streams_) > 0;
        made_progress |= promotePendingDecodeStreams() > 0;
        decode_batch = running_streams_;
    }
    if (!decode_batch.empty()) {
        decode_since_prefill_ += 1;
        prefill_since_decode_ = 0;
    }
    if (!pending_decode_streams_.empty() || (made_progress && !waiting_streams_.empty())) {
        schedule_trigger_ = true;
    }
    if (independent_decode && decode_batch.empty() && !made_progress && !running_streams_.empty()) {
        // A completed reserve owner's asynchronous release may still be in flight.
        cond_.wait_for(lock, kNoProgressScheduleGap, [this] { return stop_ || schedule_trigger_; });
    } else if (running_streams_.empty() && !made_progress && !waiting_streams_.empty()) {
        cond_.wait_for(lock, kNoProgressScheduleGap, [this] {
            return stop_ || schedule_trigger_ || !loading_cache_streams_.empty() || !running_streams_.empty()
                   || !pending_decode_streams_.empty();
        });
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return decode_batch;
}

bool PDFusionRatioScheduler::useIndependentPoolDecode() const {
    if (pd_sep_config_.role_type != RoleType::PDFUSION || !cache_manager_
        || !cache_manager_->cacheConfig().use_independent_block_pools) {
        return false;
    }
    // Beam expansion allocates inside output dispatch. Keep lifetime reservations for that path.
    const auto single_sequence = [](const auto& streams) {
        return std::all_of(
            streams.begin(), streams.end(), [](const auto& stream) { return stream->maxBatchSize() == 1; });
    };
    return single_sequence(running_streams_) && single_sequence(pending_decode_streams_)
           && single_sequence(loading_cache_streams_)
           && std::all_of(waiting_streams_.begin(), waiting_streams_.end(), [](const auto& stream) {
                  return !isAdmittedWaitingStream(stream) || stream->maxBatchSize() == 1;
              });
}

std::list<GenerateStreamPtr> PDFusionRatioScheduler::scheduleIndependentPoolDecode(bool& made_progress) {
    made_progress |= !pending_decode_streams_.empty();
    running_streams_.splice(running_streams_.end(), pending_decode_streams_);
    if (running_streams_.empty()) {
        protected_decode_stream_.reset();
        return {};
    }
    admission_peak_state_.reset();
    buildAdmissionPeakState();
    const auto reserve      = admission_peak_state_->max_pool_peak;
    bool       reserve_fits = true;
    for (size_t gid = 0; gid < reserve.size(); ++gid) {
        reserve_fits &=
            reserve[gid] <= static_cast<int64_t>(admission_peak_state_->independent_pools[gid].available_blocks);
    }
    if (std::find(running_streams_.begin(), running_streams_.end(), protected_decode_stream_)
        == running_streams_.end()) {
        protected_decode_stream_.reset();
    }
    if (!protected_decode_stream_) {
        if (reserve_fits) {
            protected_decode_stream_ = running_streams_.front();
        } else {
            // Actual prefix hits and shared references can leave less capacity than the maximum
            // remaining peak. A shorter request may still finish and release the pinned blocks.
            for (const auto& stream : running_streams_) {
                const auto pools = stream->estimateIndependentPoolNeeds();
                if (std::all_of(pools.begin(), pools.end(), [](const auto& pool) {
                        return pool.peak_need_blocks <= static_cast<int64_t>(pool.available_blocks);
                    })) {
                    protected_decode_stream_ = stream;
                    break;
                }
            }
        }
        // Preserve the standalone fast path for output limits larger than the pool. Deferred
        // release and connector references must drain before this fallback is safe.
        if (!protected_decode_stream_ && admission_peak_state_->streams.size() == 1
            && running_streams_.front()->ownsAllIndependentPoolRequests()) {
            protected_decode_stream_ = running_streams_.front();
        }
    }
    admission_peak_state_.reset();

    // Prefill leaves enough free blocks for any one admitted request to complete. The owner may
    // consume that reserve until it finishes; its newly allocated blocks then become available
    // again. Other requests can allocate only outside the reserve, or run without growing KV.
    // This avoids both lifetime-sum under-admission and a deadlock between partially grown streams.
    std::list<GenerateStreamPtr> batch;
    const auto                   owner   = protected_decode_stream_;
    const auto                   advance = [&](auto it) {
        auto       stream = *it;
        const auto state  = stream->moveToNext();
        if (state == StreamState::RUNNING) {
            batch.push_back(stream);
            return std::next(it);
        }
        made_progress = true;
        it            = running_streams_.erase(it);
        addStreamToNewState(stream, state);
        return it;
    };
    if (owner) {
        advance(std::find(running_streams_.begin(), running_streams_.end(), owner));
    }
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        const auto& stream = *it;
        if (stream == owner) {
            ++it;
            continue;
        }
        bool can_run = stream->hasError() || stream->hasEvent(StreamEvents::GenerateDone);
        if (!can_run) {
            can_run          = true;
            const auto pools = stream->estimateIndependentPoolNeeds(/*include_lifecycle_peak=*/false);
            for (size_t gid = 0; gid < pools.size(); ++gid) {
                if (pools[gid].initial_need_blocks > 0
                    && pools[gid].initial_need_blocks + reserve[gid]
                           > static_cast<int64_t>(pools[gid].available_blocks)) {
                    can_run = false;
                    break;
                }
            }
        }
        if (can_run) {
            it = advance(it);
        } else {
            ++it;
        }
    }
    return batch;
}

PDFusionRatioScheduler::RoundType PDFusionRatioScheduler::chooseRound() {
    if (waiting_streams_.empty()) {
        return RoundType::DECODE;
    }
    if (running_streams_.empty() && pending_decode_streams_.empty()) {
        return RoundType::PREFILL;
    }
    if (decode_prefill_step_ == 0) {
        return RoundType::PREFILL;
    }
    if (decode_prefill_step_ >= 1) {
        return decode_since_prefill_ >= decode_prefill_step_ ? RoundType::PREFILL : RoundType::DECODE;
    }
    const int64_t m = -decode_prefill_step_;
    return prefill_since_decode_ < m ? RoundType::PREFILL : RoundType::DECODE;
}

void PDFusionRatioScheduler::buildAdmissionPeakState() {
    auto state = std::make_unique<AdmissionPeakState>();
    state->streams.reserve(loading_cache_streams_.size() + running_streams_.size() + pending_decode_streams_.size()
                           + waiting_streams_.size());
    auto append_stream = [&](const GenerateStreamPtr& s) {
        const auto pools = s->estimateIndependentPoolNeeds();
        state->streams.push_back({s, pools.empty() ? remainingKVAllocationSteps(s) : 0});
        if (pools.empty()) {
            state->initial_need_blocks += estimateInitialNeedBlocks(s);
        }
        if (state->independent_pools.empty()) {
            state->independent_pools = pools;
            state->max_pool_growth.resize(pools.size());
            state->max_pool_peak.resize(pools.size());
        } else {
            for (size_t gid = 0; gid < pools.size(); ++gid) {
                state->independent_pools[gid].initial_need_blocks += pools[gid].initial_need_blocks;
                state->independent_pools[gid].peak_need_blocks += pools[gid].peak_need_blocks;
            }
        }
        for (size_t gid = 0; gid < pools.size(); ++gid) {
            state->max_pool_growth[gid] =
                std::max(state->max_pool_growth[gid], pools[gid].peak_need_blocks - pools[gid].initial_need_blocks);
            state->max_pool_peak[gid] = std::max(state->max_pool_peak[gid], pools[gid].peak_need_blocks);
        }
    };
    for (const auto& s : loading_cache_streams_) {
        append_stream(s);
    }
    for (const auto& s : running_streams_) {
        append_stream(s);
    }
    for (const auto& s : pending_decode_streams_) {
        append_stream(s);
    }
    for (const auto& s : waiting_streams_) {
        if (isAdmittedWaitingStream(s)) {
            append_stream(s);
        }
    }
    if (state->independent_pools.empty()) {
        std::sort(state->streams.begin(), state->streams.end(), longerLifetimeFirst);
        state->peak = estimateApproximatePeak(state->streams);
    }
    admission_peak_state_ = std::move(state);
}

bool PDFusionRatioScheduler::tryAddToAdmissionPeakState(const GenerateStreamPtr& new_stream,
                                                        int64_t                  initial_capacity,
                                                        int64_t                  lifecycle_capacity) {
    auto&     state              = *admission_peak_state_;
    const auto candidate_pools    = new_stream->estimateIndependentPoolNeeds();
    const int  remaining_kv_steps = candidate_pools.empty() ? remainingKVAllocationSteps(new_stream) : 0;
    // Preserve the idle fast path, where the allocator reports an impossible standalone request.
    const bool    enforce_capacity = !state.streams.empty();
    // Device KV-cache matching happens later in the allocator. At scheduler admission time the actual prefix hit is
    // unknown, so use the conservative no-hit estimate here.
    const int64_t initial_delta = candidate_pools.empty() ? estimateInitialNeedBlocks(new_stream) : 0;

    const bool cooperative = pd_sep_config_.role_type == RoleType::PDFUSION && new_stream->maxBatchSize() == 1
                             && std::all_of(state.streams.begin(), state.streams.end(), [](const auto& info) {
                                    return info.stream->maxBatchSize() == 1;
                                });
    if (enforce_capacity) {
        for (size_t gid = 0; gid < candidate_pools.size(); ++gid) {
            const auto& current           = state.independent_pools[gid];
            const auto& candidate         = candidate_pools[gid];
            const auto  initial_available = current.available_blocks > current.reserve_blocks ?
                                                current.available_blocks - current.reserve_blocks :
                                                0;
            const auto  initial           = current.initial_need_blocks + candidate.initial_need_blocks;
            const auto  growth =
                std::max(state.max_pool_growth[gid], candidate.peak_need_blocks - candidate.initial_need_blocks);
            // Keep a common reserve large enough for any single stream, and arbitrate its use on
            // each decode step. Beam/PD paths retain the sum of lifetime peaks.
            const auto peak = cooperative ? initial + growth : current.peak_need_blocks + candidate.peak_need_blocks;
            if (initial > static_cast<int64_t>(initial_available)
                || peak > static_cast<int64_t>(current.available_blocks)) {
                return false;
            }
        }
    }

    if (candidate_pools.empty() && enforce_capacity
        && (state.initial_need_blocks + initial_delta > initial_capacity
            || state.peak.need_blocks > lifecycle_capacity)) {
        return false;
    }

    const auto insert_it = std::lower_bound(state.streams.begin(),
                                            state.streams.end(),
                                            AdmissionStreamInfo{new_stream, remaining_kv_steps},
                                            longerLifetimeFirst);
    const auto candidate_it   = state.streams.insert(insert_it, {new_stream, remaining_kv_steps});
    auto       candidate_peak = candidate_pools.empty() ? estimateApproximatePeak(state.streams) : AdmissionPeak{};
    // Adding work cannot invalidate a block requirement already sampled before this candidate.
    candidate_peak.need_blocks = std::max(candidate_peak.need_blocks, state.peak.need_blocks);
    if (candidate_pools.empty() && enforce_capacity && candidate_peak.need_blocks > lifecycle_capacity) {
        state.streams.erase(candidate_it);
        return false;
    }

    state.initial_need_blocks += initial_delta;
    state.peak = candidate_peak;
    if (state.independent_pools.empty()) {
        state.independent_pools = candidate_pools;
        state.max_pool_growth.resize(candidate_pools.size());
        state.max_pool_peak.resize(candidate_pools.size());
    } else {
        for (size_t gid = 0; gid < candidate_pools.size(); ++gid) {
            state.independent_pools[gid].initial_need_blocks += candidate_pools[gid].initial_need_blocks;
            state.independent_pools[gid].peak_need_blocks += candidate_pools[gid].peak_need_blocks;
        }
    }
    for (size_t gid = 0; gid < candidate_pools.size(); ++gid) {
        state.max_pool_growth[gid] =
            std::max(state.max_pool_growth[gid],
                     candidate_pools[gid].peak_need_blocks - candidate_pools[gid].initial_need_blocks);
        state.max_pool_peak[gid] = std::max(state.max_pool_peak[gid], candidate_pools[gid].peak_need_blocks);
    }
    return true;
}

bool PDFusionRatioScheduler::tryAdmitKVForPrefill(const GenerateStreamPtr& new_stream) {
    if (!cache_manager_) {
        return false;
    }

    if (!admission_peak_state_) {
        buildAdmissionPeakState();
    }

    const size_t available        = cache_manager_->availableBlocksNum();
    const size_t reserved         = cache_manager_->reserveBlocksNum();
    const size_t initial_capacity = (available > reserved) ? (available - reserved) : 0;
    // Grouped requests are best-effort grouping, not all-or-nothing admission. Members are
    // checked individually against batch-size and prefill-token limits before KV admission;
    // KV gating intentionally preserves that behavior.
    return tryAddToAdmissionPeakState(
        new_stream, static_cast<int64_t>(initial_capacity), static_cast<int64_t>(available));
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

int64_t PDFusionRatioScheduler::pendingDecodeStreamsSize() {
    std::lock_guard<mutex> lock(lock_);
    return pending_decode_streams_.size();
}

int64_t PDFusionRatioScheduler::decodeSincePrefillForTest() {
    std::lock_guard<mutex> lock(lock_);
    return decode_since_prefill_;
}

void PDFusionRatioScheduler::cancelExtraStreams() {
    cancelStreams(pending_decode_streams_);
    cancelStreams(new_streams_);
    protected_decode_stream_.reset();
}

bool PDFusionRatioScheduler::hasExtraStreams() const {
    return !pending_decode_streams_.empty() || !new_streams_.empty();
}

int64_t PDFusionRatioScheduler::extraOnflightStreams() const {
    return pending_decode_streams_.size();
}

void PDFusionRatioScheduler::fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {
    collector.pending_decode_stream_size = pending_decode_streams_.size();
    collector.decode_since_prefill       = decode_since_prefill_;
}

}  // namespace rtp_llm
