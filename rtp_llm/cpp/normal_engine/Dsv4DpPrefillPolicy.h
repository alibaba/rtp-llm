#pragma once

#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/model_utils/Dsv4MoeStrategy.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rtp_llm {

inline bool dsv4DpPrefillPhaseSyncEnabled(std::string_view model_type,
                                          const char*      moe_strategy,
                                          RoleType         role,
                                          int64_t          pp_size,
                                          int64_t          tp_size,
                                          int64_t          dp_size,
                                          int64_t          ep_size,
                                          int64_t          world_size,
                                          bool             speculative,
                                          bool             local_cp,
                                          bool             ffn_disaggregate) {
    return model_type == "deepseek_v4" && moe_strategy && std::string_view(moe_strategy) == "sm120_decode"
           && role == RoleType::PDFUSION && pp_size == 1 && tp_size == 1 && dp_size > 1 && ep_size == dp_size
           && world_size == dp_size && !speculative && !local_cp && !ffn_disaggregate;
}

// sm120_decode CUDA-graph decode bakes a WORLD all-gather extent into each
// rank's captured graph. Only dp_decode_batch_sync_ (TP=1, DP=EP=world_size>1,
// separate DECODE role or aligned PDFUSION, no local CP, no FFN
// disaggregation) gives every rank the same batch extent; any other DP>1
// topology captures rank-local buckets and mismatches the collective, so it
// must be rejected at construction instead of hanging or corrupting at
// runtime. True PD PREFILL roles do not decode here; eager decode coordinates
// its rounds inside the executor. PREFILL engines that fall back to local
// decode (max_new_tokens>1) are rejected at admission via
// sm120DecodePrefillLocalDecodeMisaligned instead of this construction check.
inline bool sm120DecodeDpGraphMisaligned(
    const char* moe_strategy, RoleType role, int64_t dp_size, bool enable_cuda_graph, bool dp_decode_batch_sync) {
    return moe_strategy && std::string_view(moe_strategy) == "sm120_decode" && enable_cuda_graph && dp_size > 1
           && role != RoleType::PREFILL && !dp_decode_batch_sync;
}

// Construction-time profile: a PREFILL role with sm120_decode CUDA graphs and
// DP>1 has neither phase sync nor decode-batch sync. Admission rejects local
// decode: a non-fake request with max_new_tokens>1 that is not PD-separated
// (GenerateConfig::pd_separation) and not prefill-only (max_new_tokens==0).
// True PD prefill, prefill-only traffic, first-token/warmup (max_new_tokens==1),
// and fake streams stay allowed.
inline bool sm120DecodePrefillLocalDecodeMisaligned(const char* moe_strategy,
                                                    RoleType    role,
                                                    int64_t     dp_size,
                                                    bool        enable_cuda_graph) {
    return moe_strategy && std::string_view(moe_strategy) == "sm120_decode" && enable_cuda_graph && dp_size > 1
           && role == RoleType::PREFILL;
}

// sm120_fused_moe runs its own DP scheduling in Python, but its decode-capable
// WORLD collectives are not batch-aligned across DP ranks. With DP>1 a rank
// still in prefill while a peer starts decode joins a different collective and
// hangs or corrupts. Reject the explicit strategy name. Unresolved "auto" is
// not rejected here: Python strategy selection (is_pure_cp / physical_dp_size)
// decides whether auto may pick this backend, and NormalEngine has no SM120
// detector to pass a hardware flag. PREFILL-only roles never decode, so they
// remain valid.
inline bool sm120FusedMoeDpDecodeMisaligned(const char* moe_strategy, RoleType role, int64_t dp_size) {
    if (!moe_strategy || dp_size <= 1 || role == RoleType::PREFILL) {
        return false;
    }
    return std::string_view(moe_strategy) == "sm120_fused_moe";
}

// Encode priority into the existing MAX vote. Alternate only when both phases
// have real work, so prefill-only traffic does not run empty decode rounds.
inline int64_t dsv4DpPhaseVote(bool local_prefill, bool local_decode, bool previous_prefill) {
    return previous_prefill ? (local_decode ? 2 : (local_prefill ? 1 : 0)) :
                              (local_prefill ? 2 : (local_decode ? 1 : 0));
}

inline bool dsv4DpPhaseIsPrefill(int64_t vote, bool previous_prefill) {
    return previous_prefill ? vote == 1 : vote == 2;
}

// All EP peers must choose the same collective protocol: eager prefill exchanges
// row counts, whereas captured decode starts with a packed-payload all-gather.
// Deferred decode streams remain owned by the scheduler, unchanged, for its next
// round. Do not turn a real decode stream into a prefill or advance its tokens.
template<typename StreamPtr, typename ReduceAnyPrefill, typename MakeFakePrefill>
bool alignDsv4DpPrefillPhase(std::list<StreamPtr>& streams,
                             ReduceAnyPrefill      reduce_any_prefill,
                             MakeFakePrefill       make_fake_prefill) {
    const bool local_prefill =
        std::any_of(streams.begin(), streams.end(), [](const StreamPtr& stream) { return stream->isContextStream(); });
    // Empty and decode-only ranks must also participate in this decision.
    const bool any_prefill = reduce_any_prefill(local_prefill);
    if (!any_prefill) {
        return false;
    }
    streams.remove_if([](const StreamPtr& stream) { return !stream->isContextStream(); });
    if (streams.empty()) {
        streams.emplace_back(make_fake_prefill());
    }
    return true;
}

// Decode graph buckets and their EP payloads must have the same batch extent
// on every rank, including idle ranks. Count rows rather than requests because
// a stream may own multiple sequences.
template<typename StreamPtr, typename ReduceMaxBatch, typename MakeFakeDecode>
void alignDsv4DpDecodeBatch(std::list<StreamPtr>& streams,
                            ReduceMaxBatch        reduce_max_batch,
                            MakeFakeDecode        make_fake_decode,
                            int64_t               fixed_batch_size = 0,
                            bool                  speculative      = false) {
    auto batch_size = [speculative](const StreamPtr& stream) {
        if (speculative) {
            // Async speculative bookkeeping owns the token history read by
            // currentBatchSize(). These streams have one fixed sequence.
            if (stream->maxBatchSize() != 1 || stream->hasNumBeams()) {
                throw std::runtime_error("SM120 speculative decode requires one non-beam sequence per stream");
            }
            return int64_t{1};
        }
        // Non-beam sequence count is immutable; currentBatchSize() needlessly
        // reads token history that the async dispatch worker may still own.
        // The caller must join that worker before counting variable beam rows.
        return static_cast<int64_t>(stream->hasNumBeams() ? stream->currentBatchSize() : stream->maxBatchSize());
    };
    int64_t local_batch = 0;
    for (const auto& stream : streams) {
        local_batch += batch_size(stream);
    }
    const int64_t target = reduce_max_batch(std::max({int64_t{1}, local_batch, fixed_batch_size}));
    if (target < local_batch || (fixed_batch_size > 0 && target > fixed_batch_size)) {
        throw std::runtime_error("SM120 decode batch exceeds its synchronized batch capacity");
    }
    while (local_batch < target) {
        auto fake = make_fake_decode();
        if (batch_size(fake) != 1) {
            throw std::runtime_error("SM120 decode padding requires a single-sequence fake stream");
        }
        streams.emplace_back(std::move(fake));
        ++local_batch;
    }
}

// Decode batch bound the SM120 captured decode graphs are required to cover.
// Only bounds the engine itself forms count: synchronized steps pad up to
// RTP_LLM_DECODE_FIXED_BS when set, the batch-decode scheduler gates rounds by
// its own batch size when enabled (pass 0 when it does not apply), and
// max_generate_batch_size bounds scheduled streams (see
// sm120DecodeCapturedBelowMaxGenerate). Per-request fan-out is rejected at
// admission via sm120DecodeAdmissionRowCap / sm120DecodeMaxFanOutForAdmission
// (and speculative streams via sm120DecodeRejectsSpeculativeFanOut).
inline int64_t sm120DecodeGraphBatchRequirement(int64_t scheduler_batch_size, int64_t fixed_batch_size) {
    return std::max(scheduler_batch_size, fixed_batch_size > 0 ? fixed_batch_size : int64_t{0});
}

// True when the captured decode graphs end below max_generate_batch_size,
// whether the buckets are the defaults (which end at the concurrency limit)
// or an explicit decode_capture_batch_sizes list. Startup rejects such a
// config: admission could form a decode batch with no graph to replay.
inline bool sm120DecodeCapturedBelowMaxGenerate(int64_t max_generate_batch_size, int64_t largest_captured_batch_size) {
    return max_generate_batch_size > largest_captured_batch_size;
}

// Effective per-step decode row capacity for admission. RTP_LLM_DECODE_FIXED_BS
// binds when set (initScheduler also caps the batch-decode scheduler there);
// otherwise the largest captured graph bucket is the limit. Using the captured
// size alone would admit fan-out that still overflows the fixed synchronized
// step and throw inside alignDsv4DpDecodeBatch.
inline int64_t sm120DecodeAdmissionRowCap(int64_t largest_captured_batch_size, int64_t fixed_batch_size) {
    return fixed_batch_size > 0 ? fixed_batch_size : largest_captured_batch_size;
}

// Conservative per-request fan-out cap so a full scheduler round of streams
// cannot sum past the admission row cap. stream_bound is max_generate_batch_size,
// and when BatchDecodeScheduler is enabled also the raise ceiling (row_cap),
// since UpdateSchedulerInfo may grow rounds up to setMaxBatchSize.
inline int64_t sm120DecodeMaxFanOutForAdmission(int64_t row_cap, int64_t stream_bound) {
    stream_bound = std::max(int64_t{1}, stream_bound);
    return std::max(int64_t{1}, row_cap / stream_bound);
}

// True when one request's fan-out (num_return_sequences or beams) alone
// exceeds the admission row / fan-out cap, leaving no safe graph replay for
// that stream (or for a full scheduler round of such streams).
inline bool sm120DecodeRequestExceedsGraph(int64_t request_rows, int64_t admission_row_limit) {
    return request_rows > admission_row_limit;
}

// Speculative (MTP/Eagle/DSpark) aligned decode requires one non-beam sequence
// per stream — same predicate alignDsv4DpDecodeBatch enforces mid-step.
inline bool sm120DecodeRejectsSpeculativeFanOut(int64_t max_batch_size, bool has_beams) {
    return max_batch_size != 1 || has_beams;
}

// Largest decode batch size the CUDA graph runner captures. Mirrors
// CudaGraphRunner::getDecodeBatchSizesToCapture: an explicit
// decode_capture_batch_sizes list replaces the default buckets, which end at
// the concurrency limit.
inline int64_t largestDecodeCaptureBatchSize(const std::vector<int>& decode_capture_batch_sizes,
                                             int64_t                 concurrency_limit) {
    return decode_capture_batch_sizes.empty() ?
               concurrency_limit :
               static_cast<int64_t>(
                   *std::max_element(decode_capture_batch_sizes.begin(), decode_capture_batch_sizes.end()));
}

}  // namespace rtp_llm
