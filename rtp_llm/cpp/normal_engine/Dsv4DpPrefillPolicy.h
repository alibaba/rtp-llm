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
    if (!reduce_any_prefill(local_prefill)) {
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

}  // namespace rtp_llm
