#pragma once

/* C++ view of the PP layer partition; all stage-role decisions go through
   this struct. The partition is data materialized by Python into
   ParallelismConfig.pp_stage_layer_counts and consumed by prefix-sum
   lookup; the even-split formula is only a fallback for empty counts. */

#include <cstdint>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct PPLayout {
    int64_t pp_size      = 1;
    int64_t pp_rank      = 0;
    int64_t total_layers = 0;
    // Lane geometry for adjacent-stage rank derivation.
    int64_t dp_size = 1;
    int64_t tp_size = 1;
    int64_t dp_rank = 0;
    int64_t tp_rank = 0;

    // Materialized partition per stage; empty falls back to even split.
    std::vector<int64_t> layer_counts;

    static PPLayout fromParallelismConfig(const ParallelismConfig& config, int64_t total_layers) {
        PPLayout layout;
        layout.pp_size      = config.pp_size;
        layout.pp_rank      = config.pp_rank;
        layout.total_layers = total_layers;
        layout.dp_size      = config.dp_size;
        layout.tp_size      = config.tp_size;
        layout.dp_rank      = config.dp_rank;
        layout.tp_rank      = config.tp_rank;
        layout.layer_counts = config.pp_stage_layer_counts;
        return layout;
    }

    bool hasEmbedding() const {
        return pp_rank == 0;
    }
    bool hasLmHead() const {
        return pp_rank == pp_size - 1;
    }

    // World-rank stride between adjacent stages of the same lane.
    int64_t laneStride() const {
        return dp_size * tp_size;
    }

    // Ring topology: the last stage's next wraps to stage 0 (sample-result return path).
    int64_t prevStage() const {
        return (pp_rank + pp_size - 1) % pp_size;
    }
    int64_t nextStage() const {
        return (pp_rank + 1) % pp_size;
    }

    // World rank of `stage` in this (dp, tp) lane.
    int64_t rankOfStage(int64_t stage) const {
        return stage * laneStride() + dp_rank * tp_size + tp_rank;
    }
    int64_t prevRank() const {
        return rankOfStage(prevStage());
    }
    int64_t nextRank() const {
        return rankOfStage(nextStage());
    }

    /* Half-open layer range [begin, end) of `stage`: prefix-sum over layer_counts;
       the even-split fallback serves pp_size=1, stale pickles and legacy fixtures only. */
    std::pair<int64_t, int64_t> layerRangeOf(int64_t stage) const {
        RTP_LLM_CHECK_WITH_INFO(stage >= 0 && stage < pp_size, "invalid pp stage %ld for pp_size %ld", stage, pp_size);
        if (!layer_counts.empty()) {
            RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(layer_counts.size()) == pp_size,
                                    "pp_stage_layer_counts size %zu != pp_size %ld",
                                    layer_counts.size(),
                                    pp_size);
            int64_t begin = 0;
            for (int64_t s = 0; s < stage; ++s) {
                begin += layer_counts[static_cast<size_t>(s)];
            }
            return {begin, begin + layer_counts[static_cast<size_t>(stage)]};
        }
        const int64_t base  = total_layers / pp_size;
        const int64_t rem   = total_layers % pp_size;
        const int64_t count = base + (stage < rem ? 1 : 0);
        const int64_t begin = stage * base + (stage < rem ? stage : rem);
        return {begin, begin + count};
    }

    std::pair<int64_t, int64_t> myLayerRange() const {
        return layerRangeOf(pp_rank);
    }
};

}  // namespace rtp_llm
