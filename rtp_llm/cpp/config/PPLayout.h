#pragma once

// ============================================================================
// PP stage layout — the C++ view of the layer partition. All C++ PP
// stage-role decisions go through this struct; new consumers must not
// re-derive the formulas inline.
//
// Layout decisions encoded here:
//   1. Stage capability flags instead of a FIRST/MIDDLE/LAST enum
//      (pp_size=1 is both first and last; enums cannot express that):
//        has_embedding() = pp_rank == 0
//        has_lm_head()   = pp_rank == pp_size - 1
//   2. Layer partition is DATA, not an algorithm: the Python side decides
//      the partition once (default even split, optional model-level
//      partitioner) and materializes it as ParallelismConfig.
//      pp_stage_layer_counts; C++ consumes it by prefix-sum lookup.
//      The even-split formula remains only as a fallback for pp_size=1,
//      stale pickles and legacy test fixtures (empty counts).
// ============================================================================

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
    // Lane geometry, needed for adjacent-stage rank derivation; filled from
    // ParallelismConfig.
    int64_t dp_size = 1;
    int64_t tp_size = 1;
    int64_t dp_rank = 0;
    int64_t tp_rank = 0;

    // Materialized partition: layer count per stage in rank order. Empty
    // means "not materialized" and layerRangeOf falls back to the even
    // split formula.
    std::vector<int64_t> layer_counts;

    // Production constructor: carries the materialized partition over from
    // ParallelismConfig so callers never rebuild it field by field.
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

    // Stage capability flags.
    bool hasEmbedding() const {
        return pp_rank == 0;
    }
    bool hasLmHead() const {
        return pp_rank == pp_size - 1;
    }

    // World-rank stride between adjacent stages of the same (dp, tp) lane
    // under the PP-outermost layout (world_rank = pp*(dp*tp) + dp*tp + tp).
    int64_t laneStride() const {
        return dp_size * tp_size;
    }

    // Ring neighbors: stages wrap around (the last stage's next is stage 0,
    // the return path for sample results). The transport keeps one send
    // channel to next and one receive channel from prev on every stage; the
    // pipeline protocol gates what actually flows on them.
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

    // Half-open layer range [begin, end) assigned to `stage`. With a
    // materialized partition this is a prefix-sum lookup; otherwise it
    // falls back to the even split (remainder goes to earlier stages) for
    // pp_size=1, stale pickles and legacy fixtures only — the golden
    // values stay those of the Python default partition (even_split_counts),
    // e.g. 65 layers, pp_size=4 -> 17/16/16/16.
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

    // This stage's own layer range.
    std::pair<int64_t, int64_t> myLayerRange() const {
        return layerRangeOf(pp_rank);
    }
};

}  // namespace rtp_llm
