#pragma once

/** C++ parallel rank/stage layout built purely from ParallelismConfig. Covers
 *  the rank lattice (mirrors rtp_llm/models_py/distributed/rank_layout.py),
 *  PP ring peers, and the PP layer partition. Mirrored formulas must stay
 *  identical to the Python side.
 */

#include <cstdint>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct RankLayout {
    int64_t pp_size = 1;
    int64_t pp_rank = 0;
    int64_t dp_size = 1;
    int64_t tp_size = 1;
    int64_t dp_rank = 0;
    int64_t tp_rank = 0;
    /** Materialized PP layer partition per stage; empty falls back to even split. */
    std::vector<int64_t> layer_counts;

    static RankLayout fromParallelismConfig(const ParallelismConfig& config) {
        RankLayout layout;
        layout.pp_size      = config.pp_size;
        layout.pp_rank      = config.pp_rank;
        layout.dp_size      = config.dp_size;
        layout.tp_size      = config.tp_size;
        layout.dp_rank      = config.dp_rank;
        layout.tp_rank      = config.tp_rank;
        layout.layer_counts = config.pp_stage_layer_counts;
        return layout;
    }

    /** Rank lattice helpers shared with the Python layout implementation. */
    int64_t laneStride() const {
        return dp_size * tp_size;
    }
    int64_t stageRank() const {
        return dp_rank * tp_size + tp_rank;
    }
    int64_t worldRankOf(int64_t pp_stage) const {
        return pp_stage * laneStride() + stageRank();
    }

    bool hasEmbedding() const {
        return pp_rank == 0;
    }
    bool hasLmHead() const {
        return pp_rank == pp_size - 1;
    }

    /** PP ring wraps the last stage to stage zero for sampled-result return. */
    int64_t prevStage() const {
        return (pp_rank + pp_size - 1) % pp_size;
    }
    int64_t nextStage() const {
        return (pp_rank + 1) % pp_size;
    }
    int64_t prevRank() const {
        return worldRankOf(prevStage());
    }
    int64_t nextRank() const {
        return worldRankOf(nextStage());
    }

    /** Half-open layer range using materialized counts or the compatibility even split. */
    std::pair<int64_t, int64_t> layerRangeOf(int64_t stage, int64_t total_layers) const {
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

    std::pair<int64_t, int64_t> myLayerRange(int64_t total_layers) const {
        return layerRangeOf(pp_rank, total_layers);
    }

    int64_t firstMoeLayer(int64_t total_layers, const std::vector<int64_t>& moe_layer_ids) const {
        const auto [begin, end] = myLayerRange(total_layers);
        for (const auto layer_id : moe_layer_ids) {
            if (layer_id >= begin && layer_id < end) {
                return layer_id;
            }
        }
        return -1;
    }
};

}  // namespace rtp_llm
