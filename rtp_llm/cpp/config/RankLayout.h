#pragma once

/** Keep rank-coordinate formulas consistent with rtp_llm/models_py/distributed/rank_layout.py. */

#include <cstdint>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

struct RankLayout {
    int64_t world_size = 1;
    int64_t pp_size    = 1;
    int64_t pp_rank    = 0;
    int64_t dp_size    = 1;
    int64_t tp_size    = 1;
    int64_t dp_rank    = 0;
    int64_t tp_rank    = 0;

    static RankLayout fromParallelismConfig(const ParallelismConfig& config) {
        RankLayout layout;
        layout.world_size = config.world_size;
        layout.pp_size    = config.pp_size;
        layout.pp_rank    = config.pp_rank;
        layout.dp_size    = config.dp_size;
        layout.tp_size    = config.tp_size;
        layout.dp_rank    = config.dp_rank;
        layout.tp_rank    = config.tp_rank;
        return layout;
    }

    /** Rank lattice helpers shared with the Python layout implementation. */
    int64_t laneStride() const {
        return world_size / pp_size;
    }
    int64_t stageRank() const {
        return dp_rank * tp_size + tp_rank;
    }
    int64_t worldRankOf(int64_t pp_stage) const {
        return pp_stage * laneStride() + stageRank();
    }

    bool isFirstStage() const {
        return pp_rank == 0;
    }
    bool isLastStage() const {
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
};

}  // namespace rtp_llm
