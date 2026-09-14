#include "rtp_llm/cpp/model_rpc/StagePeerGroups.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/config/RankLayout.h"

namespace rtp_llm {

std::vector<StagePeerGroup> buildStagePeerGroups(const ParallelismConfig&        parallelism_config,
                                                 const std::vector<std::string>& workers) {
    const int64_t pp_size = std::max<int64_t>(1, parallelism_config.pp_size);
    if (pp_size <= 1) {
        return {};
    }
    const auto& layer_counts = parallelism_config.pp_stage_layer_counts;
    RTP_LLM_CHECK_WITH_INFO(
        !layer_counts.empty(), "pp_size=%ld requires a materialized layer partition (pp_stage_layer_counts)", pp_size);
    const int64_t tp_size = std::max<int64_t>(1, parallelism_config.tp_size);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(workers.size()) == pp_size * tp_size,
                            "worker count %zu does not match pp_size %ld * tp_size %ld",
                            workers.size(),
                            pp_size,
                            tp_size);

    const RankLayout layout       = RankLayout::fromParallelismConfig(parallelism_config);
    const int64_t    total_layers = std::accumulate(layer_counts.begin(), layer_counts.end(), int64_t{0});

    std::vector<StagePeerGroup> groups;
    groups.reserve(static_cast<size_t>(pp_size));
    for (int64_t stage = 0; stage < pp_size; ++stage) {
        const auto [begin, end] = layout.layerRangeOf(stage, total_layers);
        StagePeerGroup group;
        group.range.begin = static_cast<uint32_t>(begin);
        group.range.size  = static_cast<uint32_t>(end - begin);
        const auto first  = workers.begin() + static_cast<ptrdiff_t>(stage * tp_size);
        group.peer_addrs.assign(first, first + static_cast<ptrdiff_t>(tp_size));
        groups.push_back(std::move(group));
    }
    return groups;
}

std::vector<StagePeerSlice> planStagePeerSlices(int prefill_tp, int decode_tp, int decode_tp_rank, bool replicated_kv) {
    RTP_LLM_CHECK_WITH_INFO(
        prefill_tp > 0 && decode_tp > 0, "invalid TP sizes: prefill=%d decode=%d", prefill_tp, decode_tp);
    RTP_LLM_CHECK_WITH_INFO(decode_tp_rank >= 0 && decode_tp_rank < decode_tp,
                            "decode tp_rank %d out of [0, %d)",
                            decode_tp_rank,
                            decode_tp);
    if (replicated_kv) {
        // Every lane owns the full block: one whole-block peer per stage suffices.
        return {{static_cast<size_t>(decode_tp_rank % prefill_tp), 1, 0, 1, 0}};
    }
    RTP_LLM_CHECK_WITH_INFO(prefill_tp % decode_tp == 0 || decode_tp % prefill_tp == 0,
                            "unsupported TP ratio prefill=%d decode=%d",
                            prefill_tp,
                            decode_tp);
    if (prefill_tp == decode_tp) {
        // Symmetric: the same lane index on the prefill stage owns my block.
        return {{static_cast<size_t>(decode_tp_rank), 1, 0, 1, 0}};
    }
    if (prefill_tp > decode_tp) {
        // Prefill TP finer: assemble consecutive peer slices into my block.
        const int                   group_num = prefill_tp / decode_tp;
        std::vector<StagePeerSlice> slices;
        slices.reserve(static_cast<size_t>(group_num));
        for (int j = 0; j < group_num; ++j) {
            slices.push_back({static_cast<size_t>(decode_tp_rank * group_num + j), group_num, j, 1, 0});
        }
        return slices;
    }
    // Decode TP finer: read one sub-slice of a single prefill peer block.
    const int group_num = decode_tp / prefill_tp;
    return {{static_cast<size_t>(decode_tp_rank / group_num), 1, 0, group_num, decode_tp_rank % group_num}};
}

void validateStagePeerGroups(const std::vector<StagePeerGroup>& groups, int64_t total_layers) {
    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "stage peer groups must not be empty");
    uint32_t expected_begin = 0;
    for (const auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(group.range.begin == expected_begin,
                                "stage peer groups do not tile the layer space: expected begin %u, got %u",
                                expected_begin,
                                group.range.begin);
        RTP_LLM_CHECK_WITH_INFO(group.range.size > 0, "stage peer group at layer %u has no layers", group.range.begin);
        RTP_LLM_CHECK_WITH_INFO(
            !group.peer_addrs.empty(), "stage peer group [%u, %u) has no peers", group.range.begin, group.range.end());
        expected_begin = group.range.end();
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(expected_begin) == total_layers,
                            "stage peer groups cover %u layers but the local model has %ld",
                            expected_begin,
                            total_layers);
}

}  // namespace rtp_llm
