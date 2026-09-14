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
