#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace rtp_llm {

inline bool isSingleDevicePrefillCudaGraphConfig(const ParallelismConfig& config) {
    return config.world_size == 1 && config.tp_size == 1 && config.dp_size == 1 && config.ep_size == 1
           && config.pp_size == 1 && config.ffn_sp_size == 1 && config.ffn_tp_size == 1 && !config.enable_sp
           && !config.prefill_cp_config.is_enabled() && !config.prefill_cp_config.is_prefill_enabled()
           && !config.ffn_disaggregate_config.enable_ffn_disaggregate;
}

// The first prefill CUDA Graph implementation uses one fully materialized
// sentinel block table. Sparse LINEAR/SWA groups can contain null entries for
// inactive slots, so replaying a padded token into them is not safe. Keep the
// initial contract deliberately narrow until each sparse topology has its own
// fixed-address scratch representation.
inline bool supportsPrefillCudaGraphCacheTopology(const std::vector<CacheGroupType>& group_types) {
    return group_types.size() == 1 && group_types.front() == CacheGroupType::FULL;
}

inline std::vector<int> defaultPrefillCudaGraphCaptureSeqLens(int64_t max_seq_len) {
    static constexpr int kDefaultMaxBucket = 1024;
    static constexpr int kDefaultBuckets[] = {64, 128, 256, 384, 512, 768, 1024};
    std::vector<int>     buckets;
    if (max_seq_len <= 0) {
        return buckets;
    }
    const int capped_max = static_cast<int>(std::min<int64_t>(max_seq_len, kDefaultMaxBucket));
    for (int bucket : kDefaultBuckets) {
        if (bucket > capped_max) {
            break;
        }
        buckets.push_back(bucket);
    }
    if (buckets.empty() || buckets.back() != capped_max) {
        buckets.push_back(capped_max);
    }
    return buckets;
}

// Prefill graph support for MoE is intentionally narrower than decode
// graph support. The masked DeepGEMM V2 path keeps routing counts on device and
// uses bucket-derived fixed-capacity tensors, while the other MoE strategies
// may use host synchronization, dynamic layouts, or graph-unsafe collectives.
inline bool supportsPrefillCudaGraphMoe(const GptModelDescription& description,
                                        const ParallelismConfig&   parallelism_config,
                                        const MoeConfig&           moe_config) {
    if (!description.ffn_conf.moe_configs.has_value()) {
        return true;
    }

    const auto& model_moe_config = description.ffn_conf.moe_configs.value();
    return description.act_qscheme == QScheme::Qfp8PerTokenBlock
           && moe_config.moe_strategy == "fp8_per_block_no_dp_masked" && moe_config.use_all_gather
           && !moe_config.use_deepep_moe && !moe_config.use_deepep_internode && !moe_config.use_deepep_low_latency
           && !moe_config.use_deepep_p2p_low_latency && !moe_config.use_mori_ep && !moe_config.fake_balance_expert
           && !moe_config.hack_moe_expert && isSingleDevicePrefillCudaGraphConfig(parallelism_config)
           && model_moe_config.tp_size == 1 && model_moe_config.ep_size == 1 && model_moe_config.dp_size == 1
           && model_moe_config.use_all_gather && model_moe_config.expert_num > 0 && model_moe_config.top_k > 0
           && model_moe_config.top_k <= model_moe_config.expert_num && model_moe_config.extra_expert_num == 0
           && !model_moe_config.enable_eplb;
}

}  // namespace rtp_llm
