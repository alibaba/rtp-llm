#pragma once

#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm {

// Full-prefill graph support for MoE is intentionally narrower than decode
// graph support. The masked DeepGEMM V2 path keeps routing counts on device and
// uses bucket-derived fixed-capacity tensors, while the other MoE strategies
// may use host synchronization, dynamic layouts, or graph-unsafe collectives.
inline bool supportsFullPrefillCudaGraphMoe(const GptModelDescription& description,
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
           && !moe_config.hack_moe_expert && parallelism_config.tp_size == 1 && parallelism_config.ep_size == 1
           && parallelism_config.dp_size == 1 && parallelism_config.pp_size == 1 && parallelism_config.world_size == 1
           && model_moe_config.tp_size == 1 && model_moe_config.ep_size == 1 && model_moe_config.dp_size == 1
           && model_moe_config.use_all_gather && model_moe_config.expert_num > 0 && model_moe_config.top_k > 0
           && model_moe_config.top_k <= model_moe_config.expert_num && model_moe_config.extra_expert_num == 0
           && !model_moe_config.enable_eplb;
}

}  // namespace rtp_llm
