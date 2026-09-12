#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace rtp_llm {

// A deployment-wide configuration may retain generation-prefill buckets on
// non-PDFUSION roles. Those roles ignore the feature, including its execution-
// mode conflict checks. Keep this scope shared by engine validation and runner
// ownership; speculative execution is checked separately within that scope.
inline bool isGenerationPrefillCudaGraphRequested(const HWKernelConfig& hw_kernel_config, RoleType role_type) {
    return role_type == RoleType::PDFUSION && hw_kernel_config.enable_cuda_graph
           && !hw_kernel_config.generation_prefill_capture_token_buckets.empty();
}

inline bool supportsGenerationPrefillCudaGraphExecutionMode(SpeculativeType speculative_type, bool has_propose_model) {
    return speculative_type == SP_TYPE_NONE && !has_propose_model;
}

inline bool generationPrefillCudaGraphPaddedTokenIndexFitsInt32(int64_t max_requests, int64_t max_bucket) {
    if (max_requests <= 0 || max_bucket <= 0) {
        return false;
    }
    // Padding rows use max_requests * bucket as their offset. The last token
    // in the bucket therefore addresses (max_requests + 1) * bucket - 1.
    // Both inputs originate as int32 configuration values, so this product is
    // safe in int64 even at their largest accepted values.
    const int64_t last_padded_token_index = (max_requests + 1) * max_bucket - 1;
    return last_padded_token_index <= std::numeric_limits<int32_t>::max();
}

inline int64_t generationPrefillCudaGraphReachableRequestCapacity(int64_t max_context_batch_size,
                                                                  int64_t concurrency_limit) {
    if (max_context_batch_size <= 0 || concurrency_limit <= 0) {
        return 0;
    }
    return std::min(max_context_batch_size, concurrency_limit);
}

inline bool generationPrefillCudaGraphMaxRequestsFitsCapacity(int64_t max_requests,
                                                              int64_t max_context_batch_size,
                                                              int64_t concurrency_limit) {
    const int64_t reachable_capacity =
        generationPrefillCudaGraphReachableRequestCapacity(max_context_batch_size, concurrency_limit);
    return max_requests > 0 && max_requests <= HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests
           && max_requests <= reachable_capacity;
}

// Generation prefill is an optional second runner owned only by the normal
// PDFUSION main-generation wrapper. PREFILL/DECODE role wrappers,
// MTP/DSpARK target/draft wrappers, and embedding
// wrappers keep using their primary CUDA-graph roles. In particular, the
// speculative target wrapper does not acquire a generation-prefill runner;
// its initial prompt prefill remains eager in the first implementation.
// This predicate describes wrapper ownership, not configuration acceptance.
// NormalEngine rejects an explicit generation-prefill/speculative combination
// on PDFUSION before any wrapper is constructed; other roles ignore it.
inline bool shouldCreateGenerationPrefillCudaGraph(const HWKernelConfig& hw_kernel_config,
                                                   bool                  allow_cuda_graph,
                                                   bool                  primary_graph_is_prefill,
                                                   RoleType              role_type,
                                                   SpeculativeType       speculative_type) {
    return isGenerationPrefillCudaGraphRequested(hw_kernel_config, role_type) && allow_cuda_graph
           && !primary_graph_is_prefill
           && supportsGenerationPrefillCudaGraphExecutionMode(speculative_type, /*has_propose_model=*/false);
}

inline bool isSingleDeviceGenerationPrefillCudaGraphConfig(const ParallelismConfig& config) {
    return config.world_size == 1 && config.tp_size == 1 && config.dp_size == 1 && config.ep_size == 1
           && config.pp_size == 1 && config.ffn_sp_size == 1 && config.ffn_tp_size == 1 && !config.enable_sp
           && !config.prefill_cp_config.is_enabled() && !config.prefill_cp_config.is_prefill_enabled()
           && !config.ffn_disaggregate_config.enable_ffn_disaggregate;
}

// The first generation-prefill CUDA Graph implementation supports one ordinary FULL cache
// group. Sparse LINEAR/SWA groups have different block-table semantics, so
// routing their padding rows to the reserved block 0 has not been validated.
// Keep the initial contract deliberately narrow until those topologies define
// and test their own graph-padding behavior.
inline bool supportsGenerationPrefillCudaGraphCacheTopology(const std::vector<CacheGroupType>& group_types) {
    return group_types.size() == 1 && group_types.front() == CacheGroupType::FULL;
}

// The CUDA 12.9 open-source image currently packages DeepGEMM 2.1.1. Its
// masked grouped GEMM dispatch has an SM90 recipe, but no SM12x recipe. Keep
// the first-version capability check explicit and fail closed until each new
// architecture passes the masked-MoE graph replay test.
inline bool supportsGenerationPrefillCudaGraphMaskedMoeBackend(int cuda_compute_capability_major) {
    return cuda_compute_capability_major == 9;
}

// Generation-prefill graph support for MoE is intentionally narrower than
// decode graph support. The masked DeepGEMM V2 path keeps routing counts on
// device and uses bucket-derived fixed-capacity tensors, while the other MoE
// strategies may use host synchronization, dynamic layouts, or graph-unsafe
// collectives.
inline bool supportsGenerationPrefillCudaGraphMoe(const GptModelDescription& description,
                                                  const ParallelismConfig&   parallelism_config,
                                                  const MoeConfig&           moe_config,
                                                  bool                       masked_moe_backend_supported) {
    if (!description.ffn_conf.moe_configs.has_value()) {
        return true;
    }

    const auto& model_moe_config = description.ffn_conf.moe_configs.value();
    return masked_moe_backend_supported && description.act_qscheme == QScheme::Qfp8PerTokenBlock
           && moe_config.moe_strategy == "fp8_per_block_no_dp_masked" && moe_config.use_all_gather
           && !moe_config.use_deepep_moe && !moe_config.use_deepep_internode && !moe_config.use_deepep_low_latency
           && !moe_config.use_deepep_p2p_low_latency && !moe_config.use_mori_ep && !moe_config.fake_balance_expert
           && !moe_config.hack_moe_expert && isSingleDeviceGenerationPrefillCudaGraphConfig(parallelism_config)
           && model_moe_config.tp_size == 1 && model_moe_config.ep_size == 1 && model_moe_config.dp_size == 1
           && model_moe_config.use_all_gather && model_moe_config.expert_num > 0 && model_moe_config.top_k > 0
           && model_moe_config.top_k <= model_moe_config.expert_num && model_moe_config.extra_expert_num == 0
           && !model_moe_config.enable_eplb;
}

}  // namespace rtp_llm
