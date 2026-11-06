#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <sstream>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.reservable == rhs.reservable && lhs.explicit_block_num == rhs.explicit_block_num
           && lhs.charge_to_paged_budget == rhs.charge_to_paged_budget && lhs.memory_placement == rhs.memory_placement
           && lhs.active_tail_blocks == rhs.active_tail_blocks && lhs.validate_tail_blocks == rhs.validate_tail_blocks
           && lhs.cp_mapping == rhs.cp_mapping && lhs.cp_slice == rhs.cp_slice
           && lhs.sliding_window_size == rhs.sliding_window_size && lhs.shared_chunk_state == rhs.shared_chunk_state;
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    main_config.setTopology(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::setGroupBlockLayout(const std::vector<std::string>& group_tags,
                                      const std::vector<uint32_t>&    block_nums,
                                      const std::vector<size_t>&      kv_block_stride_bytes,
                                      const std::vector<size_t>&      kv_scale_stride_bytes) {
    main_config.setGroupBlockLayout(group_tags, block_nums, kv_block_stride_bytes, kv_scale_stride_bytes);
}

const std::shared_ptr<const CacheTopology>& CacheConfig::allModuleTopologyPtr() const {
    if (all_module_topology_ == nullptr) {
        std::vector<GroupBase> union_groups;
        std::vector<LayerBase> union_layers;
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            const auto& module_config = moduleConfig(module_id);
            const auto& module_groups = module_config.groups();
            union_groups.insert(union_groups.end(), module_groups.begin(), module_groups.end());
            for (const auto& layer : module_config.topology().layers()) {
                LayerBase union_layer = layer;
                union_layer.layer_id  = static_cast<int>(union_layers.size());
                union_layers.push_back(std::move(union_layer));
            }
        }
        all_module_topology_ = CacheTopology::create(std::move(union_groups), std::move(union_layers));
    }
    return all_module_topology_;
}

void CacheConfig::validateCrossModuleInvariants() const {
    RTP_LLM_CHECK_WITH_INFO(moduleCount() > 0, "cache config requires at least one module");
    // seq_size_per_block is a shared field on CacheConfig, not per-module —
    // no cross-module invariant check needed.
}

void CacheConfig::fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                   const std::vector<std::vector<int>>& layers_by_group,
                                   const std::vector<CacheGroupType>&   types,
                                   const std::vector<std::string>&      tags,
                                   const std::vector<CacheGroupPolicy>& policies) {
    main_config.fromGroupedSpecs(specs, layers_by_group, types, tags, policies);
}

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    RTP_LLM_CHECK_WITH_INFO(global_block_num > 0, "cache configuration requires positive baseline block count");
    for (auto& sub_cfg : mtp_sub_configs) {
        sub_cfg.finalizeBlockNums(global_block_num, runtime_config);
    }

    main_config.finalizeBlockNums(global_block_num, runtime_config);
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Shared Configuration:\n";
    OUTPUT_FIELD(cache_min_replay_tokens);
    OUTPUT_FIELD(cache_key_hash_seed);
    OUTPUT_FIELD(seq_size_per_block);
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    os << indent1 << "main_config:\n";
    os << main_config.debugString(indent + 2);
    os << "\n";

    os << indent1 << "mtp_sub_configs.size()=" << mtp_sub_configs.size() << "\n";
    for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
        os << indent1 << "mtp_sub_configs[" << i << "]:\n";
        os << mtp_sub_configs[i].debugString(indent + 4);
    }
    os << "\n";

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
