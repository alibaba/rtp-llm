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
           && lhs.sliding_window_size == rhs.sliding_window_size && lhs.shared_chunk_state == rhs.shared_chunk_state
           && lhs.prefill_borrow_tail == rhs.prefill_borrow_tail;
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

std::vector<GroupIdentity> CacheConfig::allGroupIdentities() const {
    std::vector<GroupIdentity> identities;
    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        for (const auto& tag : moduleConfig(module_id).groupTags()) {
            identities.push_back({tag, module_id});
        }
    }
    return identities;
}

void CacheConfig::validateCrossModuleInvariants() const {
    RTP_LLM_CHECK_WITH_INFO(moduleCount() > 0, "cache config requires at least one module");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "cache-key block span must be positive");
    std::vector<std::string> seen_tags;
    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        const auto& module = moduleConfig(module_id);
        RTP_LLM_CHECK_WITH_INFO(module.seq_size_per_block == seq_size_per_block,
                                "module %zu cache-key block span %zu differs from shared span %zu",
                                module_id,
                                module.seq_size_per_block,
                                seq_size_per_block);
        for (const auto& tag : module.groupTags()) {
            RTP_LLM_CHECK_WITH_INFO(
                std::find(seen_tags.begin(), seen_tags.end(), tag) == seen_tags.end(),
                "cache tag '%s' (module %zu) collides with the tag of another module; the Python model "
                "description must declare module-scoped tags (e.g. mtp_<tag>)",
                tag.c_str(),
                module_id);
            seen_tags.push_back(tag);
        }
        for (size_t earlier_id = 0; earlier_id < module_id; ++earlier_id) {
            for (const auto& group : module.groups()) {
                if (!group.policy.enable_prefix_reuse) {
                    continue;
                }
                for (const auto& earlier : moduleConfig(earlier_id).groups()) {
                    if (earlier.policy.enable_prefix_reuse && earlier.policy.group_type == group.policy.group_type) {
                        RTP_LLM_CHECK_WITH_INFO(
                            earlier.policy.explicit_block_num == group.policy.explicit_block_num,
                            "jointly reusable groups %s and %s must use the same explicit block capacity",
                            earlier.tag.c_str(),
                            group.tag.c_str());
                    }
                }
            }
        }
    }
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
    main_config.finalizeBlockNums(global_block_num, runtime_config, &linear_prefill_pool_ratio);
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
    OUTPUT_FIELD_EXPR("local_kv_cache_sharded", (local_kv_cache_sharded ? "true" : "false"));
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
