#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace rtp_llm {

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.reuse_policy == rhs.reuse_policy && lhs.evict_policy == rhs.evict_policy
           && lhs.validate_tail_blocks == rhs.validate_tail_blocks && lhs.prefix_reusable == rhs.prefix_reusable
           && lhs.is_reservable == rhs.is_reservable && lhs.group_type == rhs.group_type;
}

std::shared_ptr<CacheConfig>
CacheConfig::mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "CacheConfig::mergeMTPModule requires destination topology views");
    RTP_LLM_CHECK_WITH_INFO(!propose_config.groups.empty(),
                            "CacheConfig::mergeMTPModule requires propose topology views");
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "CacheConfig::mergeMTPModule invalid module_index=%d", module_index);

    auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
    sub_cfg->block_num     = block_num;
    sub_cfg->layer_all_num = sub_cfg->layer_num;

    const auto mtp_layer_num = propose_config.layer_num;
    const auto total_layers =
        static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index + 1) * mtp_layer_num;
    resizeLayerRoutes(total_layers);
    if (layer_to_block_stride_bytes.size() < total_layers) {
        layer_to_block_stride_bytes.resize(total_layers, 0);
    }

    const auto                              target_group_num = groups.size();
    std::unordered_map<std::string, size_t> propose_gid_by_tag;
    for (size_t gid = 0; gid < propose_config.groups.size(); ++gid) {
        propose_gid_by_tag.emplace(propose_config.tagForGroup(gid), gid);
    }

    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (size_t target_gid = 0; target_gid < target_group_num; ++target_gid) {
        const auto&  tag               = tagForGroup(target_gid);
        const auto   propose_it        = propose_gid_by_tag.find(tag);
        const bool   has_propose_group = propose_it != propose_gid_by_tag.end();
        const auto&  source_config     = has_propose_group ? propose_config : *this;
        const size_t source_gid        = has_propose_group ? propose_it->second : target_gid;
        const auto&  source_group      = source_config.groups[source_gid];

        GroupBase sub_group;
        sub_group.spec                  = source_group.spec->clone();
        sub_group.policy                = source_group.policy;
        sub_group.block_num             = source_group.block_num;
        sub_group.local_kv_head_num     = source_group.local_kv_head_num;
        sub_group.kv_block_stride_bytes = source_group.kv_block_stride_bytes;
        sub_group.kv_scale_stride_bytes = source_group.kv_scale_stride_bytes;

        if (!has_propose_group) {
            sub_groups.push_back(std::move(sub_group));
            continue;
        }

        for (int local_layer_id : propose_config.layerIdsForGroup(source_gid)) {
            if (local_layer_id < 0 || local_layer_id >= static_cast<int>(mtp_layer_num)) {
                continue;
            }
            const auto global_layer_id = mtpGlobalLayerId(main_layer_num, module_index, mtp_layer_num, local_layer_id);
            RTP_LLM_CHECK_WITH_INFO(global_layer_id != std::numeric_limits<uint32_t>::max(),
                                    "CacheConfig::mergeMTPModule invalid global layer: main=%u module=%d "
                                    "module_layers=%u local=%d",
                                    main_layer_num,
                                    module_index,
                                    mtp_layer_num,
                                    local_layer_id);
            const auto global_layer = static_cast<size_t>(global_layer_id);

            sub_group.layer_ids.push_back(local_layer_id);
            auto& sub_layer = sub_layers[static_cast<size_t>(local_layer_id)];
            sub_layer.group_ids.push_back(static_cast<int>(target_gid));
            sub_layer.tag_to_gid[tag] = static_cast<int>(target_gid);

            appendLayerToGroup(target_gid, static_cast<int>(global_layer_id), tag);

            RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(local_layer_id) < sub_cfg->layer_to_block_stride_bytes.size(),
                                    "CacheConfig::mergeMTPModule local layer stride missing layer=%d size=%zu",
                                    local_layer_id,
                                    sub_cfg->layer_to_block_stride_bytes.size());
            layer_to_block_stride_bytes[global_layer] =
                sub_cfg->layer_to_block_stride_bytes[static_cast<size_t>(local_layer_id)];
        }

        sub_groups.push_back(std::move(sub_group));
    }

    RTP_LLM_CHECK_WITH_INFO(sub_groups.size() == target_group_num,
                            "CacheConfig::mergeMTPModule sub group count %zu != target group count %zu",
                            sub_groups.size(),
                            target_group_num);
    for (size_t layer_id = 0; layer_id < sub_layers.size(); ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(!sub_layers[layer_id].group_ids.empty(),
                                "CacheConfig::mergeMTPModule missing group mapping for sub layer %zu",
                                layer_id);
    }

    std::unordered_map<std::string, int> sub_tag_to_gid;
    for (size_t gid = 0; gid < sub_groups.size(); ++gid) {
        RTP_LLM_CHECK_WITH_INFO(
            sub_groups[gid].spec != nullptr, "CacheConfig::mergeMTPModule null sub group spec gid=%zu", gid);
        sub_tag_to_gid.emplace(sub_groups[gid].spec->tag, static_cast<int>(gid));
    }

    sub_cfg->groups                         = std::move(sub_groups);
    sub_cfg->layers                         = std::move(sub_layers);
    sub_cfg->tag_to_gid                     = std::move(sub_tag_to_gid);
    sub_cfg->group_block_layout_initialized = group_block_layout_initialized;
    return sub_cfg;
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig::setTopology requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::setTopology requires positive layer_num");
    RTP_LLM_CHECK_WITH_INFO(new_layers.size() == static_cast<size_t>(layer_num),
                            "CacheConfig::setTopology layer count %zu != layer_num %u",
                            new_layers.size(),
                            layer_num);

    std::unordered_map<std::string, int> new_tag_to_gid;
    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        auto& group = new_groups[gid];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(
            !group.spec->tag.empty(), "CacheConfig::setTopology requires non-empty tag for group %zu", gid);
        const auto [it, inserted] = new_tag_to_gid.emplace(group.spec->tag, static_cast<int>(gid));
        (void)it;
        RTP_LLM_CHECK_WITH_INFO(
            inserted, "CacheConfig::setTopology duplicate group tag=%s gid=%zu", group.spec->tag.c_str(), gid);
        group.spec = group.spec->clone();
    }

    std::vector<std::vector<bool>> group_has_layer(new_groups.size(),
                                                   std::vector<bool>(static_cast<size_t>(layer_num), false));
    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        for (int layer_id : new_groups[gid].layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                    "CacheConfig::setTopology tag=%s has invalid layer id %d for layer_num=%u",
                                    new_groups[gid].spec->tag.c_str(),
                                    layer_id,
                                    layer_num);
            const auto layer_index = static_cast<size_t>(layer_id);
            RTP_LLM_CHECK_WITH_INFO(!group_has_layer[gid][layer_index],
                                    "CacheConfig::setTopology tag=%s has duplicate layer id %d",
                                    new_groups[gid].spec->tag.c_str(),
                                    layer_id);
            group_has_layer[gid][layer_index] = true;
        }
    }

    for (size_t layer_id = 0; layer_id < new_layers.size(); ++layer_id) {
        auto& layer = new_layers[layer_id];
        RTP_LLM_CHECK_WITH_INFO(
            !layer.group_ids.empty(), "CacheConfig::setTopology missing group mapping for layer %zu", layer_id);
        std::unordered_set<int> seen_gids;
        for (int gid : layer.group_ids) {
            RTP_LLM_CHECK_WITH_INFO(gid >= 0 && static_cast<size_t>(gid) < new_groups.size(),
                                    "CacheConfig::setTopology layer %zu has invalid gid %d",
                                    layer_id,
                                    gid);
            RTP_LLM_CHECK_WITH_INFO(seen_gids.emplace(gid).second,
                                    "CacheConfig::setTopology layer %zu has duplicate gid %d",
                                    layer_id,
                                    gid);
            RTP_LLM_CHECK_WITH_INFO(group_has_layer[static_cast<size_t>(gid)][layer_id],
                                    "CacheConfig::setTopology layer %zu gid %d is missing reverse group layer id",
                                    layer_id,
                                    gid);
        }

        for (const auto& [tag, gid] : layer.tag_to_gid) {
            RTP_LLM_CHECK_WITH_INFO(gid >= 0 && static_cast<size_t>(gid) < new_groups.size(),
                                    "CacheConfig::setTopology layer %zu tag=%s has invalid gid %d",
                                    layer_id,
                                    tag.c_str(),
                                    gid);
            RTP_LLM_CHECK_WITH_INFO(tag == new_groups[static_cast<size_t>(gid)].spec->tag,
                                    "CacheConfig::setTopology layer %zu tag=%s does not match gid %d tag=%s",
                                    layer_id,
                                    tag.c_str(),
                                    gid,
                                    new_groups[static_cast<size_t>(gid)].spec->tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(std::find(layer.group_ids.begin(), layer.group_ids.end(), gid)
                                        != layer.group_ids.end(),
                                    "CacheConfig::setTopology layer %zu tag=%s gid %d is not in layer groups",
                                    layer_id,
                                    tag.c_str(),
                                    gid);
        }
    }

    groups                         = std::move(new_groups);
    layers                         = std::move(new_layers);
    tag_to_gid                     = std::move(new_tag_to_gid);
    group_block_layout_initialized = false;
}

void CacheConfig::fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                   const std::vector<std::vector<int>>& layers_by_group,
                                   const std::vector<CacheGroupType>&   types,
                                   const std::vector<std::string>&      tags,
                                   const std::vector<CacheGroupPolicy>& policies) {
    const size_t group_num = specs.size();
    RTP_LLM_CHECK_WITH_INFO(group_num > 0, "CacheConfig::fromGroupedSpecs requires at least one cache spec");
    RTP_LLM_CHECK_WITH_INFO(layers_by_group.size() == group_num,
                            "CacheConfig::fromGroupedSpecs layer group count %zu != spec count %zu",
                            layers_by_group.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(types.size() == group_num,
                            "CacheConfig::fromGroupedSpecs group type count %zu != spec count %zu",
                            types.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(tags.empty() || tags.size() == group_num,
                            "CacheConfig::fromGroupedSpecs tag count %zu != spec count %zu",
                            tags.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(policies.empty() || policies.size() == group_num,
                            "CacheConfig::fromGroupedSpecs policy count %zu != spec count %zu",
                            policies.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromGroupedSpecs requires positive layer_num");

    const bool             has_explicit_policies = !policies.empty();
    std::vector<GroupBase> new_groups;
    std::vector<LayerBase> new_layers(static_cast<size_t>(layer_num));
    new_groups.reserve(group_num);

    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheConfig::fromGroupedSpecs got null spec at group %zu", gid);
        std::string tag = tags.empty() ? spec->tag : tags[gid];
        if (tag.empty() && group_num == 1) {
            tag = "default";
        }
        RTP_LLM_CHECK_WITH_INFO(
            !tag.empty(), "CacheConfig::fromGroupedSpecs requires non-empty tag for cache spec %zu", gid);
        auto stored_spec = spec->clone();
        stored_spec->tag = tag;

        GroupBase group;
        group.spec   = stored_spec;
        group.policy = has_explicit_policies ? policies[gid] : defaultCacheGroupPolicy(types[gid]);
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == types[gid],
                                "CacheConfig::fromGroupedSpecs policy type mismatch gid=%zu policy=%d type=%d",
                                gid,
                                static_cast<int>(group.policy.group_type),
                                static_cast<int>(types[gid]));
        group.layer_ids = layers_by_group[gid];
        new_groups.push_back(group);

        for (int layer_id : layers_by_group[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                    "CacheConfig::fromGroupedSpecs tag=%s has invalid layer id %d for layer_num=%u",
                                    tag.c_str(),
                                    layer_id,
                                    layer_num);
            auto& layer = new_layers[static_cast<size_t>(layer_id)];
            layer.group_ids.push_back(static_cast<int>(gid));
            const auto [it, inserted] = layer.tag_to_gid.emplace(tag, static_cast<int>(gid));
            RTP_LLM_CHECK_WITH_INFO(inserted || it->second == static_cast<int>(gid),
                                    "CacheConfig::fromGroupedSpecs layer %d tag %s maps to both group %d and %zu",
                                    layer_id,
                                    tag.c_str(),
                                    inserted ? static_cast<int>(gid) : it->second,
                                    gid);
        }
    }

    setTopology(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    (void)runtime_config;
    if (global_block_num == 0 || groups.empty()) {
        return;
    }
    for (auto& group : groups) {
        group.block_num = global_block_num;
    }
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Model Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD(layer_all_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    os << "\n";

    os << indent1 << "# Block Configuration:\n";
    OUTPUT_FIELD(block_num);
    OUTPUT_FIELD(seq_size_per_block);
    OUTPUT_FIELD(kernel_seq_size_per_block);
    os << "\n";

    os << indent1 << "# Block Sizing Information:\n";
    OUTPUT_FIELD(kv_block_size_bytes);
    OUTPUT_FIELD(kv_scale_size_bytes);
    OUTPUT_FIELD(block_size_bytes);
    OUTPUT_FIELD(kv_block_stride_bytes);
    OUTPUT_FIELD(kv_scale_stride_bytes);
    os << "\n";

    const auto                    group_policies   = groupPoliciesSnapshot();
    const auto                    group_block_nums = groupBlockNumsSnapshot();
    const auto                    group_layer_ids  = layerGroupIdsSnapshot();
    const auto                    group_tags       = groupTagsSnapshot();
    std::vector<std::vector<int>> layers_by_group;
    layers_by_group.reserve(groups.size());
    for (const auto& group : groups) {
        layers_by_group.push_back(group.layer_ids);
    }

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD(linear_step);
    OUTPUT_FIELD(group_layer_num);
    OUTPUT_FIELD_EXPR("full_group_num",
                      std::count_if(group_policies.begin(), group_policies.end(), [](const CacheGroupPolicy& p) {
                          return p.group_type == CacheGroupType::FULL;
                      }));
    OUTPUT_FIELD_EXPR("linear_group_num",
                      std::count_if(group_policies.begin(), group_policies.end(), [](const CacheGroupPolicy& p) {
                          return p.group_type == CacheGroupType::LINEAR;
                      }));
    os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
    os << "\n";

    os << indent1 << "# Cache Specifications:\n";
    OUTPUT_FIELD_EXPR("groups.size()", groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& spec = groups[i].spec;
        if (!spec) {
            os << indent1 << "groups[" << i << "].spec=null\n";
            continue;
        }

        os << indent1 << "groups[" << i << "] {\n";
        os << spec->debugString(indent + 2);
        os << indent1 << "}\n";
    }
    os << "\n";

    os << indent1 << "# Layer Mapping:\n";
    OUTPUT_FIELD_EXPR("layers_by_group.size()", layers_by_group.size());
    os << indent1 << "layers_by_group=" << rtp_llm::vectorsToString(layers_by_group) << "\n";
    OUTPUT_FIELD_EXPR("group_policies.size()", group_policies.size());
    os << indent1 << "group_types=[";
    for (size_t i = 0; i < group_policies.size(); ++i) {
        os << static_cast<int>(group_policies[i].group_type);
        if (i + 1 < group_policies.size()) {
            os << ",";
        }
    }
    os << "]\n";
    OUTPUT_FIELD_EXPR("group_tags.size()", group_tags.size());
    os << indent1 << "group_tags=[";
    for (size_t i = 0; i < group_tags.size(); ++i) {
        os << group_tags[i];
        if (i + 1 < group_tags.size()) {
            os << ",";
        }
    }
    os << "]\n";
    OUTPUT_FIELD_EXPR("layer_to_group_ids.size()", group_layer_ids.size());
    os << indent1 << "layer_to_group_ids=" << rtp_llm::vectorsToString(group_layer_ids) << "\n";
    os << "\n";

    os << indent1 << "# MTP Configurations:\n";
    OUTPUT_FIELD_EXPR("mtp_sub_configs.size()", mtp_sub_configs.size());
    for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
        const auto& sub = mtp_sub_configs[i];
        if (!sub) {
            os << indent1 << "mtp_sub_configs[" << i << "]=null\n";
            continue;
        }
        os << indent1 << "mtp_sub_configs[" << i << "]:\n";
        os << sub->debugString(indent + 4);
    }
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
