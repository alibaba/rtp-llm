#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct CacheConfig {
private:
    std::shared_ptr<const CacheTopology> cache_topology;
    std::optional<uint32_t>              finalized_global_block_num_;
    bool                                 configured_sparse_ = false;

public:
    bool use_independent_block_pools = false;

    uint32_t layer_num     = 0;  // the number of main model layers
    uint32_t layer_all_num = 0;  // the number of all layers including mtp modules
    bool     use_mla       = false;

    // Block configuration
    // Global cache-key and cache-manager physical base granularity B. Raw
    // cache keys and request-level block ordinals always use this value.
    size_t seq_size_per_block = 1;

    size_t seqSizePerBlockForGroup(std::string_view tag) const {
        return specForGroup(tag)->seq_size_per_block;
    }

    size_t kernelSeqSizePerBlockForGroup(std::string_view tag) const {
        return specForGroup(tag)->kernel_seq_size_per_block;
    }

    size_t kernelBlocksPerKvBlockForGroup(std::string_view tag) const {
        const auto group_seq    = seqSizePerBlockForGroup(tag);
        const auto group_kernel = kernelSeqSizePerBlockForGroup(tag);
        RTP_LLM_CHECK_WITH_INFO(
            group_kernel > 0 && group_seq >= group_kernel && group_seq % group_kernel == 0,
            "invalid block subdivision: physical seq_size_per_block(%zu), kernel_seq_size_per_block(%zu), tag=%s",
            group_seq,
            group_kernel,
            std::string(tag).c_str());
        return group_seq / group_kernel;
    }

    // Resolve the unique FULL group consumed through kernel-page block tables.
    // All such groups must agree on K because model inputs expose one scalar K.
    std::optional<std::string> kernelAddressedFullGroupTag() const;

    size_t kernelSeqSizePerBlockForModel() const {
        const auto tag = kernelAddressedFullGroupTag();
        return tag.has_value() ? kernelSeqSizePerBlockForGroup(*tag) : seq_size_per_block;
    }

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    static uint32_t
    mtpGlobalLayerId(uint32_t main_layer_num, int module_index, uint32_t module_layer_num, int local_layer_id) {
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        if (module_index < 0 || module_layer_num == 0 || local_layer_id < 0
            || static_cast<uint32_t>(local_layer_id) >= module_layer_num) {
            return invalid;
        }
        const uint64_t global_layer_id = static_cast<uint64_t>(main_layer_num)
                                         + static_cast<uint64_t>(module_index) * module_layer_num
                                         + static_cast<uint32_t>(local_layer_id);
        return global_layer_id < invalid ? static_cast<uint32_t>(global_layer_id) : invalid;
    }

    int groupNums() const {
        return cache_topology == nullptr ? 0 : static_cast<int>(cache_topology->groups().size());
    }

    uint32_t          totalLayerNum() const;
    uint32_t          blockNum() const;
    size_t            groupLayerNum() const;
    size_t            layerBlockStrideBytes(int layer_id) const;
    size_t            explicitReserveBytesForGroup(std::string_view tag) const;
    size_t            explicitlySizedPoolReserveBytes() const;
    bool              usesTypedCacheRegions() const;
    bool              usesOpaqueKVCacheStore() const;
    rtp_llm::DataType cacheDType() const;

    bool isSparse() const {
        return configured_sparse_
               || (cache_topology != nullptr
                   && std::any_of(cache_topology->groups().begin(),
                                  cache_topology->groups().end(),
                                  [](const auto& group) { return group.spec->type == KVCacheSpecType::OpaqueKV; }));
    }

    const CacheTopology& topology() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheConfig topology is not initialized");
        return *cache_topology;
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheConfig topology is not initialized");
        return cache_topology;
    }

    const GroupBase& group(std::string_view tag) const {
        return topology().group(tag);
    }

    CacheTopology::GroupRefs groupsForLayer(int layer_id) const {
        return topology().groupsForLayer(layer_id);
    }

    const GroupBase& groupForLayer(int layer_id, std::string_view tag) const {
        return topology().groupForLayer(layer_id, tag);
    }

    const std::shared_ptr<const KVCacheSpec>& specForGroup(std::string_view tag) const {
        return group(tag).spec;
    }

    CacheGroupType typeForGroup(std::string_view tag) const {
        return group(tag).policy.group_type;
    }

    const std::vector<int>& layerIdsForGroup(std::string_view tag) const {
        return group(tag).layer_ids;
    }

    uint32_t blockNumForGroup(std::string_view tag) const {
        return group(tag).block_num;
    }

    size_t kvBlockStrideBytesForGroup(std::string_view tag) const {
        return group(tag).kv_block_stride_bytes;
    }

    size_t kvScaleStrideBytesForGroup(std::string_view tag) const {
        return group(tag).kv_scale_stride_bytes;
    }

    size_t blockSizeBytesForGroup(std::string_view tag) const {
        return layerIdsForGroup(tag).size() * (kvBlockStrideBytesForGroup(tag) + kvScaleStrideBytesForGroup(tag));
    }

    uint32_t localKvHeadNumForGroup(std::string_view tag) const {
        const auto& group_config = group(tag);
        RTP_LLM_CHECK_WITH_INFO(group_config.local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNumForGroup invalid local_kv_head_num=%u tag=%s",
                                group_config.local_kv_head_num,
                                group_config.tag.c_str());
        return group_config.local_kv_head_num;
    }

    void setGroupPolicies(const std::unordered_map<std::string, CacheGroupPolicy>& policies);

    std::shared_ptr<CacheConfig>
    mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);

    uint32_t explicitIndependentBlocks(std::string_view tag) const {
        return policyForGroup(tag).explicit_block_num;
    }

    bool usesExplicitIndependentBlocks(std::string_view tag) const {
        return explicitIndependentBlocks(tag) > 0;
    }

    CacheGroupPolicy policyForGroup(std::string_view tag) const {
        return group(tag).policy;
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    void        setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers);
    void        finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    std::string debugString(size_t indent = 0) const;
};

}  // namespace rtp_llm
