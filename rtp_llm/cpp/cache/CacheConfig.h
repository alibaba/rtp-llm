#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

using GroupTag   = std::string;
using CacheLayer = std::vector<GroupTag>;

struct CacheGroup {
    std::string                        tag;
    std::shared_ptr<const KVCacheSpec> spec;
    CacheGroupPolicy                   policy;

    uint32_t block_num             = 0;
    uint32_t local_kv_head_num     = 1;
    size_t   kv_block_stride_bytes = 0;
    size_t   kv_scale_stride_bytes = 0;

    size_t seqSizePerBlock() const {
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheGroup tag=%s has null spec", tag.c_str());
        return spec->seq_size_per_block;
    }

    size_t kernelSeqSizePerBlock() const {
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheGroup tag=%s has null spec", tag.c_str());
        return spec->kernel_seq_size_per_block;
    }

    size_t kernelBlocksPerKvBlock() const {
        const size_t physical = seqSizePerBlock();
        const size_t kernel   = kernelSeqSizePerBlock();
        RTP_LLM_CHECK_WITH_INFO(kernel > 0 && physical % kernel == 0,
                                "group seq_size_per_block(%zu) must be divisible by "
                                "kernel_seq_size_per_block(%zu), tag=%s",
                                physical,
                                kernel,
                                tag.c_str());
        return std::max<size_t>(1, physical / kernel);
    }

    // Boundary block tables use one address per physical block for state/ring
    // groups; only paged FULL groups expose their kernel-page subdivision.
    size_t storedKernelBlocksPerKvBlock() const {
        return policy.group_type == CacheGroupType::FULL ? kernelBlocksPerKvBlock() : 1;
    }
};

struct CacheConfig {
private:
    std::vector<CacheGroup>                           groups_;
    std::vector<CacheLayer>                           layers_;
    std::unordered_map<std::string, size_t>           tag_to_idx_;
    std::unordered_map<std::string, std::vector<int>> tag_to_layer_ids_;

    static void validateAndBuildIndex(std::vector<CacheGroup>&                           groups,
                                      const std::vector<CacheLayer>&                     layers,
                                      std::unordered_map<std::string, size_t>&           tag_to_idx,
                                      std::unordered_map<std::string, std::vector<int>>& tag_to_layer_ids);

public:
    bool use_typed_cache_regions                  = false;
    bool use_opaque_kv_cache_store                = false;
    bool disable_decode_first_malloc_device_reuse = false;

    rtp_llm::DataType dtype = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          layer_num;      // the number of main model layers
    const uint32_t    layer_all_num;  // the number of all layers including mtp modules
    bool              use_mla                 = false;
    bool              enable_hybrid_attention = false;
    bool              is_sparse               = false;

    // Block configuration
    uint32_t block_num          = 0;
    size_t   seq_size_per_block = 1;

    // How many kernel blocks fit inside one physical block of a cache group.
    // Derived and validated, so it stays a method; every plain per-group field is
    // read directly from the tagged group record via group(tag).
    size_t kernelBlocksPerKvBlock(std::string_view tag) const {
        return group(tag).kernelBlocksPerKvBlock();
    }

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig(): layer_num(0), layer_all_num(0) {}
    CacheConfig(std::vector<CacheGroup> new_groups, std::vector<CacheLayer> new_layers, uint32_t main_layer_num);
    CacheConfig(CacheConfig&&) noexcept = default;
    CacheConfig& operator=(CacheConfig&& other) noexcept;

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
        return static_cast<int>(groups_.size());
    }

    const std::vector<CacheGroup>& groups() const {
        return groups_;
    }

    const std::vector<CacheLayer>& layers() const {
        return layers_;
    }

    const CacheGroup&       group(std::string_view tag) const;
    const CacheLayer&       groupsForLayer(int layer_id) const;
    const std::vector<int>& groupLayerIds(std::string_view tag) const;
    const CacheGroup&       groupForLayer(int layer_id, std::string_view tag) const;
    const CacheGroup&       soleGroupForLayer(int layer_id) const;

    bool hasSingleGlobalGroup() const {
        return groups_.size() == 1;
    }
    bool hasOneGroupPerLayer() const;

    // Total bytes one logical block of a cache group occupies across all of the
    // group's layers. Derived, so it stays a method.
    size_t blockSizeBytes(std::string_view tag) const {
        const auto& group_config = group(tag);
        RTP_LLM_CHECK_WITH_INFO(group_config.kv_scale_stride_bytes
                                    <= std::numeric_limits<size_t>::max() - group_config.kv_block_stride_bytes,
                                "CacheConfig::blockSizeBytes stride overflow tag=%s",
                                group_config.tag.c_str());
        const auto stride_bytes = group_config.kv_block_stride_bytes + group_config.kv_scale_stride_bytes;
        const auto layer_count  = groupLayerIds(tag).size();
        RTP_LLM_CHECK_WITH_INFO(
            layer_count == 0 || stride_bytes <= std::numeric_limits<size_t>::max() / layer_count,
            "CacheConfig::blockSizeBytes layer multiplication overflow tag=%s layers=%zu stride=%zu",
            group_config.tag.c_str(),
            layer_count,
            stride_bytes);
        return layer_count * stride_bytes;
    }

    size_t pagedBlockSizeBytes() const;
    size_t swaBlockSizeBytes() const;
    size_t explicitlySizedPoolReserveBytes() const;
    size_t totalGroupBlockSizeBytes() const;

    uint32_t localKvHeadNum(std::string_view tag) const {
        const auto& group_config = group(tag);
        RTP_LLM_CHECK_WITH_INFO(group_config.local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNum invalid local_kv_head_num=%u tag=%s",
                                group_config.local_kv_head_num,
                                group_config.tag.c_str());
        return group_config.local_kv_head_num;
    }

    uint32_t explicitIndependentBlocks(std::string_view tag) const {
        return group(tag).policy.explicit_block_num;
    }

    bool usesExplicitIndependentBlocks(std::string_view tag) const {
        return explicitIndependentBlocks(tag) > 0;
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    void        finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    std::string debugString(size_t indent = 0) const;
};

}  // namespace rtp_llm
