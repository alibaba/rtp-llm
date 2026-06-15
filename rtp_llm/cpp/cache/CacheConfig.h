#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct CacheConfig {
    // Cache specification and layer mapping
    std::vector<KVCacheSpecPtr>    cache_specs;
    std::vector<std::vector<int>>  global_layer_ids;  // including mtp module layers
    std::vector<std::vector<int>>  layer_ids;
    std::vector<std::vector<int>>  linear_groups;  // for hybrid attention
    std::vector<std::vector<int>>  full_groups;    // for hybrid attention
    std::vector<CacheGroupType>    group_types;    // for hybrid attention
    std::vector<CacheGroupType>    layer_group_types;
    std::vector<KVCacheRegionName> group_region_names;        // group id -> cache identity
    std::vector<std::string>       group_tags;                // group id -> semantic cache tag
    std::vector<std::vector<int>>  layer_to_group_ids;        // layer id -> all group ids needed by the layer
    std::vector<std::vector<int>>  layer_region_to_group_id;  // layer id -> region name id -> group id
    std::vector<std::map<std::string, int>> layer_tag_to_group_id;  // layer id -> semantic tag -> group id
    std::vector<int>               layer_to_group_id;
    std::vector<int>               layer_to_block_stride_bytes;
    std::vector<size_t>            group_seq_size_per_block;
    std::vector<size_t>            group_kv_block_stride_bytes;
    std::vector<size_t>            group_kv_scale_stride_bytes;
    std::vector<size_t>            group_block_size_bytes;
    std::vector<uint32_t>          group_block_nums;
    uint32_t                       dsv4_fixed_pool_blocks                  = 0;
    uint32_t                       dsv4_hca_state_pool_blocks              = 0;
    bool                           use_independent_block_pools              = false;
    bool                           use_typed_cache_regions                  = false;
    bool                           use_opaque_kv_cache_store                = false;
    bool                           disable_decode_first_malloc_device_reuse = false;

    // Model configuration
    rtp_llm::DataType dtype;
    uint32_t          layer_num;      // the number of main model layers
    uint32_t          layer_all_num;  // the number of all layers including mtp modules
    bool              use_mla   = false;
    bool              is_sparse = false;

    // Block configuration
    uint32_t block_num;
    size_t   seq_size_per_block        = 1;
    size_t   kernel_seq_size_per_block = 1;

    // Returns how many kernel blocks fit inside one physical (kv-manager) block.
    size_t kernelBlocksPerKvBlock() const {
        if (kernel_seq_size_per_block == 0) {
            return 1;
        }
        return std::max<size_t>(1, seq_size_per_block / kernel_seq_size_per_block);
    }

    // Block sizing information
    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size_bytes  = 0;
    size_t kv_scale_size_bytes  = 0;
    size_t block_size_bytes     = 0;  // (kv + scales together)
    size_t swa_block_size_bytes = 0;  // SWA groups (joint allocation with full groups)

    // Per-block bytes of all DSV4 STATE pools (INDEXER_STATE, CSA_STATE,
    // HCA_STATE). Excluded from the HBM fixed-pool reservation only when
    // fixed_pool_uses_pinned_cpu is true.
    size_t state_block_size_bytes = 0;

    // True when DSV4 fixed pools (STATE pools and SWA_KV) should be allocated
    // on pinned CPU memory and excluded from HBM fixed-pool reservation.
    bool fixed_pool_uses_pinned_cpu = false;

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Bytes pre-reserved for fixed-allocation pools (e.g. DSV4 state / SWA pools).
    // CacheConfigCreator deducts this from kv_cache_mem_size before computing the
    // paged block_num, so paged pools don't overcommit HBM. 0 means no reservation.
    size_t fixed_pool_reserve_bytes = 0;

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks
    int linear_fixed_cap =
        0;  // >0 = ring buffer of this many blocks per LINEAR group (per request); 0 = legacy unbounded
    int group_layer_num  = 1;  // Number of layers per group for hybrid attention
    int linear_group_num = 0;  // Number of linear attention groups
    int swa_group_num    = 0;  // Number of sliding-window attention groups
    int full_group_num   = 0;  // Number of full attention groups

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    int groupNums() const {
        return std::max<int>(1, static_cast<int>(cache_specs.size()));
    }

    int groupIdForLayerTag(int layer_id, const std::string& tag) const {
        if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_tag_to_group_id.size()) {
            return -1;
        }
        const auto& tag_to_group = layer_tag_to_group_id[static_cast<size_t>(layer_id)];
        const auto  it           = tag_to_group.find(tag);
        return it == tag_to_group.end() ? -1 : it->second;
    }

    static std::string specFingerprint(const KVCacheSpecPtr& spec) {
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheConfig got null kv_cache spec");
        std::ostringstream os;
        os << "tag=" << spec->tag << ";type=" << static_cast<int>(spec->type) << ";dtype=" << static_cast<int>(spec->dtype)
           << ";local_head_num_kv=" << spec->local_head_num_kv
           << ";seq_size_per_block=" << spec->seq_size_per_block;
        if (const auto* mha = dynamic_cast<const MHAKVCacheSpec*>(spec.get())) {
            os << ";mha.size_per_head=" << mha->size_per_head;
        } else if (const auto* mla = dynamic_cast<const MLAKVCacheSpec*>(spec.get())) {
            os << ";mla.kv_lora_rank=" << mla->kv_lora_rank << ";mla.rope_head_dim=" << mla->rope_head_dim;
        } else if (const auto* linear = dynamic_cast<const LinearKVCacheSpec*>(spec.get())) {
            os << ";linear.local_num_k_heads=" << linear->local_num_k_heads
               << ";linear.local_num_v_heads=" << linear->local_num_v_heads
               << ";linear.head_k_dim=" << linear->head_k_dim
               << ";linear.head_v_dim=" << linear->head_v_dim
               << ";linear.conv_kernel_dim=" << linear->conv_kernel_dim
               << ";linear.ssm_state_dtype=" << static_cast<int>(linear->ssm_state_dtype)
               << ";linear.conv_state_dtype=" << static_cast<int>(linear->conv_state_dtype);
        } else if (const auto* dsv4_kv = dynamic_cast<const DSV4KVSpec*>(spec.get())) {
            os << ";dsv4kv.cache_type=" << static_cast<int>(dsv4_kv->cache_type)
               << ";dsv4kv.entry_elems=" << dsv4_kv->entry_elems
               << ";dsv4kv.compression_ratio=" << dsv4_kv->compression_ratio
               << ";dsv4kv.store_dtype=" << static_cast<int>(dsv4_kv->store_dtype)
               << ";dsv4kv.block_size_bytes_alignment=" << dsv4_kv->block_size_bytes_alignment;
        } else if (const auto* dsv4_state = dynamic_cast<const DSV4StateSpec*>(spec.get())) {
            os << ";dsv4state.cache_type=" << static_cast<int>(dsv4_state->cache_type)
               << ";dsv4state.state_dim=" << dsv4_state->state_dim
               << ";dsv4state.store_dtype=" << static_cast<int>(dsv4_state->store_dtype)
               << ";dsv4state.block_size_bytes_override=" << dsv4_state->block_size_bytes_override
               << ";dsv4state.block_size_bytes_alignment=" << dsv4_state->block_size_bytes_alignment
               << ";dsv4state.block_size_alignment_min_entries=" << dsv4_state->block_size_alignment_min_entries;
        }
        return os.str();
    }

    static CacheGroupType inferGroupType(const KVCacheSpecPtr& spec, KVCacheRegionName region_name) {
        if (region_name == KVCacheRegionName::SWA_KV || isStateRegion(region_name)) {
            return CacheGroupType::SWA;
        }
        return spec->type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
    }

    static KVCacheRegionName inferRegionName(const KVCacheSpecPtr& spec) {
        if (const auto* kv_spec = dynamic_cast<const DSV4KVSpec*>(spec.get())) {
            return kv_spec->cache_type;
        }
        if (const auto* state_spec = dynamic_cast<const DSV4StateSpec*>(spec.get())) {
            return state_spec->cache_type;
        }
        return KVCacheRegionName::DEFAULT;
    }

    void fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&             specs,
                          const std::vector<std::vector<int>>&          layers_by_group,
                          const std::vector<CacheGroupType>&            types,
                          const std::vector<KVCacheRegionName>&         regions = {},
                          const std::vector<std::string>&               tags    = {}) {
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
        RTP_LLM_CHECK_WITH_INFO(regions.empty() || regions.size() == group_num,
                                "CacheConfig::fromGroupedSpecs region count %zu != spec count %zu",
                                regions.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(tags.empty() || tags.size() == group_num,
                                "CacheConfig::fromGroupedSpecs tag count %zu != spec count %zu",
                                tags.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromGroupedSpecs requires positive layer_num");

        cache_specs.clear();
        global_layer_ids.clear();
        layer_ids.clear();
        group_types.clear();
        group_region_names.clear();
        group_tags.clear();

        cache_specs.reserve(group_num);
        global_layer_ids.reserve(group_num);
        layer_ids.reserve(group_num);
        group_types.reserve(group_num);
        group_region_names.reserve(group_num);
        group_tags.reserve(group_num);

        layer_to_group_id.assign(layer_num, -1);
        layer_to_group_ids.assign(layer_num, std::vector<int>());
        layer_group_types.assign(layer_num, CacheGroupType::FULL);

        const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
        layer_region_to_group_id.assign(layer_num, std::vector<int>(region_count, -1));
        layer_tag_to_group_id.assign(layer_num, std::map<std::string, int>());

        std::map<std::string, std::string> tag_fingerprints;
        for (size_t gid = 0; gid < group_num; ++gid) {
            const auto& spec = specs[gid];
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheConfig::fromGroupedSpecs got null spec at group %zu", gid);

            const auto region_name = regions.empty() ? inferRegionName(spec) : regions[gid];
            const auto region_id   = static_cast<size_t>(region_name);
            RTP_LLM_CHECK_WITH_INFO(region_id < region_count,
                                    "CacheConfig::fromGroupedSpecs invalid region id %zu for group %zu",
                                    region_id,
                                    gid);

            std::string tag = tags.empty() ? spec->tag : tags[gid];
            if (tag.empty() && group_num == 1) {
                tag = "default";
            }
            RTP_LLM_CHECK_WITH_INFO(!tag.empty(),
                                    "CacheConfig::fromGroupedSpecs requires non-empty tag for cache spec %zu",
                                    gid);
            const auto fingerprint = specFingerprint(spec);
            const auto fp_it       = tag_fingerprints.find(tag);
            if (fp_it == tag_fingerprints.end()) {
                tag_fingerprints.emplace(tag, fingerprint);
            } else {
                RTP_LLM_CHECK_WITH_INFO(fp_it->second == fingerprint,
                                        "CacheConfig::fromGroupedSpecs tag=%s has multiple physical prototypes",
                                        tag.c_str());
            }

            auto stored_spec = spec->clone();
            stored_spec->tag = tag;
            stored_spec->layers = layers_by_group[gid];

            cache_specs.push_back(stored_spec);
            global_layer_ids.push_back(layers_by_group[gid]);
            layer_ids.push_back(layers_by_group[gid]);
            group_types.push_back(types[gid]);
            group_region_names.push_back(region_name);
            group_tags.push_back(tag);

            std::vector<bool> seen_layer(static_cast<size_t>(layer_num), false);
            for (int layer_id : layers_by_group[gid]) {
                RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_num,
                                        "CacheConfig::fromGroupedSpecs tag=%s has invalid layer id %d for layer_num=%u",
                                        tag.c_str(),
                                        layer_id,
                                        layer_num);
                const auto layer = static_cast<size_t>(layer_id);
                RTP_LLM_CHECK_WITH_INFO(!seen_layer[layer],
                                        "CacheConfig::fromGroupedSpecs tag=%s has duplicate layer id %d",
                                        tag.c_str(),
                                        layer_id);
                seen_layer[layer] = true;

                layer_to_group_ids[layer].push_back(static_cast<int>(gid));
                const int current_region_gid = layer_region_to_group_id[layer][region_id];
                RTP_LLM_CHECK_WITH_INFO(current_region_gid < 0 || current_region_gid == static_cast<int>(gid),
                                        "CacheConfig::fromGroupedSpecs layer %d region %zu maps to both group %d and %zu",
                                        layer_id,
                                        region_id,
                                        current_region_gid,
                                        gid);
                layer_region_to_group_id[layer][region_id] = static_cast<int>(gid);
                layer_tag_to_group_id[layer][tag]          = static_cast<int>(gid);
                if (region_name == KVCacheRegionName::DEFAULT) {
                    layer_to_group_id[layer] = static_cast<int>(gid);
                    layer_group_types[layer] = types[gid];
                }
            }
        }

        const auto swa_region_id = static_cast<size_t>(KVCacheRegionName::SWA_KV);
        for (size_t layer = 0; layer < static_cast<size_t>(layer_num); ++layer) {
            if (layer_to_group_id[layer] < 0 && swa_region_id < layer_region_to_group_id[layer].size()) {
                layer_to_group_id[layer] = layer_region_to_group_id[layer][swa_region_id];
            }
            if (layer_to_group_id[layer] < 0 && !layer_to_group_ids[layer].empty()) {
                layer_to_group_id[layer] = layer_to_group_ids[layer].back();
            }
            RTP_LLM_CHECK_WITH_INFO(layer_to_group_id[layer] >= 0,
                                    "CacheConfig::fromGroupedSpecs missing group mapping for layer %zu",
                                    layer);
            const auto gid = static_cast<size_t>(layer_to_group_id[layer]);
            if (gid < group_types.size()) {
                layer_group_types[layer] = group_types[gid];
            }
        }
    }

    void fromLayerSpecs(const std::map<int64_t, std::vector<KVCacheSpecPtr>>& layer_specs) {
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromLayerSpecs requires positive layer_num");
        RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(layer_num),
                                "CacheConfig::fromLayerSpecs layer map size %zu != layer_num %u",
                                layer_specs.size(),
                                layer_num);

        std::vector<KVCacheSpecPtr> specs;
        std::vector<std::vector<int>> layers_by_group;
        std::vector<CacheGroupType> types;
        std::vector<KVCacheRegionName> regions;
        std::vector<std::string> tags;
        std::map<std::string, size_t> tag_to_group;
        std::map<std::string, std::string> tag_fingerprints;

        for (uint32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
            const auto layer_it = layer_specs.find(static_cast<int64_t>(layer_id));
            RTP_LLM_CHECK_WITH_INFO(layer_it != layer_specs.end(),
                                    "CacheConfig::fromLayerSpecs missing specs for layer %u",
                                    layer_id);
            RTP_LLM_CHECK_WITH_INFO(!layer_it->second.empty(),
                                    "CacheConfig::fromLayerSpecs layer %u has no specs",
                                    layer_id);
            std::map<std::string, bool> layer_seen_tags;
            for (const auto& spec : layer_it->second) {
                RTP_LLM_CHECK_WITH_INFO(spec != nullptr,
                                        "CacheConfig::fromLayerSpecs layer %u has null spec",
                                        layer_id);
                std::string tag = spec->tag;
                if (tag.empty() && layer_it->second.size() == 1) {
                    tag = "default";
                }
                RTP_LLM_CHECK_WITH_INFO(!tag.empty(),
                                        "CacheConfig::fromLayerSpecs layer %u has empty cache spec tag",
                                        layer_id);
                RTP_LLM_CHECK_WITH_INFO(layer_seen_tags.emplace(tag, true).second,
                                        "CacheConfig::fromLayerSpecs layer %u has duplicate tag=%s",
                                        layer_id,
                                        tag.c_str());

                const auto fingerprint = specFingerprint(spec);
                const auto fp_it = tag_fingerprints.find(tag);
                if (fp_it == tag_fingerprints.end()) {
                    tag_fingerprints.emplace(tag, fingerprint);
                } else {
                    RTP_LLM_CHECK_WITH_INFO(fp_it->second == fingerprint,
                                            "CacheConfig::fromLayerSpecs tag=%s has multiple physical prototypes",
                                            tag.c_str());
                }

                auto group_it = tag_to_group.find(tag);
                if (group_it == tag_to_group.end()) {
                    const size_t gid = specs.size();
                    tag_to_group.emplace(tag, gid);
                    specs.push_back(spec);
                    layers_by_group.emplace_back();
                    const auto region_name = inferRegionName(spec);
                    regions.push_back(region_name);
                    types.push_back(inferGroupType(spec, region_name));
                    tags.push_back(tag);
                    group_it = tag_to_group.find(tag);
                }
                layers_by_group[group_it->second].push_back(static_cast<int>(layer_id));
            }
        }

        fromGroupedSpecs(specs, layers_by_group, types, regions, tags);
    }

    void finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
        (void)runtime_config;
        if (!use_independent_block_pools || group_block_nums.empty()) {
            fixed_pool_reserve_bytes = 0;
            return;
        }

        const int step    = std::max(1, linear_step);
        size_t    reserve = 0;
        for (size_t gid = 0; gid < group_block_nums.size(); ++gid) {
            const bool is_swa = gid < group_types.size() && group_types[gid] == CacheGroupType::SWA;
            const auto region = gid < group_region_names.size() ? group_region_names[gid] : KVCacheRegionName::DEFAULT;
            const bool is_dsv4_fixed_region       = isDsv4FixedRegion(region);
            const bool use_explicit_hca_blocks    = region == KVCacheRegionName::HCA_STATE
                                                 && dsv4_hca_state_pool_blocks > 0;
            const bool use_explicit_fixed_blocks  = is_dsv4_fixed_region && dsv4_fixed_pool_blocks > 0;
            const bool use_explicit_dsv4_blocks   = use_explicit_hca_blocks || use_explicit_fixed_blocks;
            uint32_t   rule_blocks;
            if (use_explicit_hca_blocks) {
                rule_blocks = dsv4_hca_state_pool_blocks;
            } else if (use_explicit_fixed_blocks) {
                rule_blocks = dsv4_fixed_pool_blocks;
            } else if ((is_swa || is_dsv4_fixed_region) && step > 1 && global_block_num > 0) {
                rule_blocks = std::max(1u, global_block_num / static_cast<uint32_t>(step));
            } else {
                rule_blocks = global_block_num;
            }
            group_block_nums[gid] = rule_blocks;

            // Explicit DSV4 fixed pools are allocated outside the paged FULL
            // pool budget. The linear-step fallback is accounted by the
            // effective block-size formula instead, so no reserve is needed.
            const bool exclude_from_reserve = is_dsv4_fixed_region && fixed_pool_uses_pinned_cpu;
            if (use_explicit_dsv4_blocks && gid < group_block_size_bytes.size() && !exclude_from_reserve) {
                reserve += static_cast<size_t>(rule_blocks) * group_block_size_bytes[gid];
            }
        }
        fixed_pool_reserve_bytes = reserve;
    }

    std::string debugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";
        const std::string indent2    = indent_str + "    ";

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";

// Macro to simplify repeated field output and eliminate duplicate field names
#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"

// Helper macro for complex expressions
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

        // Model configuration section
        os << indent1 << "# Model Configuration:\n";
        OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
        OUTPUT_FIELD(layer_num);
        OUTPUT_FIELD(layer_all_num);
        OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
        os << "\n";

        // Block configuration section
        os << indent1 << "# Block Configuration:\n";
        OUTPUT_FIELD(block_num);
        OUTPUT_FIELD(seq_size_per_block);
        OUTPUT_FIELD(kernel_seq_size_per_block);
        os << "\n";

        // Block sizing information section
        os << indent1 << "# Block Sizing Information:\n";
        OUTPUT_FIELD(kv_block_size_bytes);
        OUTPUT_FIELD(kv_scale_size_bytes);
        OUTPUT_FIELD(block_size_bytes);
        OUTPUT_FIELD(swa_block_size_bytes);
        OUTPUT_FIELD(kv_block_stride_bytes);
        OUTPUT_FIELD(kv_scale_stride_bytes);
        os << "\n";

        // Attention-specific configuration section
        os << indent1 << "# Attention Configuration:\n";
        OUTPUT_FIELD(linear_step);
        OUTPUT_FIELD(linear_fixed_cap);
        OUTPUT_FIELD(group_layer_num);
        OUTPUT_FIELD(linear_group_num);
        OUTPUT_FIELD(swa_group_num);
        OUTPUT_FIELD(full_group_num);
        OUTPUT_FIELD(dsv4_fixed_pool_blocks);
        OUTPUT_FIELD(dsv4_hca_state_pool_blocks);
        OUTPUT_FIELD(use_independent_block_pools);
        OUTPUT_FIELD(use_typed_cache_regions);
        OUTPUT_FIELD(use_opaque_kv_cache_store);
        OUTPUT_FIELD(disable_decode_first_malloc_device_reuse);
        os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
        os << "\n";

        // Cache specification section
        os << indent1 << "# Cache Specifications:\n";
        OUTPUT_FIELD_EXPR("cache_specs.size()", cache_specs.size());
        for (size_t i = 0; i < cache_specs.size(); ++i) {
            const auto& spec = cache_specs[i];
            if (!spec) {
                os << indent1 << "cache_specs[" << i << "]=null\n";
                continue;
            }

            os << indent1 << "cache_specs[" << i << "] {\n";
            os << spec->debugString(indent + 2);
            os << indent1 << "}\n";
        }
        os << "\n";

        // Layer mapping section
        os << indent1 << "# Layer Mapping:\n";
        OUTPUT_FIELD_EXPR("global_layer_ids.size()", global_layer_ids.size());
        os << indent1 << "global_layer_ids=" << rtp_llm::vectorsToString(global_layer_ids) << "\n";
        OUTPUT_FIELD_EXPR("layer_ids.size()", layer_ids.size());
        os << indent1 << "layer_ids=" << rtp_llm::vectorsToString(layer_ids) << "\n";
        OUTPUT_FIELD_EXPR("group_types.size()", group_types.size());
        os << indent1 << "group_types=[";
        for (size_t i = 0; i < group_types.size(); ++i) {
            os << static_cast<int>(group_types[i]);
            if (i + 1 < group_types.size()) {
                os << ",";
            }
        }
        os << "]\n";
        OUTPUT_FIELD_EXPR("group_region_names.size()", group_region_names.size());
        os << indent1 << "group_region_names=[";
        for (size_t i = 0; i < group_region_names.size(); ++i) {
            os << static_cast<int>(group_region_names[i]);
            if (i + 1 < group_region_names.size()) {
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
        OUTPUT_FIELD_EXPR("layer_to_group_ids.size()", layer_to_group_ids.size());
        os << indent1 << "layer_to_group_ids=" << rtp_llm::vectorsToString(layer_to_group_ids) << "\n";
        OUTPUT_FIELD_EXPR("layer_group_types.size()", layer_group_types.size());
        os << indent1 << "layer_group_types=[";
        for (size_t i = 0; i < layer_group_types.size(); ++i) {
            os << static_cast<int>(layer_group_types[i]);
            if (i + 1 < layer_group_types.size()) {
                os << ",";
            }
        }
        os << "]\n";
        os << "\n";

        // mtp configurations section
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
};

}  // namespace rtp_llm
