#include "rtp_llm/cpp/cache/DSV41CacheConfigHelper.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"

namespace rtp_llm {
namespace {

uint32_t contextParallelSize(const ParallelismConfig& parallelism) {
    if (!parallelism.prefill_cp_config.kv_cache_sharded) {
        RTP_LLM_CHECK_WITH_INFO(!parallelism.prefill_cp_config.is_prefill_enabled(),
                                "V4.1 CP execution requires CPRR cache sharding");
        return 1;
    }
    const auto cp = parallelism.role_type == RoleType::PREFILL ? parallelism.tp_size :
                                                                 parallelism.prefill_cp_config.prefill_cp_size;
    RTP_LLM_CHECK_WITH_INFO(cp == 8, "V4.1 sharded cache requires explicit CP8, got %ld", cp);
    return static_cast<uint32_t>(cp);
}

int globalOwner(int layer) {
    if (layer < 2 || layer >= 40) {
        return -1;
    }
    return layer < 20 ? 2 + ((layer - 2) / 6) * 6 : 20;
}

}  // namespace

void DSV41CacheConfigHelper::applyConfig(CacheConfig&             config,
                                         const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism,
                                         const KVCacheConfig&     kv_config,
                                         bool                     is_draft,
                                         int                      gen_num_per_cycle) {
    const auto& attn = model_config.attn_config;
    RTP_LLM_CHECK_WITH_INFO(attn.dsv41_cache_layout_version == 1,
                            "unsupported V4.1 cache layout version %d",
                            attn.dsv41_cache_layout_version);
    const int layers = is_draft ? 3 : 40;
    RTP_LLM_CHECK_WITH_INFO(model_config.num_layers == layers
                                && attn.layer_compress_ratios.size() == static_cast<size_t>(layers),
                            "V4.1 cache requires %d %s layers",
                            layers,
                            is_draft ? "draft" : "target");
    RTP_LLM_CHECK_WITH_INFO(attn.size_per_head == 512 && attn.kv_head_num == 1 && attn.sliding_window == 128,
                            "V4.1 cache requires head512/MQA/window128");
    RTP_LLM_CHECK_WITH_INFO(is_draft || (attn.indexer_head_dim == 128 && attn.indexer_head_num == 32),
                            "V4.1 indexer requires 32 heads with dimension128");
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle == 0 || gen_num_per_cycle == 5,
                            "V4.1 speculative cache requires gamma5 or the non-spec diagnostic mode");
    RTP_LLM_CHECK_WITH_INFO(!kv_config.enable_memory_cache_disk, "V4.1 does not support disk KV offload");
    RTP_LLM_CHECK_WITH_INFO(!kv_config.enable_dsv4_state_block_independent_eviction,
                            "V4.1 requires joint SWA/global checkpoint eviction");
    RTP_LLM_CHECK_WITH_INFO(!kv_config.dsv4_fixed_pool_use_memory,
                            "V4.1 active SWA/pair backing must be on GPU; CPU offload uses the memory connector");
    for (int layer = 0; layer < layers; ++layer) {
        const int expected = is_draft || layer < 2 ? 0 : (layer < 20 ? 2 : 1);
        RTP_LLM_CHECK_WITH_INFO(attn.layer_compress_ratios[static_cast<size_t>(layer)] == expected,
                                "V4.1 layer %d requires compression ratio %d",
                                layer,
                                expected);
    }
    const uint32_t block  = kv_config.seq_size_per_block > 0 ? kv_config.seq_size_per_block : 128;
    const uint32_t kernel = kv_config.kernel_seq_size_per_block > 0 ? kv_config.kernel_seq_size_per_block : block;
    RTP_LLM_CHECK_WITH_INFO((block == 128 || block == 256) && kernel == block,
                            "V4.1 compact cache requires identical physical/kernel blocks of 128 or 256");
    const uint32_t         cp            = contextParallelSize(parallelism);
    const bool             byte_slice    = cp > 1 && parallelism.role_type == RoleType::PREFILL;
    const uint32_t         alignment     = std::lcm(2u, cp);
    const uint32_t         swa_entries   = ((128u + gen_num_per_cycle + alignment - 1) / alignment) * alignment;
    const uint32_t         snapshots     = gen_num_per_cycle + 2;
    const std::vector<int> pair_owners   = is_draft ? std::vector<int>{} : std::vector<int>{2, 8, 14};
    const std::vector<int> ratio1_owners = is_draft ? std::vector<int>{} : std::vector<int>{20};
    std::vector<int>       all_layers(layers);
    std::iota(all_layers.begin(), all_layers.end(), 0);

    config.dsv41_cache_layout_version = 1;
    config.dsv41_draft_cache          = is_draft;
    config.layer_num = config.layer_all_num         = layers;
    config.use_mla                                  = false;
    config.is_sparse                                = true;
    config.seq_size_per_block                       = block;
    config.kernel_seq_size_per_block                = block;
    config.use_typed_cache_regions                  = true;
    config.use_opaque_kv_cache_store                = true;
    config.disable_decode_first_malloc_device_reuse = true;
    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();
    config.group_seq_size_per_block.clear();
    auto add = [&](KVCacheRegionName       region,
                   const std::vector<int>& owners,
                   uint32_t                ratio,
                   uint32_t                entries,
                   CacheGroupType          group_type) {
        const bool     fixed  = group_type == CacheGroupType::SWA;
        const uint32_t tokens = fixed ? block * cp : block;
        config.cache_specs.push_back(
            std::make_shared<DSV41KVCacheSpec>(region, owners.size(), ratio, entries, tokens, cp, fixed && byte_slice));
        config.global_layer_ids.push_back(owners);
        config.layer_ids.push_back(owners);
        config.group_types.push_back(group_type);
        config.group_region_names.push_back(region);
        config.group_seq_size_per_block.push_back(tokens);
    };
    add(KVCacheRegionName::DSV41_GLOBAL_KV, pair_owners, 2, block / 2, CacheGroupType::FULL);
    add(KVCacheRegionName::DSV41_GLOBAL_KV, ratio1_owners, 1, block, CacheGroupType::FULL);
    add(KVCacheRegionName::DSV41_INDEX_KV, pair_owners, 2, block / 2, CacheGroupType::FULL);
    add(KVCacheRegionName::DSV41_INDEX_KV, ratio1_owners, 1, block, CacheGroupType::FULL);
    add(KVCacheRegionName::DSV41_PAIR_STATE, pair_owners, 2, snapshots, CacheGroupType::SWA);
    add(KVCacheRegionName::SWA_KV, all_layers, 0, swa_entries, CacheGroupType::SWA);
}

void DSV41CacheConfigHelper::populateOwnerMappings(CacheConfig& config) {
    if (config.dsv41_cache_layout_version == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(config.dsv41_cache_layout_version == 1, "unsupported V4.1 owner schema");
    const size_t count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_owner.assign(config.layer_all_num, std::vector<int>(count, -1));
    config.dsv41_topk_owner.assign(config.layer_all_num, -1);
    for (size_t layer = 0; layer < config.layer_all_num; ++layer) {
        for (size_t region = 0; region < count; ++region) {
            if (config.layer_region_to_group_id[layer][region] >= 0) {
                config.layer_region_to_owner[layer][region] = layer;
            }
        }
        const int owner = config.dsv41_draft_cache ? -1 : globalOwner(layer);
        if (owner < 0) {
            continue;
        }
        config.dsv41_topk_owner[layer] = layer < 20 ? owner : 20 + ((static_cast<int>(layer) - 20) / 4) * 4;
        for (auto region : {KVCacheRegionName::DSV41_GLOBAL_KV, KVCacheRegionName::DSV41_INDEX_KV}) {
            const auto rid = static_cast<size_t>(region);
            const int  gid = config.layer_region_to_group_id[static_cast<size_t>(owner)][rid];
            RTP_LLM_CHECK_WITH_INFO(gid >= 0, "V4.1 source layer %d has no physical group", owner);
            config.layer_region_to_group_id[layer][rid] = gid;
            config.layer_region_to_owner[layer][rid]    = owner;
            auto& groups                                = config.layer_to_group_ids[layer];
            if (std::find(groups.begin(), groups.end(), gid) == groups.end()) {
                groups.push_back(gid);
            }
        }
    }
}

}  // namespace rtp_llm
