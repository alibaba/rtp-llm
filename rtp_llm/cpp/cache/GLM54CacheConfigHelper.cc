#include "rtp_llm/cpp/cache/GLM54CacheConfigHelper.h"

#include <algorithm>
#include <unordered_set>

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

uint32_t evenCeil(uint32_t value) {
    return (value + 1u) & ~1u;
}

std::vector<int> resolveIndexerLayers(const ModelConfig& model_config) {
    const auto& attn = model_config.attn_config;
    std::vector<int> layers;

    if (!attn.indexer_layer_ids.empty()) {
        layers = attn.indexer_layer_ids;
    } else {
        layers.reserve(static_cast<size_t>(model_config.num_layers));
        const auto& hybrid = model_config.hybrid_attention_config;
        for (int layer = 0; layer < model_config.num_layers; ++layer) {
            const bool is_linear =
                hybrid.enable_hybrid_attention
                && static_cast<size_t>(layer) < hybrid.hybrid_attention_types.size()
                && hybrid.hybrid_attention_types[static_cast<size_t>(layer)] == HybridAttentionType::LINEAR;
            if (!is_linear) {
                layers.push_back(layer);
            }
        }
    }

    std::unordered_set<int> seen;
    for (int layer : layers) {
        RTP_LLM_CHECK_WITH_INFO(layer >= 0 && layer < model_config.num_layers,
                                "GLM5.4 indexer layer id %d is outside [0, %ld)",
                                layer,
                                model_config.num_layers);
        RTP_LLM_CHECK_WITH_INFO(seen.insert(layer).second, "GLM5.4 duplicate indexer layer id %d", layer);
        const auto& hybrid = model_config.hybrid_attention_config;
        if (hybrid.enable_hybrid_attention) {
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(layer) < hybrid.hybrid_attention_types.size()
                    && hybrid.hybrid_attention_types[static_cast<size_t>(layer)] != HybridAttentionType::LINEAR,
                "GLM5.4 indexer layer %d must be an MLA/full-attention layer, not KDA/LINEAR",
                layer);
        }
    }
    return layers;
}

void appendPool(CacheConfig&                  config,
                const std::vector<int>&       layer_ids,
                CacheGroupType                group_type,
                KVCacheRegionName             region_name,
                const KVCacheSpecPtr&         spec,
                uint32_t                      tokens_per_block) {
    config.cache_specs.push_back(spec);
    config.global_layer_ids.push_back(layer_ids);
    config.layer_ids.push_back(layer_ids);
    config.group_types.push_back(group_type);
    config.group_region_names.push_back(region_name);
    config.group_seq_size_per_block.push_back(tokens_per_block);
}

}  // namespace

void GLM54CacheConfigHelper::appendIndexerPools(CacheConfig&             config,
                                                 const ModelConfig&       model_config,
                                                 const ParallelismConfig& parallelism_config,
                                                 const KVCacheConfig&     kv_cache_config,
                                                 int                      gen_num_per_cycle) {
    (void)parallelism_config;
    const auto& attn    = model_config.attn_config;
    const int   ratio   = attn.indexer_compress_ratio;
    const int   overlap = attn.indexer_compressor_overlap;

    RTP_LLM_CHECK_WITH_INFO(attn.is_sparse, "GLM5.4 compressed indexer requires sparse attention");
    RTP_LLM_CHECK_WITH_INFO(attn.use_mla, "GLM5.4 compressed indexer requires MLA on full-attention layers");
    RTP_LLM_CHECK_WITH_INFO(ratio == 4,
                            "GLM5.4 first implementation supports indexer_compress_ratio=4, got %d",
                            ratio);
    RTP_LLM_CHECK_WITH_INFO(overlap == 0 || overlap == 1,
                            "GLM5.4 indexer_compressor_overlap must be 0 or 1, got %d",
                            overlap);
    RTP_LLM_CHECK_WITH_INFO(attn.indexer_head_dim > 0,
                            "GLM5.4 indexer_head_dim must be positive, got %d",
                            attn.indexer_head_dim);
    RTP_LLM_CHECK_WITH_INFO(attn.indexer_topk > 0,
                            "GLM5.4 indexer_topk must be positive, got %d",
                            attn.indexer_topk);
    const int expected_attention_topk = attn.indexer_topk * ratio;
    RTP_LLM_CHECK_WITH_INFO(attn.sparse_attention_topk == 0
                                || attn.sparse_attention_topk == expected_attention_topk,
                            "GLM5.4 sparse_attention_topk must be indexer_topk * ratio (%d), got %d",
                            expected_attention_topk,
                            attn.sparse_attention_topk);
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "GLM5.4 gen_num_per_cycle must be non-negative, got %d",
                            gen_num_per_cycle);

    const uint32_t physical_tokens_per_block =
        kv_cache_config.seq_size_per_block > 0 ? static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                                 static_cast<uint32_t>(attn.tokens_per_block);
    const uint32_t kernel_tokens_per_block =
        kv_cache_config.kernel_seq_size_per_block > 0 ?
            static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
        attn.kernel_tokens_per_block > 0 ? static_cast<uint32_t>(attn.kernel_tokens_per_block) :
                                          physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0 && kernel_tokens_per_block > 0,
                            "GLM5.4 cache block sizes must be positive (physical=%u kernel=%u)",
                            physical_tokens_per_block,
                            kernel_tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block % kernel_tokens_per_block == 0,
                            "GLM5.4 physical tokens per block %u must be divisible by kernel tokens per block %u",
                            physical_tokens_per_block,
                            kernel_tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % static_cast<uint32_t>(ratio) == 0,
                            "GLM5.4 kernel tokens per block %u must be divisible by ratio %d",
                            kernel_tokens_per_block,
                            ratio);

    const auto indexer_layers = resolveIndexerLayers(model_config);
    RTP_LLM_CHECK_WITH_INFO(!indexer_layers.empty(), "GLM5.4 compressed indexer has no owning MLA layers");

    const bool fp8_cache = attn.kv_cache_dtype == KvCacheDataType::FP8;
    RTP_LLM_CHECK_WITH_INFO(fp8_cache || attn.kv_cache_dtype == KvCacheDataType::BASE,
                            "GLM5.4 compressed indexer supports BASE or FP8 cache only");
    const uint32_t indexer_entry_bytes =
        fp8_cache ? DSV4_FP8_INDEXER_ENTRY_BYTES : static_cast<uint32_t>(attn.indexer_head_dim) * 2u;
    const uint32_t indexer_entries_per_kernel_block = kernel_tokens_per_block / static_cast<uint32_t>(ratio);
    const uint32_t state_entries = evenCeil(
        static_cast<uint32_t>((1 + overlap) * ratio + gen_num_per_cycle));
    // Each raw state entry stores both projected KV and score+APE. The
    // projection has (1+overlap) branches, matching DSV4's CSA compressor.
    const uint32_t state_elements =
        2u * static_cast<uint32_t>(1 + overlap) * static_cast<uint32_t>(attn.indexer_head_dim);

    auto indexer_kv_spec = std::make_shared<DSV4KVSpec>(KVCacheRegionName::INDEXER_KV,
                                                        static_cast<uint32_t>(indexer_layers.size()),
                                                        indexer_entry_bytes,
                                                        indexer_entries_per_kernel_block,
                                                        DataType::TYPE_UINT8,
                                                        physical_tokens_per_block);
    auto indexer_state_spec = std::make_shared<DSV4StateSpec>(KVCacheRegionName::INDEXER_STATE,
                                                              static_cast<uint32_t>(indexer_layers.size()),
                                                              state_elements,
                                                              state_entries,
                                                              DataType::TYPE_FP32,
                                                              physical_tokens_per_block);

    // The base MLA/KDA groups were created before this helper and do not
    // carry per-group row sizes yet. Seed them before appending typed regions
    // so group ids and row-size entries stay aligned.
    config.group_seq_size_per_block.resize(config.cache_specs.size(), physical_tokens_per_block);

    appendPool(config,
               indexer_layers,
               CacheGroupType::FULL,
               KVCacheRegionName::INDEXER_KV,
               indexer_kv_spec,
               physical_tokens_per_block);
    appendPool(config,
               indexer_layers,
               CacheGroupType::SWA,
               KVCacheRegionName::INDEXER_STATE,
               indexer_state_spec,
               physical_tokens_per_block);

    config.seq_size_per_block                       = physical_tokens_per_block;
    config.kernel_seq_size_per_block                = kernel_tokens_per_block;
    config.use_typed_cache_regions                  = true;
    config.use_opaque_kv_cache_store                = true;
    config.disable_decode_first_malloc_device_reuse = true;
    config.dsv4_fixed_pool_blocks                   = kv_cache_config.dsv4_fixed_pool_blocks;

    RTP_LLM_LOG_INFO("GLM5.4 compressed indexer cache: layers=%zu ratio=%d overlap=%d selection_topk=%d "
                     "attention_topk=%d physical_tpb=%u kernel_tpb=%u indexer_entries=%u state_entries=%u",
                     indexer_layers.size(),
                     ratio,
                     overlap,
                     attn.indexer_topk,
                     attn.sparse_attention_topk,
                     physical_tokens_per_block,
                     kernel_tokens_per_block,
                     indexer_entries_per_kernel_block,
                     state_entries);
}

}  // namespace rtp_llm
