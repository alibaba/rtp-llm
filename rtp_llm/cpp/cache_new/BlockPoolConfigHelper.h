#pragma once

#include "rtp_llm/cpp/cache_new/CacheConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    /**
     * Create a Layer-First layout configuration.
     * Memory layout: [layer_num, num_blocks, block_size]
     *
     * @param layer_num Number of layers
     * @param block_num Number of blocks
     * @param block_size Size of each block (not necessarily the K/V block size)
     */
    static BlockPoolConfig createLayerFirstConfig(uint32_t layer_num, uint32_t block_num, uint32_t block_size) {

        BlockPoolConfig config;
        config.layer_num  = layer_num;
        config.block_num  = block_num;
        config.block_size = block_size;
        config.layout     = LAYER_FIRST;
        config.total_size = static_cast<size_t>(layer_num) * block_num * block_size;

        // For generic layer-first configs without KVCacheSpec, assume an even K/V split.
        config.k_block_size = block_size / 2;
        config.v_block_size = block_size - config.k_block_size;

        return config;
    }

    /**
     * Create a KV-First layout configuration.
     * Memory layout: [2, layer_num, num_blocks, kv_block_size]
     * where 2 represents the contiguous K region and the contiguous V region.
     *
     * @param layer_num Number of layers
     * @param block_num Number of blocks
     * @param k_block_size Size of each K block (per layer)
     * @param v_block_size Size of each V block (per layer)
     */
    static BlockPoolConfig
    createKVFirstConfig(uint32_t layer_num, uint32_t block_num, uint32_t k_block_size, uint32_t v_block_size) {

        BlockPoolConfig config;
        config.layer_num = layer_num;
        config.block_num = block_num;
        config.layout    = KV_FIRST;

        config.k_block_size = k_block_size;
        config.v_block_size = v_block_size;
        config.block_size   = k_block_size + v_block_size;

        // layer_num * block_num * (k_block_size + v_block_size)
        config.total_size = static_cast<size_t>(layer_num) * block_num * (k_block_size + v_block_size);

        config.k_block_stride = k_block_size;
        config.v_block_stride = v_block_size;

        RTP_LLM_LOG_INFO(
            "create KVFirstConfig: layer_num=%d, block_num=%d, k_block_size=%d, v_block_size=%d, total_size=%zu",
            layer_num,
            block_num,
            k_block_size,
            v_block_size,
            config.total_size);

        return config;
    }

    /**
     * Derive a KV-First configuration from KVCacheSpec (for adaption, should be removed in future).
     * - Automatically compute k_block_size and v_block_size from spec
     * - local_head_num_kv: the shared local_head_num_kv for both MHA and MLA
     * - seq_size_per_block: spec->seq_size_per_block
     * - size_per_head: use size_per_head for MHA; use rope_head_dim for MLA
     */
    static BlockPoolConfig
    createKVFirstConfig(uint32_t layer_num, uint32_t block_num, const std::shared_ptr<KVCacheSpec>& spec) {
        const uint32_t  k_block_size       = static_cast<uint32_t>(spec->k_block_size());
        const uint32_t  v_block_size       = static_cast<uint32_t>(spec->v_block_size());
        const uint32_t  local_head_num_kv  = static_cast<uint32_t>(spec->local_head_num_kv);
        const uint32_t  seq_size_per_block = static_cast<uint32_t>(spec->seq_size_per_block);
        BlockPoolConfig config             = createKVFirstConfig(layer_num, block_num, k_block_size, v_block_size);
        config.dtype                       = spec->dtype;
        config.seq_size_per_block          = seq_size_per_block;
        config.local_head_num_kv           = local_head_num_kv;
        config.is_mla                      = spec->type == KVCacheType::MultiHeadLatentAttention;
        config.k_token_size                = static_cast<uint32_t>(spec->k_token_size());
        config.v_token_size                = static_cast<uint32_t>(spec->v_token_size());
        return config;
    }

    static BlockPoolConfig
    createLayerFirstConfig(uint32_t layer_num, uint32_t block_num, const std::shared_ptr<KVCacheSpec>& spec) {
        const uint32_t k_block_size       = static_cast<uint32_t>(spec->k_block_size());
        const uint32_t v_block_size       = static_cast<uint32_t>(spec->v_block_size());
        const uint32_t local_head_num_kv  = static_cast<uint32_t>(spec->local_head_num_kv);
        const uint32_t seq_size_per_block = static_cast<uint32_t>(spec->seq_size_per_block);
        const uint32_t k_token_size       = static_cast<uint32_t>(spec->k_token_size());
        const uint32_t v_token_size       = static_cast<uint32_t>(spec->v_token_size());
        const uint32_t block_size         = static_cast<uint32_t>(spec->block_size());

        BlockPoolConfig config    = createLayerFirstConfig(layer_num, block_num, block_size);
        config.seq_size_per_block = seq_size_per_block;
        config.local_head_num_kv  = local_head_num_kv;
        config.is_mla             = spec->type == KVCacheType::MultiHeadLatentAttention;
        config.dtype              = spec->dtype;
        config.k_token_size       = k_token_size;
        config.v_token_size       = v_token_size;
        config.k_block_size       = k_block_size;
        config.v_block_size       = v_block_size;

        return config;
    }
};

}  // namespace rtp_llm
