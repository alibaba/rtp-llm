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
