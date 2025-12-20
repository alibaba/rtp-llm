#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    static BlockPoolConfig createLayerFirstConfig(uint32_t layer_num, uint32_t block_num, size_t block_stride_bytes) {
        BlockPoolConfig config;
        config.layer_num          = layer_num;
        config.block_num          = block_num;
        config.block_stride_bytes = block_stride_bytes;
        config.layout             = LAYER_FIRST;

        config.total_size_bytes = static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * block_stride_bytes;
        return config;
    }

    /**
     * Create a Layer-First layout configuration.
     * Memory layout: [layer_num, num_blocks, block_stride]
     *
     * @param layer_num Number of layers
     * @param block_num Number of blocks
     * @param spec KVCacheSpec
     */
    static BlockPoolConfig
    createLayerFirstConfig(uint32_t layer_num, uint32_t block_num, const std::shared_ptr<KVCacheSpec>& spec) {
        BlockPoolConfig config;

        const size_t block_stride         = spec->block_size();
        const size_t k_block_stride       = spec->k_block_size();
        const size_t v_block_stride       = spec->v_block_size();
        const size_t block_stride_bytes   = spec->block_size_bytes();
        const size_t k_block_stride_bytes = spec->k_block_size_bytes();
        const size_t v_block_stride_bytes = spec->v_block_size_bytes();
        const size_t local_head_num_kv    = spec->local_head_num_kv;
        const size_t seq_size_per_block   = spec->seq_size_per_block;
        const size_t k_token_size         = spec->k_token_size();
        const size_t v_token_size         = spec->v_token_size();

        config.layer_num        = layer_num;
        config.block_num        = block_num;
        config.layout           = LAYER_FIRST;
        config.total_size       = static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * block_stride;
        config.total_size_bytes = static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * block_stride_bytes;

        config.seq_size_per_block = seq_size_per_block;
        config.local_head_num_kv  = local_head_num_kv;
        config.is_mla             = spec->type == KVCacheType::MultiHeadLatentAttention;
        config.dtype              = spec->dtype;
        config.k_token_size       = k_token_size;
        config.v_token_size       = v_token_size;

        config.block_stride   = block_stride;
        config.k_block_stride = k_block_stride;
        config.v_block_stride = v_block_stride;

        config.block_stride_bytes   = block_stride_bytes;
        config.k_block_stride_bytes = k_block_stride_bytes;
        config.v_block_stride_bytes = v_block_stride_bytes;

        config.block_size   = block_stride * static_cast<size_t>(layer_num);
        config.k_block_size = k_block_stride * static_cast<size_t>(layer_num);
        config.v_block_size = v_block_stride * static_cast<size_t>(layer_num);

        config.block_size_bytes   = block_stride_bytes * static_cast<size_t>(layer_num);
        config.k_block_size_bytes = k_block_stride_bytes * static_cast<size_t>(layer_num);
        config.v_block_size_bytes = v_block_stride_bytes * static_cast<size_t>(layer_num);

        return config;
    }
};

}  // namespace rtp_llm
