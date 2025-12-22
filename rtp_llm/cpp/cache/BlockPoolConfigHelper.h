#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    static inline bool needKvScale(rtp_llm::DataType dtype) {
        return dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3;
    }

    static BlockPoolConfig createLayerFirstConfig(uint32_t layer_num, uint32_t block_num, size_t block_stride_bytes) {
        BlockPoolConfig config;
        config.layer_num             = layer_num;
        config.block_num             = block_num;
        config.kv_block_stride_bytes = block_stride_bytes;
        config.layout                = LAYER_FIRST;

        config.block_stride_bytes = config.kv_block_stride_bytes;
        config.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * config.kv_block_stride_bytes;
        config.kv_scale_offset_bytes = config.kv_block_pool_size_bytes;
        config.total_size_bytes      = config.kv_block_pool_size_bytes;
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

        const size_t kv_block_stride       = spec->block_size();
        const size_t k_block_stride        = spec->k_block_size();
        const size_t v_block_stride        = spec->v_block_size();
        const size_t kv_block_stride_bytes = spec->block_size_bytes();
        const size_t k_block_stride_bytes  = spec->k_block_size_bytes();
        const size_t v_block_stride_bytes  = spec->v_block_size_bytes();
        const size_t local_head_num_kv     = spec->local_head_num_kv;
        const size_t seq_size_per_block    = spec->seq_size_per_block;
        const size_t k_token_size          = spec->k_token_size();
        const size_t v_token_size          = spec->v_token_size();

        config.layer_num             = layer_num;
        config.block_num             = block_num;
        config.layout                = LAYER_FIRST;
        config.kv_block_stride       = kv_block_stride;
        config.kv_block_stride_bytes = kv_block_stride_bytes;

        config.seq_size_per_block = seq_size_per_block;
        config.local_head_num_kv  = local_head_num_kv;
        config.is_mla             = spec->type == KVCacheType::MultiHeadLatentAttention;
        config.dtype              = spec->dtype;
        config.k_token_size       = k_token_size;
        config.v_token_size       = v_token_size;

        config.k_block_stride = k_block_stride;
        config.v_block_stride = v_block_stride;

        config.k_block_stride_bytes = k_block_stride_bytes;
        config.v_block_stride_bytes = v_block_stride_bytes;

        config.kv_block_size = kv_block_stride * static_cast<size_t>(layer_num);
        config.k_block_size  = k_block_stride * static_cast<size_t>(layer_num);
        config.v_block_size  = v_block_stride * static_cast<size_t>(layer_num);

        config.kv_block_size_bytes = kv_block_stride_bytes * static_cast<size_t>(layer_num);
        config.k_block_size_bytes  = k_block_stride_bytes * static_cast<size_t>(layer_num);
        config.v_block_size_bytes  = v_block_stride_bytes * static_cast<size_t>(layer_num);

        // Optional kv scale pool appended after kv pool.
        if (needKvScale(spec->dtype)) {
            config.enable_kv_scale      = true;
            config.kv_scale_block_bytes = local_head_num_kv * seq_size_per_block * sizeof(float);
            // scale pool stores K/V scales as separate blocks indexed by (block_id * 2 + kv_idx)
            config.kv_scale_stride_bytes = 2 * config.kv_scale_block_bytes;
            config.kv_scale_stride       = 2 * local_head_num_kv * seq_size_per_block;
            config.kv_scale_pool_size_bytes =
                static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * config.kv_scale_stride_bytes;
            config.kv_scale_size_bytes = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
            config.kv_scale_size       = static_cast<size_t>(layer_num) * config.kv_scale_stride;
        } else {
            config.kv_scale_block_bytes     = 0;
            config.kv_scale_pool_size_bytes = 0;
            config.kv_scale_stride          = 0;
            config.kv_scale_stride_bytes    = 0;
            config.kv_scale_size            = 0;
            config.kv_scale_size_bytes      = 0;
        }

        config.block_stride       = config.kv_block_stride + config.kv_scale_stride;
        config.block_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
        config.block_size         = config.kv_block_size + config.kv_scale_size;
        config.block_size_bytes   = config.kv_block_size_bytes + config.kv_scale_size_bytes;

        config.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * config.kv_block_stride_bytes;
        // Align with legacy allocator behavior: scale pool starts right after kv pool.
        config.kv_scale_offset_bytes = config.kv_block_pool_size_bytes;
        config.total_size_bytes      = config.kv_block_pool_size_bytes + config.kv_scale_pool_size_bytes;
        config.total_size            = config.total_size_bytes;

        return config;
    }
};

}  // namespace rtp_llm
