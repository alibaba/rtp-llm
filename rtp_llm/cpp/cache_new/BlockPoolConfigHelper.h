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

    /**
     * Create a KV-First layout configuration (only for full-attention-only scenarios).
     * Memory layout: [2, layer_num, num_blocks, kv_block_size]
     * where 2 represents the contiguous K region and the contiguous V region.
     *
     * @param layer_num Number of layers
     * @param block_num Number of blocks
     * @param kv_block_size Size of each K or V block
     */
    static BlockPoolConfig createKVFirstConfig(uint32_t layer_num, uint32_t block_num, uint32_t kv_block_size) {

        BlockPoolConfig config;
        config.layer_num = layer_num;
        config.block_num = block_num;
        config.layout    = KV_FIRST;

        config.k_block_size = kv_block_size;
        config.v_block_size = kv_block_size;
        config.block_size   = kv_block_size * 2;

        // 2 (K+V) * layer_num * block_num * kv_block_size
        config.total_size = 2 * static_cast<size_t>(layer_num) * block_num * kv_block_size;

        config.k_block_stride = kv_block_size;
        config.v_block_stride = kv_block_size;
        config.k_layer_stride = static_cast<size_t>(block_num) * kv_block_size;
        config.v_layer_stride = static_cast<size_t>(block_num) * kv_block_size;

        RTP_LLM_LOG_INFO("create KVFirstConfig: layer_num=%d, block_num=%d, kv_block_size=%d, total_size=%zu",
                         layer_num,
                         block_num,
                         kv_block_size,
                         config.total_size);

        return config;
    }

    /**
     * Create a KV-First layout configuration (with logical KV shape metadata).
     * The shape metadata is used by downstream operators for shape validation:
     * [layer_block, kv_head_num, tokens_per_block, size_per_head]
     */
    static BlockPoolConfig createKVFirstConfig(uint32_t          layer_num,
                                               uint32_t          block_num,
                                               uint32_t          kv_block_size,
                                               uint32_t          kv_head_num,
                                               uint32_t          tokens_per_block,
                                               uint32_t          size_per_head,
                                               rtp_llm::DataType dtype) {

        BlockPoolConfig config = createKVFirstConfig(layer_num, block_num, kv_block_size);
        // Set logical shape metadata
        config.kv_head_num      = kv_head_num;
        config.tokens_per_block = tokens_per_block;
        config.size_per_head    = size_per_head;
        config.dtype            = dtype;
        return config;
    }

    /**
     * Derive a KV-First configuration from KVCacheSpec (for adaption, should be removed in future).
     * - Automatically compute kv_block_size = spec->k_block_size()
     * - kv_head_num: the shared local_head_num_kv for both MHA and MLA
     * - tokens_per_block: spec->seq_size_per_block
     * - size_per_head: use size_per_head for MHA; use rope_head_dim for MLA
     */
    static BlockPoolConfig
    createKVFirstConfig(uint32_t layer_num, uint32_t block_num, const std::shared_ptr<KVCacheSpec>& spec) {
        const uint32_t kv_block_size = static_cast<uint32_t>(spec->k_block_size());
        const uint32_t kv_head_num =
            static_cast<uint32_t>(spec->type == KVCacheType::MultiHeadAttention ?
                                      std::dynamic_pointer_cast<MHAKVCacheSpec>(spec)->local_head_num_kv :
                                      std::dynamic_pointer_cast<MLAKVCacheSpec>(spec)->local_head_num_kv);
        const uint32_t tokens_per_block = static_cast<uint32_t>(spec->seq_size_per_block);
        uint32_t       size_per_head    = 0;
        if (spec->type == KVCacheType::MultiHeadAttention) {
            size_per_head = static_cast<uint32_t>(std::dynamic_pointer_cast<MHAKVCacheSpec>(spec)->size_per_head);
        } else if (spec->type == KVCacheType::MultiHeadLatentAttention) {
            size_per_head = static_cast<uint32_t>(std::dynamic_pointer_cast<MLAKVCacheSpec>(spec)->rope_head_dim);
        } else {
            size_per_head = 0;
        }
        return createKVFirstConfig(
            layer_num, block_num, kv_block_size, kv_head_num, tokens_per_block, size_per_head, spec->dtype);
    }
};

}  // namespace rtp_llm
