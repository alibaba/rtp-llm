#pragma once

#include "rtp_llm/cpp/cache_new/CacheConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    /**
     * 创建 Layer-First 布局配置
     * 内存布局：[layer_num, num_blocks, block_size]
     * 
     * @param layer_num 层数
     * @param block_num 块数量
     * @param block_size 每个块的大小（不一定是 K 和 V 的块大小）
     */
    static BlockPoolConfig createLayerFirstConfig(
        uint32_t layer_num,
        uint32_t block_num, 
        uint32_t block_size) {
        
        BlockPoolConfig config;
        config.layer_num = layer_num;
        config.block_num = block_num;
        config.block_size = block_size;
        config.layout = LAYER_FIRST;
        config.total_size = static_cast<size_t>(layer_num) * block_num * block_size;
        
        return config;
    }
    
    /**
     * 创建 KV-First 布局配置（仅适用于 full attention only 场景）
     * 内存布局：[2, layer_num, num_blocks, kv_block_size]
     * 其中 2 代表 K 连续缓存区和 V 连续缓存区
     * 
     * @param layer_num 层数
     * @param block_num 块数量
     * @param kv_block_size 每个 K 或 V 块的大小
     */
    static BlockPoolConfig createKVFirstConfig(
        uint32_t layer_num,
        uint32_t block_num,
        uint32_t kv_block_size) {
        
        BlockPoolConfig config;
        config.layer_num = layer_num;
        config.block_num = block_num;
        config.layout = KV_FIRST;
        
        config.k_block_size = kv_block_size;
        config.v_block_size = kv_block_size;
        config.block_size = kv_block_size * 2;
        
        // 2 (K+V) * layer_num * block_num * kv_block_size
        config.total_size = 2 * static_cast<size_t>(layer_num) * block_num * kv_block_size;
        
        config.k_block_stride = kv_block_size;
        config.v_block_stride = kv_block_size;
        config.k_layer_stride = static_cast<size_t>(block_num) * kv_block_size;
        config.v_layer_stride = static_cast<size_t>(block_num) * kv_block_size;
        
        return config;
    }
};

}  // namespace rtp_llm
