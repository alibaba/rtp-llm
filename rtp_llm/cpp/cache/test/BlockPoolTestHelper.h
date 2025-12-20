#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

BlockPoolConfig createTestConfig(MemoryLayout layout               = LAYER_FIRST,
                                 size_t       k_block_stride_bytes = 512,
                                 size_t       v_block_stride_bytes = 512) {
    BlockPoolConfig config;
    config.layer_num             = 4;
    config.block_num             = 10;
    config.kv_block_stride_bytes = k_block_stride_bytes + v_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.block_stride_bytes    = config.kv_block_stride_bytes;
    config.layout                = layout;

    config.kv_block_pool_size_bytes = config.layer_num * config.block_num * config.kv_block_stride_bytes;
    config.kv_scale_offset_bytes    = config.kv_block_pool_size_bytes;
    config.total_size_bytes         = config.kv_block_pool_size_bytes;
    config.total_size               = config.total_size_bytes;

    config.k_block_stride_bytes = k_block_stride_bytes;
    config.v_block_stride_bytes = v_block_stride_bytes;

    config.k_block_size_bytes = config.k_block_stride_bytes * config.layer_num;
    config.v_block_size_bytes = config.v_block_stride_bytes * config.layer_num;
    return config;
}

DeviceBase* createDevice() {
    torch::manual_seed(114514);
    rtp_llm::GptInitParameter gpt_init_params;
    gpt_init_params.device_resource_config.device_reserve_memory_bytes = 1024L * 1024 * 1024;  // 1GB
    gpt_init_params.device_resource_config.host_reserve_memory_bytes   = 1024L * 1024 * 1024;  // 1GB
    rtp_llm::DeviceFactory::initDevices(gpt_init_params);
    auto device = rtp_llm::DeviceFactory::getDefaultDevice();
    return device;
}

BlockPoolPtr createBlockPool() {
    auto device     = createDevice();
    auto config     = createTestConfig();
    auto block_pool = std::make_shared<BlockPool>(config, device);
    return block_pool;
}

}  // namespace rtp_llm
