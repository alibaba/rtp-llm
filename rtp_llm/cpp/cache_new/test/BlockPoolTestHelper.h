#pragma once

#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

BlockPoolConfig
createTestConfig(MemoryLayout layout = LAYER_FIRST, size_t k_block_size = 512, size_t v_block_size = 512) {
    BlockPoolConfig config;
    config.layer_num  = 4;
    config.block_num  = 10;
    config.block_size = 1024;
    config.layout     = layout;
    config.total_size = config.layer_num * config.block_num * config.block_size;

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
