#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

using namespace std;

namespace rtp_llm {

BlockPoolConfig createTestConfig(MemoryLayout layout               = LAYER_FIRST,
                                 size_t       k_block_stride_bytes = 512,
                                 size_t       v_block_stride_bytes = 512) {
    BlockPoolConfig config;
    config.layer_num          = 4;
    config.block_num          = 10;
    config.block_stride_bytes = k_block_stride_bytes + v_block_stride_bytes;
    config.layout             = layout;

    config.total_size_bytes   = config.layer_num * config.block_num * config.block_stride_bytes;
    config.k_block_size_bytes = config.k_block_stride_bytes * config.layer_num;
    config.v_block_size_bytes = config.v_block_stride_bytes * config.layer_num;
    return config;
}

DeviceBase* createDevice() {
    torch::manual_seed(114514);
    rtp_llm::ParallelismConfig           parallelism_config;
    rtp_llm::ModelConfig                 model_config;
    rtp_llm::EPLBConfig                  eplb_config;
    rtp_llm::FMHAConfig                  fmha_config;
    rtp_llm::DeviceResourceConfig        device_resource_config;
    rtp_llm::MoeConfig                   moe_config;
    rtp_llm::SpeculativeExecutionConfig  sp_config;
    rtp_llm::MiscellaneousConfig         misc_config;
    rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
    rtp_llm::HWKernelConfig              hw_kernel_config;
    rtp_llm::ConcurrencyConfig           concurrency_config;
    rtp_llm::FfnDisAggregateConfig       ffn_disaggregate_config;
    rtp_llm::RuntimeConfig               runtime_config;

    device_resource_config.device_reserve_memory_bytes = 1024L * 1024 * 1024;  // 1GB
    device_resource_config.host_reserve_memory_bytes   = 1024L * 1024 * 1024;  // 1GB

    rtp_llm::DeviceFactory::initDevices(parallelism_config,
                                        model_config,
                                        eplb_config,
                                        fmha_config,
                                        device_resource_config,
                                        moe_config,
                                        sp_config,
                                        misc_config,
                                        profiling_debug_logging_config,
                                        hw_kernel_config,
                                        concurrency_config,
                                        ffn_disaggregate_config,
                                        runtime_config);
    return rtp_llm::DeviceFactory::getDefaultDevice();
}

BlockPoolPtr createBlockPool() {
    auto device     = createDevice();
    auto config     = createTestConfig();
    auto block_pool = std::make_shared<BlockPool>(config, device);
    return block_pool;
}

}  // namespace rtp_llm
