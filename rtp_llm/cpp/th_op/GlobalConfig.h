// EngineInitParams.h
#pragma once
#include <memory>
#include <iostream> // For showDebugInfo
#include <mutex> // for std::once_flag and std::call_once (although Meyers is usually fine without explicit mutex)
#include "rtp_llm/cpp/th_op/ConfigModules.h"

namespace rtp_llm{

struct ConfigCollection {
    ParallelismDistributedConfig parallelism_distributed_config;
    ConcurrencyConfig concurrency_config;
    FMHAConfig fmha_config;
    KVCacheConfig kv_cache_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    HWKernelConfig hw_kernel_config;
    DeviceResourceConfig device_resource_config;
    SamplerConfig sampler_config;
    MoeConfig moe_config;
    ModelSpecificConfig model_specific_config;
    SpeculativeExecutionConfig sp_config;
    ServiceDiscoveryConfig service_discovery_config;
    CacheStoreConfig cache_store_config;
    SchedulerConfig scheduler_config;
    BatchDecodeSchedulerConfig batch_decode_scheduler_config;
    FIFOSchedulerConfig fifo_scheduler_config;
    MiscellaneousConfig misc_config;
};

class GlobalConfig {
public:
    static rtp_llm::ConfigCollection& get() {
        static ConfigCollection global_config;
        return global_config;
    }

    static void update_from_env_for_test(){
        ConfigCollection& config = GlobalConfig::get();
        config.parallelism_distributed_config.update_from_env_for_test();
        config.concurrency_config.update_from_env_for_test();
        config.fmha_config.update_from_env_for_test();
        config.kv_cache_config.update_from_env_for_test();
        config.profiling_debug_logging_config.update_from_env_for_test();
        config.hw_kernel_config.update_from_env_for_test();
        config.device_resource_config.update_from_env_for_test();
        config.sampler_config.update_from_env_for_test();
        config.moe_config.update_from_env_for_test();
        config.model_specific_config.update_from_env_for_test();
        config.sp_config.update_from_env_for_test();
        config.service_discovery_config.update_from_env_for_test();
        config.cache_store_config.update_from_env_for_test();
        config.scheduler_config.update_from_env_for_test();
        config.batch_decode_scheduler_config.update_from_env_for_test();
        config.fifo_scheduler_config.update_from_env_for_test();
        config.misc_config.update_from_env_for_test();
    }

    static std::string debug_to_string(){
        return "GlobalConfig:\n" 
        + get().parallelism_distributed_config.to_string() + "\n"
        + get().concurrency_config.to_string() + "\n"
        + get().fmha_config.to_string() + "\n"
        + get().kv_cache_config.to_string() + "\n"
        + get().profiling_debug_logging_config.to_string() + "\n"
        + get().hw_kernel_config.to_string() + "\n"
        + get().device_resource_config.to_string() + "\n"
        + get().sampler_config.to_string() + "\n"
        + get().moe_config.to_string() + "\n"
        + get().model_specific_config.to_string() + "\n"
        + get().sp_config.to_string() + "\n"
        + get().service_discovery_config.to_string() + "\n"
        + get().cache_store_config.to_string() + "\n"
        + get().scheduler_config.to_string() + "\n"
        + get().batch_decode_scheduler_config.to_string() + "\n"
        + get().fifo_scheduler_config.to_string() + "\n"
        + get().misc_config.to_string() + "\n";
    }
private:
    GlobalConfig() = default;
    ~GlobalConfig() = default;
    GlobalConfig(const GlobalConfig&) = delete;
    GlobalConfig& operator=(const GlobalConfig&) = delete;
};

}
