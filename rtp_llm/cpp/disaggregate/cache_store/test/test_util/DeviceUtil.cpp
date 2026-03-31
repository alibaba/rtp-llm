#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/DeviceUtil.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cuda_runtime.h>

using namespace std;

namespace rtp_llm {

DeviceUtil::DeviceUtil(const DeviceResourceConfig device_resource_config) {
    rtp_llm::ParallelismConfig           parallelism_config;
    rtp_llm::ModelConfig                 model_config;
    rtp_llm::EPLBConfig                  eplb_config;
    rtp_llm::FMHAConfig                  fmha_config;
    rtp_llm::DeviceResourceConfig        device_resource_config_copy = device_resource_config;
    rtp_llm::MoeConfig                   moe_config;
    rtp_llm::SpeculativeExecutionConfig  sp_config;
    rtp_llm::MiscellaneousConfig         misc_config;
    rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
    rtp_llm::HWKernelConfig              hw_kernel_config;
    rtp_llm::ConcurrencyConfig           concurrency_config;
    rtp_llm::FfnDisAggregateConfig       ffn_disaggregate_config;
    rtp_llm::RuntimeConfig               runtime_config;
    rtp_llm::ModelSpecificConfig         model_specific_config;

    rtp_llm::initExecCtx(parallelism_config,
                         model_config,
                         eplb_config,
                         fmha_config,
                         device_resource_config_copy,
                         moe_config,
                         sp_config,
                         misc_config,
                         profiling_debug_logging_config,
                         hw_kernel_config,
                         concurrency_config,
                         ffn_disaggregate_config,
                         runtime_config,
                         model_specific_config,
                         rtp_llm::NcclCommConfig{});
}

DeviceUtil::~DeviceUtil() {}

void* DeviceUtil::mallocCPU(size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto  tensor = torch::empty({(int64_t)size}, torch::TensorOptions().dtype(torch::kUInt8)).pin_memory();
    void* ptr    = tensor.data_ptr();
    buffer_map_.insert({ptr, tensor});
    return ptr;
}

void DeviceUtil::freeCPU(void* ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         iter = buffer_map_.find(ptr);
    if (iter != buffer_map_.end()) {
        buffer_map_.erase(iter);
    } else {
        RTP_LLM_LOG_ERROR("freeCPU failed, ptr not found");
    }
}

void* DeviceUtil::mallocGPU(size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto  tensor = torch::empty({(int64_t)size}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    void* ptr    = tensor.data_ptr();
    buffer_map_.insert({ptr, tensor});
    return ptr;
}

void DeviceUtil::freeGPU(void* ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         iter = buffer_map_.find(ptr);
    if (iter != buffer_map_.end()) {
        buffer_map_.erase(iter);
    } else {
        RTP_LLM_LOG_ERROR("freeGPU failed, ptr not found");
    }
}

void DeviceUtil::memsetCPU(void* ptr, int value, size_t len) {
    memset(ptr, value, len);
}

bool DeviceUtil::memsetGPU(void* ptr, int value, size_t len) {
    cudaMemset(ptr, value, len);
    return true;
}

bool DeviceUtil::memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size) {
    auto dst_device = dst_gpu ? torch::kCUDA : torch::kCPU;
    auto src_device = src_gpu ? torch::kCUDA : torch::kCPU;
    auto dst_t = torch::from_blob(dst, {(int64_t)size}, torch::TensorOptions().dtype(torch::kUInt8).device(dst_device));
    auto src_t = torch::from_blob(
        const_cast<void*>(src), {(int64_t)size}, torch::TensorOptions().dtype(torch::kUInt8).device(src_device));
    dst_t.copy_(src_t);
    runtimeSyncAndCheck();
    return true;
}

}  // namespace rtp_llm
