#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/DeviceUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cuda_runtime.h>

using namespace std;

namespace rtp_llm {

DeviceUtil::DeviceUtil(const DeviceResourceConfig device_resource_config) {
    rtp_llm::initRuntime(/*device_id=*/0,
                         /*trace_memory=*/false,
                         device_resource_config.enable_comm_overlap,
                         rtp_llm::MlaOpsType::AUTO);
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
