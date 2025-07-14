#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/DeviceUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace std;

namespace rtp_llm {

DeviceUtil::DeviceUtil(const DeviceResourceConfig device_resource_config) {
    GptInitParameter gpt_init_params;
    gpt_init_params.device_resource_config = device_resource_config;
    rtp_llm::DeviceFactory::initDevices(gpt_init_params);
    device_ = DeviceFactory::getDefaultDevice();
}

DeviceUtil::~DeviceUtil() {}

void* DeviceUtil::mallocCPU(size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         buffer = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::HOST});
    buffer_map_.insert({buffer->data(), buffer});
    return buffer->data();
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
    auto buffer = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::DEVICE});
    buffer_map_.insert({buffer->data(), buffer});
    return buffer->data();
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
    auto buffer = rtp_llm::Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_UINT8, {len}, ptr);
    device_->bufMemset(buffer, value);
    return true;
}

bool DeviceUtil::memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size) {
    const auto dst_memory_type = dst_gpu ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
    const auto src_memory_type = src_gpu ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
    auto       dst_buffer      = rtp_llm::Buffer(dst_memory_type, DataType::TYPE_UINT8, {size}, dst);
    auto       src_buffer      = rtp_llm::Buffer(src_memory_type, DataType::TYPE_UINT8, {size}, src);
    device_->copy({dst_buffer, src_buffer});
    device_->syncAndCheck();
    return true;
}

}  // namespace rtp_llm
