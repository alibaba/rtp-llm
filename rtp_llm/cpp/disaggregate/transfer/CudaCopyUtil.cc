#include "rtp_llm/cpp/disaggregate/transfer/CudaCopyUtil.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool CudaCopyUtil::batchCopyToHost(std::vector<CopyTask>& tasks) {
    if (tasks.empty()) {
        return true;
    }

    auto* device = DeviceFactory::getDefaultDevice();
    if (!device) {
        RTP_LLM_LOG_WARNING("Device is not initialized");
        return false;
    }

    // 构建 MultiCopyParams
    MultiCopyParams params;
    params.multi_src.reserve(tasks.size());
    params.multi_dst.reserve(tasks.size());

    for (auto& task : tasks) {
        if (!task.dst_ptr) {
            RTP_LLM_LOG_WARNING("dst_ptr is nullptr, caller must pre-allocate dst_ptr");
            return false;
        }

        // 创建临时 Buffer 包装 raw pointer
        // src 来自 GPU (MEMORY_GPU)，dst 是 CPU (MEMORY_CPU_PINNED 或 MEMORY_CPU)
        auto src_buffer = std::make_shared<Buffer>(
            MemoryType::MEMORY_GPU, DataType::TYPE_BYTES, std::vector<size_t>{task.size}, task.src_ptr);
        auto dst_buffer = std::make_shared<Buffer>(
            MemoryType::MEMORY_CPU_PINNED, DataType::TYPE_BYTES, std::vector<size_t>{task.size}, task.dst_ptr);

        params.multi_src.push_back(src_buffer);
        params.multi_dst.push_back(dst_buffer);
    }

    // 调用 DeviceOps::noBlockCopy
    device->noBlockCopy(params);
    return true;
}

bool CudaCopyUtil::batchCopyToDevice(std::vector<CopyTask>& tasks) {
    if (tasks.empty()) {
        return true;
    }

    auto* device = DeviceFactory::getDefaultDevice();
    if (!device) {
        RTP_LLM_LOG_WARNING("Device is not initialized");
        return false;
    }

    // 构建 MultiCopyParams
    MultiCopyParams params;
    params.multi_src.reserve(tasks.size());
    params.multi_dst.reserve(tasks.size());

    for (auto& task : tasks) {
        if (!task.dst_ptr) {
            RTP_LLM_LOG_WARNING("dst_ptr is nullptr, caller must pre-allocate dst_ptr");
            return false;
        }

        // 创建临时 Buffer 包装 raw pointer
        // src 来自 CPU (MEMORY_CPU_PINNED 或 MEMORY_CPU)，dst 是 GPU (MEMORY_GPU)
        auto src_buffer = std::make_shared<Buffer>(
            MemoryType::MEMORY_CPU_PINNED, DataType::TYPE_BYTES, std::vector<size_t>{task.size}, task.src_ptr);
        auto dst_buffer = std::make_shared<Buffer>(
            MemoryType::MEMORY_GPU, DataType::TYPE_BYTES, std::vector<size_t>{task.size}, task.dst_ptr);

        params.multi_src.push_back(src_buffer);
        params.multi_dst.push_back(dst_buffer);
    }

    // 调用 DeviceOps::noBlockCopy
    device->noBlockCopy(params);
    return true;
}

}  // namespace rtp_llm
