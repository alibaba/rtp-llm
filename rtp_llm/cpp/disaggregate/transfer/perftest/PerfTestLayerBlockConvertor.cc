#include "rtp_llm/cpp/disaggregate/transfer/perftest/PerfTestLayerBlockConvertor.h"

#include <iostream>

namespace rtp_llm {

PerfTestLayerBlockConvertor::PerfTestLayerBlockConvertor(DeviceBase* device, size_t block_size):
    device_(device), block_size_(block_size) {}

void PerfTestLayerBlockConvertor::addBuffer(int layer_id, int block_id, BufferPtr buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_map_[layer_id][block_id].push_back(buffer);
}

std::vector<BufferPtr> PerfTestLayerBlockConvertor::convertIndexToBuffer(int layer_id,
                                                                         int block_id,
                                                                         int partition_count,
                                                                         int partition_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = buffer_map_.find(layer_id);
    if (iter == buffer_map_.end()) {
        return {};
    }
    auto block_iter = iter->second.find(block_id);
    if (block_iter == iter->second.end()) {
        return {};
    }
    return block_iter->second;
}

bool PerfTestLayerBlockConvertor::preallocateBuffers(int block_count, int fill_value) {
    std::cout << "Preallocating " << block_count << " blocks with size " << block_size_
              << " bytes each (fill_value=" << fill_value << ")" << std::endl;

    for (int block_id = 0; block_id < block_count; ++block_id) {
        auto buffer = device_->allocateBuffer(
            {DataType::TYPE_UINT8, {static_cast<int64_t>(block_size_)}, AllocationType::DEVICE}, {});
        if (buffer == nullptr) {
            std::cerr << "Failed to allocate buffer for block " << block_id << std::endl;
            return false;
        }
        // 填充测试数据
        device_->bufMemset(*buffer, fill_value);
        addBuffer(0, block_id, buffer);  // layer_id = 0
    }
    device_->syncAndCheck();
    std::cout << "Successfully preallocated " << block_count << " blocks" << std::endl;
    return true;
}

std::vector<BufferPtr> PerfTestLayerBlockConvertor::getBuffers() const {
    std::vector<BufferPtr>      buffers;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& layer_buffers : buffer_map_) {
        for (const auto& block_buffers : layer_buffers.second) {
            buffers.insert(buffers.end(), block_buffers.second.begin(), block_buffers.second.end());
        }
    }
    return buffers;
}

std::vector<std::pair<BufferPtr, size_t>> PerfTestLayerBlockConvertor::getAllBuffers() const {
    std::vector<std::pair<BufferPtr, size_t>> result;
    std::lock_guard<std::mutex>               lock(mutex_);
    for (const auto& layer_buffers : buffer_map_) {
        for (const auto& block_buffers : layer_buffers.second) {
            for (const auto& buffer : block_buffers.second) {
                result.emplace_back(buffer, buffer->sizeBytes());
            }
        }
    }
    return result;
}
}  // namespace rtp_llm
