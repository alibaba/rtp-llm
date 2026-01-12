#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

/// @brief Mock LayerBlockConvertor for performance testing
class PerfTestLayerBlockConvertor: public LayerBlockConvertor {
public:
    PerfTestLayerBlockConvertor(DeviceBase* device, size_t block_size);
    ~PerfTestLayerBlockConvertor() = default;

    void addBuffer(int layer_id, int block_id, BufferPtr buffer);

    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const override;

    std::vector<std::pair<BufferPtr, size_t>> getAllBuffers() const override;

    /// @brief 预分配所有需要的 buffer
    /// @param block_count 需要分配的 block 数量
    /// @param fill_value 填充值（默认为 0）
    /// @return 成功返回 true，失败返回 false
    bool preallocateBuffers(int block_count, int fill_value = 0);

    std::vector<BufferPtr> getBuffers() const;

private:
    mutable std::mutex                                                       mutex_;
    std::unordered_map<int, std::unordered_map<int, std::vector<BufferPtr>>> buffer_map_;
    DeviceBase*                                                              device_ = nullptr;
    size_t                                                                   block_size_;
};

}  // namespace rtp_llm
