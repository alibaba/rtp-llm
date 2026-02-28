#pragma once

#include <memory>
#include <vector>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/MemoryLayoutConfig.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

class MemoryLayoutStrategy {
public:
    ~MemoryLayoutStrategy() = default;

    bool init(const MemoryLayoutConfig& config,
              torch::Tensor&            kv_cache_tensor,
              torch::Tensor&            kv_scale_tensor,
              void*                     cache_base_ptr);

    std::vector<torch::Tensor> getLayerCacheTensors() const;
    std::vector<torch::Tensor> getLayerScaleCacheTensors() const;

    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const;

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const;

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    const MemoryLayoutConfig& getConfig() const {
        return config_;
    }

private:
    void                   checkLayerIdValidity(int layer_id) const;
    void                   processKVTensor(torch::Tensor& kv_cache_tensor);
    bool                   processScaleTensor(torch::Tensor& kv_scale_tensor);
    void                   clearKVTensor(torch::Tensor& kv_cache_tensor);
    void                   clearScaleTensor(torch::Tensor& kv_scale_tensor);
    BlockInfo              makeBlockInfo(const torch::Tensor& tensor, void* addr, size_t size_bytes) const;
    std::vector<BlockInfo> createBasicBlockInfo(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    createPartitionedBlockInfo(int layer_id, int block_id, int partition_count, int partition_id) const;
    std::vector<BlockInfo>
    createPartitionedSubBlocks(const torch::Tensor& layer_tensor, void* base_addr, const KVPartitionBytes& parts) const;

    MemoryLayoutConfig         config_;
    void*                      cache_base_ptr_    = nullptr;
    void*                      kv_scale_base_ptr_ = nullptr;
    rtp_llm::DataType          data_type_         = rtp_llm::TYPE_INVALID;
    std::vector<torch::Tensor> layer_kv_tensors_;
    std::vector<torch::Tensor> layer_kv_scale_tensors_;
};

}  // namespace rtp_llm