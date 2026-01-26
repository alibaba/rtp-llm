#pragma once

#include <memory>
#include <vector>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/MemoryLayout.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

class LayerFirstLayoutStrategy {
public:
    ~LayerFirstLayoutStrategy() = default;

    bool init(const MemoryLayoutConfig& config,
              torch::Tensor&            kv_cache_buffer,
              torch::Tensor&            kv_scale_buffer,
              void*                     cache_base_ptr);

    std::vector<torch::Tensor> getLayerCacheTensors() const;
    std::vector<torch::Tensor> getLayerScaleCacheTensors() const;

    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const;

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const;

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    void* getKCacheAddr(int layer_id, int block_id) const;

    void* getVCacheAddr(int layer_id, int block_id) const;

    // For backward compatibility with old cache system
    const KVCacheBuffer& kvCacheBuffer() const;

private:
    void checkLayerIdValidity(int layer_id) const;

    MemoryLayoutConfig         config_;
    void*                      cache_base_ptr_    = nullptr;
    void*                      kv_scale_base_ptr_ = nullptr;
    rtp_llm::DataType          data_type_         = rtp_llm::TYPE_INVALID;
    std::vector<torch::Tensor> layer_kv_tensors_;
    std::vector<torch::Tensor> layer_kv_scale_tensors_;
    // Byte view (INT8) tensors that point to the same underlying memory as layer_kv_tensors_ / layer_kv_scale_tensors_.
    // Used by byte-based slicing logic (e.g. splitKVPartition).
    std::vector<torch::Tensor> layer_kv_tensors_byte_;
    std::vector<torch::Tensor> layer_kv_scale_tensors_byte_;
    KVCacheBuffer              kv_cache_buffer_;
};

}  // namespace rtp_llm
