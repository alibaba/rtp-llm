#pragma once

#include <memory>
#include <vector>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class MemoryLayoutStrategy {
public:
    virtual ~MemoryLayoutStrategy() = default;

    virtual bool init(const MemoryLayoutConfig& config,
                      torch::Tensor&            kv_cache_buffer,
                      torch::Tensor&            kv_scale_buffer,
                      void*                     cache_base_ptr) = 0;

    virtual std::vector<torch::Tensor> getLayerCacheTensors() const      = 0;
    virtual std::vector<torch::Tensor> getLayerScaleCacheTensors() const = 0;

    virtual BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const = 0;

    virtual BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const = 0;

    virtual std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const = 0;

    virtual void* getKCacheAddr(int layer_id, int block_id) const = 0;

    virtual void* getVCacheAddr(int layer_id, int block_id) const = 0;

    // For backward compatibility with old cache system
    // Returns KVCacheBuffer for layouts that support K/V separation
    virtual const KVCacheBuffer& kvCacheBuffer() const = 0;

protected:
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

class LayerFirstLayoutStrategy: public MemoryLayoutStrategy {
public:
    bool init(const MemoryLayoutConfig& config,
              torch::Tensor&            kv_cache_buffer,
              torch::Tensor&            kv_scale_buffer,
              void*                     cache_base_ptr) override;

    std::vector<torch::Tensor> getLayerCacheTensors() const override;
    std::vector<torch::Tensor> getLayerScaleCacheTensors() const override;
    BlockAddrInfo              convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferPtrInfo         convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BufferPtr>
          convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    void* getKCacheAddr(int layer_id, int block_id) const override;
    void* getVCacheAddr(int layer_id, int block_id) const override;
    const KVCacheBuffer& kvCacheBuffer() const override;

private:
    void checkLayerIdValidity(int layer_id) const;
};

class MemoryLayoutStrategyFactory {
public:
    static std::unique_ptr<MemoryLayoutStrategy> create(MemoryLayout layout);
};

}  // namespace rtp_llm
