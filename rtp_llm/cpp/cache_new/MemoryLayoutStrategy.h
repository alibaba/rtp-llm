#pragma once

#include <memory>
#include <vector>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

struct BlockAddrInfo {
    BufferPtr k_addr;
    BufferPtr v_addr;
    BufferPtr k_scale_addr;
    BufferPtr v_scale_addr;
};

class MemoryLayoutStrategy {
public:
    virtual ~MemoryLayoutStrategy() = default;
    
    bool init(const BlockPoolConfig& config, 
                     torch::Tensor& cache_buffer,
                     void* cache_base_ptr) = 0;
    
    virtual std::vector<torch::Tensor> getLayerCacheTensors() const = 0;
    
    virtual BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const = 0;

    virtual BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const = 0;

    virtual void* getKCacheAddr(int layer_id, int block_id) const = 0;
    
    virtual void* getVCacheAddr(int layer_id, int block_id) const = 0;

protected:
    BlockPoolConfig config_;
    void* cache_base_ptr_ = nullptr;
    std::vector<torch::Tensor> layer_kv_tensors_;
};


class LayerFirstLayoutStrategy : public MemoryLayoutStrategy {
public:
    bool init(const BlockPoolConfig& config, 
              torch::Tensor& cache_buffer,
              void* cache_base_ptr) override;
              
    std::vector<torch::Tensor> getLayerCacheTensors() const override;
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    void* getKCacheAddr(int layer_id, int block_id) const override;
    void* getVCacheAddr(int layer_id, int block_id) const override;
};


class KVFirstLayoutStrategy : public MemoryLayoutStrategy {
public:
    bool init(const BlockPoolConfig& config, 
              torch::Tensor& cache_buffer,
              void* cache_base_ptr) override;
              
    std::vector<torch::Tensor> getLayerCacheTensors() const override;
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    void* getKCacheAddr(int layer_id, int block_id) const override;
    void* getVCacheAddr(int layer_id, int block_id) const override;

private:
    torch::Tensor k_cache_tensor_;
    torch::Tensor v_cache_tensor_;
};

class MemoryLayoutStrategyFactory {
public:
    static std::unique_ptr<MemoryLayoutStrategy> create(MemoryLayout layout);
};

}  // namespace rtp_llm
