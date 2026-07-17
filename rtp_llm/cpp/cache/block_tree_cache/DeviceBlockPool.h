#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/MemoryLayoutConfig.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

namespace rtp_llm {

class CacheStore;
class MemoryUtil;

// Config for the device (GPU) block pool. Derives BlockPoolConfigBase for the IBlockPool
// lifecycle; physical_block_count is reconciled against memory_layouts[*].block_num by
// normalizeConfig() - see DeviceBlockPool.cc.
struct DeviceBlockPoolConfig: public BlockPoolConfigBase {
    size_t                          total_size_bytes{0};
    std::vector<MemoryLayoutConfig> memory_layouts;
    bool                            use_cuda_malloc_backing{false};
};

class DeviceBlockPool;
using DeviceBlockPoolPtr = std::shared_ptr<DeviceBlockPool>;

class DeviceBlockPool: public IBlockPool {
public:
    explicit DeviceBlockPool(std::shared_ptr<const DeviceBlockPoolConfig> config);
    ~DeviceBlockPool() override;

    // Allocates the backing buffer, builds the per-layout MemoryLayoutStrategy objects and
    // the global-layer -> (layout, local layer) mapping. Invariants enforced via RTP_LLM_CHECK.
    bool init();

    
    // Stable CUDA device index of the backing buffer. Returns -1 for non-CUDA builds.
    int deviceIndex() const;
    
    MemoryType where() const;
    std::vector<torch::Tensor> allLayerCacheBase() const;
    std::vector<torch::Tensor> allLayerScaleCacheBase() const;

    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    void    deregUserMr();
    int64_t getMrCostTimeMs() const {
        return mr_cost_time_ms_;
    }
    BlockAddrInfo convertIndexToAddr(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, BlockIdxType block, int partition_count, int partition_id) const;

    void* getBaseAddress() const {
        return cache_base_ptr_;
    }
    size_t getTotalSizeBytes() const;

    std::string debugString() const override;

private:
    const DeviceBlockPoolConfig& config() const;
    // global_layer_id -> {layout_index, local_layer_id}
    std::pair<int, int> mapGlobalLayerIdToLocal(int global_layer_id) const;
    void                checkLayoutValidity(int layout_id) const;

    // Helper functions for init()
    static std::shared_ptr<const DeviceBlockPoolConfig>
         normalizeConfig(const std::shared_ptr<const DeviceBlockPoolConfig>& config);
    void initializeCacheBuffer();
    void initializeCudaMallocBuffer();
    void initializeLayerMappings();
    void initializeLayoutStrategies();

    // Helper functions for initializeLayoutStrategies()
    void processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor, size_t& global_layer_begin);
    torch::Tensor createTensor(const torch::Tensor& full_tensor,
                               int64_t              offset,
                               int64_t              size,
                               size_t               layout_idx,
                               const std::string&   tensor_type);
    void initializeLayoutStrategy(size_t                    layout_idx,
                                  const MemoryLayoutConfig& layout_cfg,
                                  torch::Tensor&            kv_cache_tensor,
                                  torch::Tensor&            kv_scale_tensor);
    void processLayerTensors(size_t layout_idx, const MemoryLayoutConfig& layout_cfg, size_t& global_layer_begin);

    // Helper functions for regUserMr/deregUserMr
    void registerUserMrForBuffer(std::shared_ptr<MemoryUtil> memory_util,
                                 size_t                      layout_idx,
                                 size_t                      offset_bytes,
                                 size_t                      bytes,
                                 size_t                      stride_bytes,
                                 bool                        gpu,
                                 const std::string&          buffer_type);
    void deregisterUserMrForBuffer(std::shared_ptr<MemoryUtil> memory_util,
                                   size_t                      layout_idx,
                                   size_t                      offset_bytes,
                                   bool                        gpu,
                                   const std::string&          buffer_type);

private:
    torch::Tensor cache_aligned_buffer_;
    void*         cache_base_ptr_{nullptr};

    bool                        kvcache_reg_mr_  = false;
    int64_t                     mr_cost_time_ms_ = 0;
    std::shared_ptr<CacheStore> cache_store_;

    std::vector<std::unique_ptr<MemoryLayoutStrategy>> layout_strategies_;
    std::vector<std::pair<int, int>>                   global_layer_to_local_;
    std::vector<torch::Tensor>                         global_layer_kv_tensors_;
    std::vector<torch::Tensor>                         global_layer_kv_scale_tensors_;
};

}  // namespace rtp_llm
