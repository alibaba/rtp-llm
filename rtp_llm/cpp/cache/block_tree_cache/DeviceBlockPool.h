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
    AllocationType                  allocation_type{AllocationType::DEVICE};
    bool                            use_pinned_cpu_backing{false};
    bool                            use_cuda_malloc_backing{false};
};

struct DeviceBlockBuffer {
    BlockIdxType block;
    void*        addr;
    size_t       bytes;
};

class DeviceBlockPool;
using DeviceBlockPoolPtr = std::shared_ptr<DeviceBlockPool>;

// DeviceBlockPool is the device (GPU) block pool: a single contiguous backing tensor
// sliced per MemoryLayoutConfig into per-layer KV (and optional scale) tensors, with the
// malloc/free/incRef/decRef/refCount lifecycle inherited from IBlockPool. Per-block access
// uses blockBuffers(); getBaseAddress()/getTotalSizeBytes() are the only raw byte-level
// accessors, exposed so RDMA can register the whole backing buffer.
class DeviceBlockPool: public IBlockPool {
public:
    explicit DeviceBlockPool(std::shared_ptr<const DeviceBlockPoolConfig> config);
    ~DeviceBlockPool() override;

    // Allocates the backing buffer, builds the per-layout MemoryLayoutStrategy objects and
    // the global-layer -> (layout, local layer) mapping. Invariants enforced via RTP_LLM_CHECK.
    bool init();

    MemoryType where() const;

    // Per-(global) layer KV / KV-scale backing tensors for the allocator path, by global layer id.
    std::vector<torch::Tensor> allLayerCacheBase() const;
    std::vector<torch::Tensor> allLayerScaleCacheBase() const;

    // Address of one (layer, block) for the allocator/device-copy fast path.
    BlockAddrInfo convertIndexToAddr(int layer_id, BlockIdxType block) const;

    // Backing buffer(s) for one (layer, block), the allocator/device-copy view. CHECK-fails
    // if the pool is uninitialized or the block is not allocated.
    std::vector<DeviceBlockBuffer> blockBuffers(int layer_id, BlockIdxType block) const;
    std::vector<DeviceBlockBuffer>
    blockBuffers(int layer_id, BlockIdxType block, int partition_count, int partition_id) const;

    // Transfer-facing (PD/P2P/RDMA/Remote/Memory) BlockInfo view over the same layout;
    // retained because those paths need torch-free is_cuda/device_index/scalar_type descriptors.
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, BlockIdxType block, int partition_count, int partition_id) const;

    // RDMA user-memory registration of the backing GPU buffer for PD separation / remote KV
    // transfer. Idempotent (guarded by kvcache_reg_mr_); model_id kept for call-site parity.
    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    void    deregUserMr();
    int64_t getMrCostTimeMs() const {
        return mr_cost_time_ms_;
    }

    // Contiguous backing-buffer accessors for the PD / Remote KV transfer periphery (bulk MR
    // registration of the whole pool buffer); the only raw byte-level accessors on this pool.
    void* getBaseAddress() const {
        return cache_base_ptr_;
    }
    size_t getTotalSizeBytes() const;

    std::string debugString() const override;

private:
    // Computes physical_block_count from memory_layouts[*].block_num BEFORE the IBlockPool
    // base ctor runs (it needs physical_block_count > 1 to seed the free list).
    static std::shared_ptr<const DeviceBlockPoolConfig> normalizeConfig(const std::shared_ptr<const DeviceBlockPoolConfig>& config);

    const DeviceBlockPoolConfig& config() const;

    // init() helpers: backing allocation + layout-strategy setup.
    void initializeCacheBuffer();
    void initializeCudaMallocBuffer();
    void initializeLayerMappings();
    void initializeLayoutStrategies();
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

    std::pair<int, int> mapGlobalLayerIdToLocal(int global_layer_id) const;
    void                checkLayoutValidity(int layout_id) const;
    std::vector<DeviceBlockBuffer> toDeviceBlockBuffers(const std::vector<BlockInfo>& infos, BlockIdxType block) const;

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
    // global_layer_id -> {layout_index, local_layer_id}
    std::vector<std::pair<int, int>> global_layer_to_local_;

    std::vector<torch::Tensor> global_layer_kv_tensors_;
    std::vector<torch::Tensor> global_layer_kv_scale_tensors_;
};

}  // namespace rtp_llm
