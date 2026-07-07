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

// Config for the v4 device (GPU) block pool. Mirrors rtp_llm::BlockPoolConfig
// (rtp_llm/cpp/cache/BlockPoolConfig.h) but derives BlockPoolConfigBase so it plugs
// into the v4 IBlockPool lifecycle. physical_block_count (inherited from
// BlockPoolConfigBase) is reconciled against memory_layouts[*].block_num by
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

// DeviceBlockPool is the v4 device (GPU) block pool. It reuses the backing-allocation and
// MemoryLayoutStrategy machinery of the legacy monolithic device pool
// (rtp_llm/cpp/cache/BlockPool.{h,cc}) - a single contiguous backing tensor sliced per
// MemoryLayoutConfig into per-layer KV (and optional scale) tensors - but lives in the
// flat rtp_llm namespace and derives IBlockPool, so all
// malloc/free/incRef/decRef/refCount/metrics lifecycle comes from IBlockPool unchanged.
// The raw byte-size accessors of the legacy pool are intentionally not re-exposed here;
// callers use blockBuffers() instead.
class DeviceBlockPool: public IBlockPool {
public:
    explicit DeviceBlockPool(std::shared_ptr<const DeviceBlockPoolConfig> config);
    ~DeviceBlockPool() override;

    // Validates the config (memory_layouts non-empty, per-layout block_num/layer_num/
    // kv_block_pool_size_bytes invariants), allocates the backing buffer (device, pinned
    // CPU, or cudaMalloc, per allocation_type/use_pinned_cpu_backing/use_cuda_malloc_backing),
    // builds the per-layout MemoryLayoutStrategy objects and the global-layer -> (layout,
    // local layer) mapping, and marks the pool initialized. Always returns true;
    // invariant violations are enforced via RTP_LLM_CHECK.
    bool init();

    MemoryType where() const;

    // Returns the backing buffer(s) for one (layer, block). RTP_LLM_CHECK-fails if the
    // pool is not initialized or the block is not currently allocated. layer_id and the
    // partitioned overload's partition_count/partition_id are validated by the
    // underlying MemoryLayoutStrategy::convertIndexToBuffer().
    std::vector<DeviceBlockBuffer> blockBuffers(int layer_id, BlockIdxType block) const;
    std::vector<DeviceBlockBuffer>
    blockBuffers(int layer_id, BlockIdxType block, int partition_count, int partition_id) const;

    std::string debugString() const override;

private:
    // Computes/validates physical_block_count from memory_layouts[*].block_num BEFORE
    // the IBlockPool base constructor runs (it requires physical_block_count > 1 and
    // seeds free blocks immediately from it). Returns a copy of config with
    // physical_block_count normalized to the computed value.
    static std::shared_ptr<const DeviceBlockPoolConfig> normalizeConfig(const std::shared_ptr<const DeviceBlockPoolConfig>& config);

    const DeviceBlockPoolConfig& config() const;

    // init() helpers, adapted from rtp_llm::BlockPool's backing-allocation + layout-
    // strategy initialization (rtp_llm/cpp/cache/BlockPool.cc).
    void initializeCacheBuffer();
    void initializePinnedCpuBuffer(const char* log_context);
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

private:
    torch::Tensor cache_aligned_buffer_;
    void*         cache_base_ptr_{nullptr};

    std::vector<std::unique_ptr<MemoryLayoutStrategy>> layout_strategies_;
    // global_layer_id -> {layout_index, local_layer_id}
    std::vector<std::pair<int, int>> global_layer_to_local_;
};

}  // namespace rtp_llm
