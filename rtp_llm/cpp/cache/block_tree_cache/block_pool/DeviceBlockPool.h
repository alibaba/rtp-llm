#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/MemoryLayoutConfig.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

namespace rtp_llm {

class CacheStore;
class MemoryUtil;

// Config for the device (GPU) block pool. Derives BlockPoolConfigBase for the IBlockPool
// lifecycle; physical_block_count is reconciled against memory_layouts[*].block_num by
// normalizeConfig() - see DeviceBlockPool.cc.
struct DeviceBlockPoolConfig: public BlockPoolConfigBase {
    size_t                          total_size_bytes{0};
    std::vector<MemoryLayoutConfig> memory_layouts;
    bool                            use_pinned_cpu_backing{false};
    bool                            use_device_malloc_backing{false};
};

class DeviceBlockPool;
using DeviceBlockPoolPtr = std::shared_ptr<DeviceBlockPool>;

enum class DeviceBlockRefType : uint8_t {
    REQUEST,
    REMOTE_CONNECTOR,
    PD_CONNECTOR,
    COUNT,
};

class DeviceBlockPool: public IBlockPool {
public:
    explicit DeviceBlockPool(std::shared_ptr<const DeviceBlockPoolConfig> config);
    ~DeviceBlockPool() override;

    // Allocates the backing buffer, builds the per-layout MemoryLayoutStrategy objects and
    // the global-layer -> (layout, local layer) mapping. Invariants enforced via RTP_LLM_CHECK.
    bool init();

    void incRef(BlockIdxType block, DeviceBlockRefType ref_type = DeviceBlockRefType::REQUEST);
    void incRef(const BlockIdList& blocks, DeviceBlockRefType ref_type = DeviceBlockRefType::REQUEST);
    void decRef(BlockIdxType block, DeviceBlockRefType ref_type = DeviceBlockRefType::REQUEST);
    void decRef(const BlockIdList& blocks, DeviceBlockRefType ref_type = DeviceBlockRefType::REQUEST);

    // Allocate release bookkeeping before acquiring a storage I/O pin.
    // The caller keeps the pool alive and invokes the returned closure exactly once.
    // Release cannot allocate and isolates capacity-notification exceptions.
    std::function<void()> pinForStorageIo(BlockIdxType block);

    struct RequestReferenceUpdate {
        BlockIdxType block;
        int          old_count;
        int          new_count;
    };
    // Atomically replace caller-owned references and allocate replacements.
    // Each entry describes old_count > 0 references to a block and their new
    // multiplicity (>= 0). Duplicate blocks are coalesced; unchanged references
    // may be omitted. Reclaimable blocks are selected in first-occurrence order.
    // The caller must serialize access/sharing of its resource through publication.
    // On capacity failure ownership is unchanged, replacements is empty, and
    // required_free_blocks reports the total free capacity needed after transfers.
    bool tryReplaceRequestReferences(const std::vector<RequestReferenceUpdate>& reference_updates,
                                     int                                        replacement_count,
                                     BlockIndicesType&                          replacements,
                                     int&                                       required_free_blocks);
    // owned_references includes the request table and its live GPU lifetime holds.
    bool isExclusiveRequestBlock(BlockIdxType block, uint32_t owned_references = 1) const;
    bool hasRequestRef(BlockIdxType block) const;

    uint32_t refCount(BlockIdxType block) const;
    using IBlockPool::referencedBlocksNum;
    size_t referencedBlocksNum(DeviceBlockRefType ref_type = DeviceBlockRefType::REQUEST) const;

    struct ReferenceStats {
        TreeReferenceStats tree;
        size_t             request_ref_blocks;
        size_t             remote_connector_ref_blocks;
        size_t             pd_connector_ref_blocks;
    };
    // Snapshot existing per-pool counters together under the pool mutex.
    ReferenceStats referenceStats() const;

    // Stable CUDA device index of the backing buffer. Returns -1 for non-CUDA builds.
    int deviceIndex() const;

    struct LayerBackingView {
        torch::Tensor device_kv;
        torch::Tensor device_scale;
        torch::Tensor host_kv;
        torch::Tensor host_scale;
        uint32_t      device_block_count  = 0;
        uint32_t      device_spare_blocks = 0;
    };
    // Both views are at physical block granularity; host row zero corresponds
    // to logical block device_block_count, never a staging slot.
    LayerBackingView layerBackingView(int layer_id) const;

    // Decode split pools expose a stable device-readable int64 tensor in
    // pinned host memory. Callers own a lease on every ID read by a kernel;
    // writers invalidate before handing overwritten data to another forward.
    torch::Tensor        blockGenerations() const;
    std::vector<int64_t> blockGenerationsSnapshot() const;
    void                 invalidateBlockGenerations(const BlockIdList& blocks);
    std::optional<BlockIdList>
         mallocForSequence(size_t n, size_t sequence_tokens, BlockIdxType last_block, size_t hbm_share_denominator);
    bool hasSplitBacking() const {
        return !physical_layouts_.empty();
    }

    struct RegistrationRegion {
        void*  addr         = nullptr;
        size_t size_bytes   = 0;
        size_t stride_bytes = 0;
        bool   is_cuda      = false;
        int    device_index = -1;
    };
    // Only logical rows are exported. Scratch may make a device region
    // noncontiguous across layers, in which case each layer is exported alone.
    std::vector<RegistrationRegion> registrationRegions() const;

    MemoryType                 where() const;
    std::vector<torch::Tensor> allLayerCacheBase() const;
    std::vector<torch::Tensor> allLayerScaleCacheBase() const;

    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    void    deregUserMr();
    int64_t getMrCostTimeMs() const {
        return mr_cost_time_ms_;
    }
    BlockAddrInfo          convertIndexToAddr(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, BlockIdxType block, int partition_count, int partition_id) const;

    const DeviceBlockPoolConfig& config() const;

    void* getBaseAddress() const {
        return hasSplitBacking() ? nullptr : cache_base_ptr_;
    }
    bool usesDedicatedDeviceAllocation() const {
        return !hasSplitBacking() && uses_dedicated_device_allocation_;
    }
    size_t getAllocationSizeBytes() const {
        return hasSplitBacking() ? 0 : allocation_size_bytes_;
    }
    size_t getTotalSizeBytes() const;
    size_t blockSizeBytes() const override;

    std::string debugString() const override;

protected:
    void onBlockAllocatedNoLock(BlockIdxType block) noexcept override;
    void onFirstTreeRefNoLock(BlockIdxType block) override;
    bool onLastTreeRefNoLock(BlockIdxType block) override;
    bool hasExternalRefNoLock(BlockIdxType block) const override;

private:
    bool hasRequestRefNoLock(BlockIdxType block) const;
    void releaseStorageIoPin(BlockIdxType block, const std::function<void()>& notify) noexcept;
    // global_layer_id -> {layout_index, local_layer_id}
    std::pair<int, int> mapGlobalLayerIdToLocal(int global_layer_id) const;
    void                checkLayoutValidity(int layout_id) const;

    // Helper functions for init()
    static std::shared_ptr<const DeviceBlockPoolConfig>
         normalizeConfig(const std::shared_ptr<const DeviceBlockPoolConfig>& config);
    void initializeCacheBuffer();
    void initializeSplitCacheBuffer();
    void initializePinnedCpuBuffer();
    void initializeDeviceMallocBuffer(size_t size_bytes);
    void initializeLayerMappings();
    void initializeLayoutStrategies();

    struct PhysicalLayout {
        MemoryLayoutConfig device;
        MemoryLayoutConfig host;
        uint32_t           device_block_count  = 0;
        uint32_t           device_spare_blocks = 0;
    };
    using BlockLocation = std::tuple<const MemoryLayoutStrategy&, int, BlockIdxType>;
    BlockLocation locateLogicalBlock(int layer_id, BlockIdxType block) const;

    // Helper functions for initializeLayoutStrategies()
    void          processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor, size_t& global_layer_begin);
    torch::Tensor createTensor(const torch::Tensor& full_tensor,
                               int64_t              offset,
                               int64_t              size,
                               size_t               layout_idx,
                               const std::string&   tensor_type);
    void          initializeLayoutStrategy(size_t                    layout_idx,
                                           const MemoryLayoutConfig& layout_cfg,
                                           torch::Tensor&            kv_cache_tensor,
                                           torch::Tensor&            kv_scale_tensor);
    void processLayerTensors(size_t layout_idx, const MemoryLayoutConfig& layout_cfg, size_t& global_layer_begin);

    std::unique_ptr<MemoryLayoutStrategy>
         makeSplitLayoutStrategy(const MemoryLayoutConfig& layout, const torch::Tensor& arena, size_t layout_idx);
    void processSplitLayout(size_t layout_idx, size_t global_layer_begin);

    // Helper functions for regUserMr/deregUserMr
    void registerUserMrForBuffer(std::shared_ptr<MemoryUtil> memory_util,
                                 size_t                      layout_idx,
                                 void*                       base_ptr,
                                 size_t                      bytes,
                                 size_t                      stride_bytes,
                                 bool                        gpu,
                                 const std::string&          buffer_type);
    void deregisterUserMrForBuffer(std::shared_ptr<MemoryUtil> memory_util,
                                   size_t                      layout_idx,
                                   void*                       base_ptr,
                                   bool                        gpu,
                                   const std::string&          buffer_type);

private:
    static constexpr size_t                          kRefTypeCount = static_cast<size_t>(DeviceBlockRefType::COUNT);
    std::vector<uint32_t>                            refcounts_;
    std::array<std::vector<uint32_t>, kRefTypeCount> refcounts_by_type_;
    std::array<size_t, kRefTypeCount>                referenced_block_counts_{};

    torch::Tensor                                      cache_aligned_buffer_;
    torch::Tensor                                      host_aligned_buffer_;
    std::vector<PhysicalLayout>                        physical_layouts_;
    std::vector<std::unique_ptr<MemoryLayoutStrategy>> host_layout_strategies_;
    std::vector<LayerBackingView>                      layer_backing_views_;
    torch::Tensor                                      block_generations_;
    int64_t*                                           block_generations_ptr_      = nullptr;
    BlockIdxType                                       device_block_boundary_      = 0;
    size_t                                             row_split_tokens_per_block_ = 0;
    void*                                              cache_base_ptr_{nullptr};
    size_t                                             allocation_size_bytes_{0};
    bool                                               uses_dedicated_device_allocation_{false};

    bool                        kvcache_reg_mr_  = false;
    int64_t                     mr_cost_time_ms_ = 0;
    std::shared_ptr<CacheStore> cache_store_;

    std::vector<std::unique_ptr<MemoryLayoutStrategy>> layout_strategies_;
    std::vector<std::pair<int, int>>                   global_layer_to_local_;
    std::vector<torch::Tensor>                         global_layer_kv_tensors_;
    std::vector<torch::Tensor>                         global_layer_kv_scale_tensors_;
};

}  // namespace rtp_llm
