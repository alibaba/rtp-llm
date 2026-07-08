#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

class DeviceBlockPool;
class DiskBlockPool;
class HostBlockPool;
enum class BlockIOStatus;

class CopyEngine {
public:
    CopyEngine(const std::vector<ComponentGroupPtr>& component_groups, const std::vector<Component>& components);
    CopyEngine() = delete;
    ~CopyEngine() = default;

    // Descriptor-based facade. Executes synchronously, returns completed handle.
    // TODO: change to async later
    TransferHandle submit(const TransferDescriptor& desc);

    static size_t computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots);

private:
    struct ResolvedComponentLayout {
        int                                  component_index{-1};
        int                                  device_pool_index{-1};
        DeviceBlockPool*                     device_pool{nullptr};
        std::vector<MemoryBlockLayerTagSlot> layer_slots;
    };

    struct ResolvedGroupLayout {
        int                                  component_group_id{-1};
        HostBlockPool*                       host_pool{nullptr};
        DiskBlockPool*                       disk_pool{nullptr};
        std::vector<ResolvedComponentLayout> components;
    };

    static CopyResult makeCopyResult(uint64_t request_id,
                                     CopyStatus status,
                                     size_t completed_entries,
                                     size_t failed_entries);

    CopyResult execute(const TransferDescriptor& desc, uint64_t request_id);

    void completeRequest(const std::shared_ptr<TransferHandle::State>& state, CopyResult result);

    CopyStatus resolveGroupPools(int component_group_id, ResolvedGroupLayout* out) const;
    CopyStatus resolveGroupLayout(int component_group_id, ResolvedGroupLayout* out) const;
    CopyStatus validateDeviceHostLayout(const ResolvedGroupLayout& layout) const;

    CopyStatus deviceToHost(const std::vector<BlockIdxType>& device_blocks,
                            BlockIdxType                     host_block,
                            const ResolvedGroupLayout&       layout) const;
    CopyStatus hostToDevice(BlockIdxType                     host_block,
                            const std::vector<BlockIdxType>& device_blocks,
                            const ResolvedGroupLayout&       layout) const;
    CopyStatus hostToDisk(BlockIdxType   host_block,
                          BlockIdxType   disk_block,
                          HostBlockPool& host_pool,
                          DiskBlockPool& disk_pool) const;
    CopyStatus diskToHost(BlockIdxType   disk_block,
                          BlockIdxType   host_block,
                          HostBlockPool& host_pool,
                          DiskBlockPool& disk_pool) const;

    struct DeviceHostCopyTile {
        void*  host_addr{nullptr};
        void*  device_addr{nullptr};
        size_t bytes{0};
    };

    struct HostZeroTile {
        void*  host_addr{nullptr};
        size_t bytes{0};
    };

    static bool        isDeviceHostTransfer(Tier source_tier, Tier target_tier);
    static bool        validAllocatedHostBlock(HostBlockPool& host_pool, BlockIdxType host_block);
    static bool        validAllocatedDiskBlock(DiskBlockPool& disk_pool, BlockIdxType disk_block);
    static bool        validDeviceBlock(DeviceBlockPool& device_pool, BlockIdxType device_block);
    static CopyStatus  blockIOStatusToCopyStatus(BlockIOStatus status);
    static const char* blockIOStatusName(BlockIOStatus status);
    static bool        hasAnyLayerSlot(const std::vector<ResolvedComponentLayout>& layouts);
    static size_t      computeLayoutsBlockSize(const std::vector<ResolvedComponentLayout>& layouts);
    static void        executeDeviceHostCopyTiles(const std::vector<DeviceHostCopyTile>& tiles, bool device_to_host);

    std::vector<std::shared_ptr<const ComponentGroup>> component_groups_;
    std::vector<Component>                             components_;
    std::atomic<uint64_t> next_request_id_{1};
};

using CopyEnginePtr = std::shared_ptr<CopyEngine>;

}  // namespace rtp_llm
