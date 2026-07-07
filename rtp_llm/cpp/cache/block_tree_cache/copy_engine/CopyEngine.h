#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TransferDescriptor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

namespace rtp_llm {

// Function that resolves a (layer_id, device_block_idx) to a raw buffer pointer + size.
// For GPU: wraps DeviceBlockPool::blockBuffers.
// For testing: wraps a simple CPU buffer map.
using DeviceBufferResolver = std::function<BlockInfo(int layer_id, BlockIdxType device_block_idx)>;

enum class TransferPriority {
    REQUEST_LOAD,
    DEVICE_EVICTION,
    HOST_EVICTION,
    PREFETCH,
    BACKGROUND,
};

struct TransferSubmitOptions {
    TransferPriority priority{TransferPriority::BACKGROUND};
    bool             require_all_or_none{true};
};

enum class CopyStatusCode {
    OK,
    INVALID_DESCRIPTOR,
    INVALID_BLOCK,
    SIZE_MISMATCH,
    ALIGNMENT_ERROR,
    DEVICE_IO_ERROR,
    HOST_IO_ERROR,
    DISK_IO_ERROR,
    STAGING_EXHAUSTED,
    PARTIAL_FAILURE,
    STALE_COMPLETION,
};

struct CopyResult {
    uint64_t       request_id{0};
    CopyStatusCode status{CopyStatusCode::OK};
    size_t         completed_entries{0};
    size_t         failed_entries{0};
    std::string    error_message;

    bool ok() const {
        return status == CopyStatusCode::OK;
    }
};

using CopyCompletionCallback = std::function<void(const CopyResult&)>;

class TransferHandle {
public:
    TransferHandle() = default;

    uint64_t requestId() const;
    void     wait() const;
    bool     done() const;
    CopyResult result() const;
    void       onComplete(CopyCompletionCallback callback) const;
    bool       valid() const {
        return state_ != nullptr;
    }

private:
    struct State;

    explicit TransferHandle(std::shared_ptr<State> state): state_(std::move(state)) {}

    std::shared_ptr<State> state_;

    friend class CopyEngine;
};

struct CopyEngineTransferResources {
    DeviceBufferResolver device_buffer_resolver;
    std::function<std::vector<MemoryBlockLayerTagSlot>(int component_group_id)>  layer_slots_resolver;
    std::function<std::shared_ptr<HostBlockPool>(int component_group_id)>        host_pool_resolver;
    std::function<std::shared_ptr<DiskBlockPool>(int component_group_id)>        disk_pool_resolver;
};

// CopyEngine: stateless data-movement utility between Device/Host/Disk tiers,
// plus a synchronous submit() facade (TransferHandle/CopyResult).
//
// D2H (deviceToHost): packs multiple device blocks into one host block,
//   using MemoryBlockLayerTagSlot layout to compute byte offsets.
// H2D (hostToDevice): unpacks one host block into multiple device blocks.
// H2Disk (hostToDisk): writes host block to a disk block.
// Disk2H (diskToHost): reads disk block into a host block.
//
// Pool ownership: CopyEngine does NOT own any pool.  HostBlockPool and
// DiskBlockPool are owned by BlockTreeCache (L2/L3 tier resources) and
// provided through CopyEngineTransferResources or compatibility helper parameters.
class CopyEngine {
public:
    CopyEngine()  = default;
    explicit CopyEngine(CopyEngineTransferResources resources);
    ~CopyEngine() = default;

    // Current facade executes synchronously and returns an already completed handle.
    TransferHandle submit(const TransferDescriptor& desc, TransferSubmitOptions options = {});

    void setTransferResources(CopyEngineTransferResources resources);

    // TODO: Remove these public helpers after BlockTreeCache call sites move to submit().
    // Compatibility byte-copy helpers. New callers should prefer submit().
    bool deviceToHost(const std::vector<BlockIdxType>&            device_blocks,
                      BlockIdxType                                host_block,
                      const std::vector<MemoryBlockLayerTagSlot>& slots,
                      const DeviceBufferResolver&                 resolver,
                      HostBlockPool&                              host_pool);

    bool hostToDevice(BlockIdxType                                host_block,
                      const std::vector<BlockIdxType>&            device_blocks,
                      const std::vector<MemoryBlockLayerTagSlot>& slots,
                      const DeviceBufferResolver&                 resolver,
                      HostBlockPool&                              host_pool);

    // ---- Host <-> Disk ----

    // Write host block to disk block.
    bool hostToDisk(BlockIdxType host_block, BlockIdxType disk_block, HostBlockPool& host_pool, DiskBlockPool& disk_pool);

    // Read disk block into host block.
    bool diskToHost(BlockIdxType disk_block, BlockIdxType host_block, HostBlockPool& host_pool, DiskBlockPool& disk_pool);

    static size_t computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots);

private:
    CopyResult executeTransfer(const TransferDescriptor&     desc,
                               const TransferSubmitOptions& options,
                               uint64_t                     request_id);
    void       completeRequest(const std::shared_ptr<TransferHandle::State>& state, CopyResult result);

    std::vector<MemoryBlockLayerTagSlot> resolveLayerSlots(int component_group_id) const;
    std::shared_ptr<HostBlockPool>       resolveHostPool(int component_group_id) const;
    std::shared_ptr<DiskBlockPool>       resolveDiskPool(int component_group_id) const;
    DeviceBufferResolver                 resolveDeviceBufferResolver() const;

    std::atomic<uint64_t> next_request_id_{1};

    mutable std::mutex                                             resources_mutex_;
    CopyEngineTransferResources                                    resources_;
};

using CopyEnginePtr = std::shared_ptr<CopyEngine>;

}  // namespace rtp_llm
