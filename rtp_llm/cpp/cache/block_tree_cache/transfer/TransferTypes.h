#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

struct TreeNode;

enum class TransferStatus {
    OK,
    INVALID_ARGS,
    DEVICE_IO_ERROR,
    DISK_IO_ERROR,
    RESOURCE_EXHAUSTED,
    CRC_MISMATCH,
};

// Diagnostic categories for the existing cache-copy metric. They do not change
// the business error or recovery policy. Higher values take precedence when a
// logical copy contains multiple failures; confirmed CRC evidence survives RPC
// failures on other ranks.
enum class CacheCopyError : uint8_t {
    NONE,
    COPY_FAILED,
    INVALID_REQUEST,
    IO_FAILED,
    RPC_FAILED,
    CRC_COMPUTE_FAILED,
    CRC_MISMATCH,
};

inline const char* cacheCopyErrorName(CacheCopyError error) {
    switch (error) {
        case CacheCopyError::NONE:
            return "NONE";
        case CacheCopyError::COPY_FAILED:
            return "COPY_FAILED";
        case CacheCopyError::INVALID_REQUEST:
            return "INVALID_REQUEST";
        case CacheCopyError::IO_FAILED:
            return "IO_FAILED";
        case CacheCopyError::RPC_FAILED:
            return "RPC_FAILED";
        case CacheCopyError::CRC_COMPUTE_FAILED:
            return "CRC_COMPUTE_FAILED";
        case CacheCopyError::CRC_MISMATCH:
            return "CRC_MISMATCH";
    }
    return "COPY_FAILED";
}

struct DeviceHostCopyOptions {
    size_t staged_sm_min_tile_count{16};
    size_t staged_sm_min_bytes{64 * 1024};
    bool   staged_sm_copy_enabled{true};
    bool   cuda_batch_copy_enabled{true};
};

struct HostBufferView {
    void*  base{nullptr};
    size_t payload_bytes{0};
    // Safe access range from base; disk I/O may require capacity beyond the logical payload.
    size_t capacity_bytes{0};
};

inline bool
isValidHostBufferView(const HostBufferView& view, size_t required_payload_bytes, size_t required_access_bytes) {
    return view.base != nullptr && view.payload_bytes <= view.capacity_bytes
           && view.payload_bytes >= required_payload_bytes && view.capacity_bytes >= required_access_bytes;
}

// Unified operation descriptor for load, eviction, and transfer execution.
// Business-only fields are ignored by transfer executors and RPC conversion.
struct TransferDescriptor {
    TransferDescriptor() = default;
    TransferDescriptor(TreeNode*                 node,
                       size_t                    group_set_id,
                       size_t                    path_index,
                       Tier                      source_tier,
                       Tier                      target_tier,
                       std::vector<BlockIdxType> source_blocks):
        node(node),
        group_set_id(group_set_id),
        path_index(path_index),
        source_tier(source_tier),
        target_tier(target_tier),
        source_blocks(std::move(source_blocks)) {}

    static TransferDescriptor
    deviceToHost(size_t group_set_id, std::vector<BlockIdxType> device_blocks, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DEVICE;
        desc.target_tier   = Tier::HOST;
        desc.source_blocks = std::move(device_blocks);
        desc.target_blocks = {host_block};
        return desc;
    }

    static TransferDescriptor
    hostToDevice(size_t group_set_id, BlockIdxType host_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::HOST;
        desc.target_tier   = Tier::DEVICE;
        desc.source_blocks = {host_block};
        desc.target_blocks = std::move(device_blocks);
        return desc;
    }

    static TransferDescriptor hostToDisk(size_t group_set_id, BlockIdxType host_block, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::HOST;
        desc.target_tier   = Tier::DISK;
        desc.source_blocks = {host_block};
        desc.target_blocks = {disk_block};
        return desc;
    }

    static TransferDescriptor diskToHost(size_t group_set_id, BlockIdxType disk_block, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DISK;
        desc.target_tier   = Tier::HOST;
        desc.source_blocks = {disk_block};
        desc.target_blocks = {host_block};
        return desc;
    }

    static TransferDescriptor
    deviceToDisk(size_t group_set_id, std::vector<BlockIdxType> device_blocks, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DEVICE;
        desc.target_tier   = Tier::DISK;
        desc.source_blocks = std::move(device_blocks);
        desc.target_blocks = {disk_block};
        return desc;
    }

    static TransferDescriptor
    diskToDevice(size_t group_set_id, BlockIdxType disk_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DISK;
        desc.target_tier   = Tier::DEVICE;
        desc.source_blocks = {disk_block};
        desc.target_blocks = std::move(device_blocks);
        return desc;
    }

    bool needsTransfer() const {
        return target_tier != Tier::NONE && source_tier != target_tier;
    }

    const std::vector<BlockIdxType>& blocksAt(Tier tier) const {
        return source_tier == tier ? source_blocks : target_blocks;
    }

    BlockIdxType singleBlockAt(Tier tier) const {
        return blocksAt(tier)[0];
    }

    bool isExecutable() const {
        const bool supported_direction =
            (source_tier == Tier::DEVICE && (target_tier == Tier::HOST || target_tier == Tier::DISK))
            || (source_tier == Tier::HOST && (target_tier == Tier::DEVICE || target_tier == Tier::DISK))
            || (source_tier == Tier::DISK && (target_tier == Tier::DEVICE || target_tier == Tier::HOST));
        return supported_direction && endpointResolved(source_tier, source_blocks)
               && endpointResolved(target_tier, target_blocks);
    }

    std::string debugString() const {
        return "TransferDescriptor{group_set_id=" + std::to_string(group_set_id)
               + ", direction=" + tierName(source_tier) + "->" + tierName(target_tier) + ", source_blocks=["
               + blocksDebugString(source_blocks) + "], target_blocks=[" + blocksDebugString(target_blocks) + "]}";
    }

    void markCorrupted() const {
        markCopyError(CacheCopyError::CRC_MISMATCH);
    }
    bool corrupted() const {
        return hasCopyError(CacheCopyError::CRC_MISMATCH);
    }

    void markCrcComputeFailed() const {
        markCopyError(CacheCopyError::CRC_COMPUTE_FAILED);
    }
    bool crcComputeFailed() const {
        return hasCopyError(CacheCopyError::CRC_COMPUTE_FAILED);
    }

    void markCopyError(CacheCopyError error) const {
        if (error != CacheCopyError::NONE) {
            diagnostic_flags_->fetch_or(uint8_t{1} << static_cast<uint8_t>(error), std::memory_order_release);
        }
    }
    CacheCopyError copyError() const {
        const auto flags = diagnostic_flags_->load(std::memory_order_acquire);
        for (uint8_t value = static_cast<uint8_t>(CacheCopyError::CRC_MISMATCH); value != 0; --value) {
            if (flags & (uint8_t{1} << value)) {
                return static_cast<CacheCopyError>(value);
            }
        }
        return CacheCopyError::NONE;
    }

    // Null for node-independent descriptors, including DEVICE-source reuse
    // whose blocks may outlive eviction of the originating tree node.
    TreeNode*                 node{nullptr};
    size_t                    group_set_id{0};
    size_t                    path_index{0};
    Tier                      source_tier{Tier::NONE};
    Tier                      target_tier{Tier::NONE};
    std::vector<BlockIdxType> source_blocks;
    std::vector<BlockIdxType> target_blocks;

private:
    bool hasCopyError(CacheCopyError error) const {
        return diagnostic_flags_->load(std::memory_order_acquire) & (uint8_t{1} << static_cast<uint8_t>(error));
    }
    // Copies across batches and disk staging refer to the same source record.
    std::shared_ptr<std::atomic<uint8_t>> diagnostic_flags_{std::make_shared<std::atomic<uint8_t>>(0)};

    static bool endpointResolved(Tier tier, const std::vector<BlockIdxType>& blocks) {
        if (tier != Tier::DEVICE && tier != Tier::HOST && tier != Tier::DISK) {
            return false;
        }
        if (blocks.empty() || (tier != Tier::DEVICE && blocks.size() != 1)) {
            return false;
        }
        return std::none_of(blocks.begin(), blocks.end(), [](BlockIdxType block) { return isNullBlockIdx(block); });
    }

    static std::string blocksDebugString(const std::vector<BlockIdxType>& blocks) {
        std::string result;
        for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            result += (block_index == 0 ? "" : ",") + std::to_string(blocks[block_index]);
        }
        return result;
    }
};

inline void recordTransferError(const std::vector<TransferDescriptor>& descriptors, TransferStatus status) {
    if (status == TransferStatus::OK) {
        return;
    }
    const auto error = status == TransferStatus::INVALID_ARGS  ? CacheCopyError::INVALID_REQUEST :
                       status == TransferStatus::DISK_IO_ERROR ? CacheCopyError::IO_FAILED :
                                                                 CacheCopyError::COPY_FAILED;
    for (const auto& descriptor : descriptors) {
        // The CRC service identifies the individual bad records. A batch-level
        // failure must never mark all its otherwise healthy records corrupted.
        descriptor.markCopyError(error);
    }
}

class TransferTask {
public:
    using Clock = std::chrono::steady_clock;

    TransferTask(std::vector<TransferDescriptor> descriptors, std::chrono::milliseconds timeout):
        descriptors_(std::move(descriptors)), deadline_(Clock::now() + timeout) {}

    const std::vector<TransferDescriptor>& descriptors() const {
        return descriptors_;
    }

    Clock::time_point deadline() const {
        return deadline_;
    }

    void addDescriptor(TransferDescriptor descriptor) {
        descriptors_.push_back(std::move(descriptor));
    }

    // Business task builders may fill or resolve descriptors before admission.
    std::vector<TransferDescriptor>& mutableDescriptorsForPreparation() {
        return descriptors_;
    }

    std::optional<std::chrono::milliseconds> remainingTimeout() const {
        const auto remaining = deadline_ - Clock::now();
        if (remaining <= Clock::duration::zero()) {
            return std::nullopt;
        }
        return std::chrono::ceil<std::chrono::milliseconds>(remaining);
    }

    bool expired() const {
        return !remainingTimeout().has_value();
    }

    std::vector<TransferDescriptor> corruptedDescriptors() const {
        std::vector<TransferDescriptor> result;
        for (const auto& descriptor : descriptors_) {
            if (descriptor.corrupted()) {
                result.push_back(descriptor);
            }
        }
        return result;
    }

    TransferTask subtask(std::vector<TransferDescriptor> descriptors) const {
        return TransferTask(std::move(descriptors), deadline_);
    }

private:
    TransferTask(std::vector<TransferDescriptor> descriptors, Clock::time_point deadline):
        descriptors_(std::move(descriptors)), deadline_(deadline) {}

    std::vector<TransferDescriptor> descriptors_;
    Clock::time_point               deadline_;
};

}  // namespace rtp_llm
