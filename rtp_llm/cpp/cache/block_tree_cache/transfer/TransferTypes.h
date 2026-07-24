#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"

namespace rtp_llm {

enum class TransferStatus {
    OK,
    INVALID_ARGS,
    DEVICE_IO_ERROR,
    DISK_IO_ERROR,
};

using TransferCompletionCallback = std::function<void(TransferStatus)>;

struct DeviceHostCopyOptions {
    size_t staged_sm_min_tile_count{16};
    size_t staged_sm_min_bytes{64 * 1024};
    bool   staged_sm_copy_enabled{false};
    bool   cuda_batch_copy_enabled{true};
};

class TransferHandle {
public:
    TransferHandle() = default;

    static TransferHandle completed(TransferStatus status, uint64_t request_id = 0);

    uint64_t       requestId() const;
    void           wait() const;
    bool           done() const;
    TransferStatus status() const;
    bool           ok() const {
        return status() == TransferStatus::OK;
    }
    void onComplete(TransferCompletionCallback callback) const;
    bool valid() const {
        return state_ != nullptr;
    }

private:
    struct State;

    explicit TransferHandle(std::shared_ptr<State> state): state_(std::move(state)) {}

    std::shared_ptr<State> state_;

    friend class PerRankBlockTransferEngine;
};

struct TransferDescriptor {
    static TransferDescriptor
    deviceToHost(int component_group_id, std::vector<BlockIdxType> device_blocks, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.component_group_id = component_group_id;
        desc.source_tier        = Tier::DEVICE;
        desc.target_tier        = Tier::HOST;
        desc.device_blocks      = std::move(device_blocks);
        desc.host_block         = host_block;
        return desc;
    }

    static TransferDescriptor
    hostToDevice(int component_group_id, BlockIdxType host_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.component_group_id = component_group_id;
        desc.source_tier        = Tier::HOST;
        desc.target_tier        = Tier::DEVICE;
        desc.host_block         = host_block;
        desc.device_blocks      = std::move(device_blocks);
        return desc;
    }

    static TransferDescriptor hostToDisk(int component_group_id, BlockIdxType host_block, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.component_group_id = component_group_id;
        desc.source_tier        = Tier::HOST;
        desc.target_tier        = Tier::DISK;
        desc.host_block         = host_block;
        desc.disk_block         = disk_block;
        return desc;
    }

    static TransferDescriptor diskToHost(int component_group_id, BlockIdxType disk_block, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.component_group_id = component_group_id;
        desc.source_tier        = Tier::DISK;
        desc.target_tier        = Tier::HOST;
        desc.disk_block         = disk_block;
        desc.host_block         = host_block;
        return desc;
    }

    int  component_group_id{-1};
    Tier source_tier{Tier::NONE};
    Tier target_tier{Tier::NONE};

    // DEVICE -> HOST: source. HOST -> DEVICE: target.
    std::vector<BlockIdxType> device_blocks;

    // DEVICE -> HOST: target. HOST -> DEVICE / HOST -> DISK: source. DISK -> HOST: target.
    BlockIdxType host_block{NULL_BLOCK_IDX};

    // HOST -> DISK: target. DISK -> HOST: source.
    BlockIdxType disk_block{NULL_BLOCK_IDX};
    std::string  storage_key;
};

}  // namespace rtp_llm
