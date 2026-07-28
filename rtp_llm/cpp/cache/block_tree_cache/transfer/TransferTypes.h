#pragma once

#include <cstddef>
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

struct DeviceHostCopyOptions {
    size_t staged_sm_min_tile_count{16};
    size_t staged_sm_min_bytes{64 * 1024};
    bool   staged_sm_copy_enabled{false};
    bool   cuda_batch_copy_enabled{true};
};

struct TransferDescriptor {
    static TransferDescriptor
    deviceToHost(size_t group_set_id, std::vector<BlockIdxType> device_blocks, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DEVICE;
        desc.target_tier   = Tier::HOST;
        desc.device_blocks = std::move(device_blocks);
        desc.host_block    = host_block;
        return desc;
    }

    static TransferDescriptor
    hostToDevice(size_t group_set_id, BlockIdxType host_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::HOST;
        desc.target_tier   = Tier::DEVICE;
        desc.host_block    = host_block;
        desc.device_blocks = std::move(device_blocks);
        return desc;
    }

    static TransferDescriptor hostToDisk(size_t group_set_id, BlockIdxType host_block, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.group_set_id = group_set_id;
        desc.source_tier  = Tier::HOST;
        desc.target_tier  = Tier::DISK;
        desc.host_block   = host_block;
        desc.disk_block   = disk_block;
        return desc;
    }

    static TransferDescriptor diskToHost(size_t group_set_id, BlockIdxType disk_block, BlockIdxType host_block) {
        TransferDescriptor desc;
        desc.group_set_id = group_set_id;
        desc.source_tier  = Tier::DISK;
        desc.target_tier  = Tier::HOST;
        desc.disk_block   = disk_block;
        desc.host_block   = host_block;
        return desc;
    }

    size_t group_set_id{0};
    Tier   source_tier{Tier::NONE};
    Tier   target_tier{Tier::NONE};

    // DEVICE -> HOST: source. HOST -> DEVICE: target.
    std::vector<BlockIdxType> device_blocks;

    // DEVICE -> HOST: target. HOST -> DEVICE / HOST -> DISK: source. DISK -> HOST: target.
    BlockIdxType host_block{NULL_BLOCK_IDX};

    // HOST -> DISK: target. DISK -> HOST: source.
    BlockIdxType disk_block{NULL_BLOCK_IDX};
    std::string  storage_key;
};

}  // namespace rtp_llm
