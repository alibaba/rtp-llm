#pragma once

#include <algorithm>
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
    RESOURCE_EXHAUSTED,
};

struct DeviceHostCopyOptions {
    size_t staged_sm_min_tile_count{16};
    size_t staged_sm_min_bytes{64 * 1024};
    bool   staged_sm_copy_enabled{false};
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

    static TransferDescriptor
    deviceToDisk(size_t group_set_id, std::vector<BlockIdxType> device_blocks, BlockIdxType disk_block) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DEVICE;
        desc.target_tier   = Tier::DISK;
        desc.device_blocks = std::move(device_blocks);
        desc.disk_block    = disk_block;
        return desc;
    }

    static TransferDescriptor
    diskToDevice(size_t group_set_id, BlockIdxType disk_block, std::vector<BlockIdxType> device_blocks) {
        TransferDescriptor desc;
        desc.group_set_id  = group_set_id;
        desc.source_tier   = Tier::DISK;
        desc.target_tier   = Tier::DEVICE;
        desc.disk_block    = disk_block;
        desc.device_blocks = std::move(device_blocks);
        return desc;
    }

    // A descriptor is valid only when every endpoint of its direction is resolved.
    bool isValid() const {
        const bool device_resolved =
            !device_blocks.empty() && std::none_of(device_blocks.begin(), device_blocks.end(), [](BlockIdxType block) {
                return isNullBlockIdx(block);
            });
        const bool host_resolved = !isNullBlockIdx(host_block);
        const bool disk_resolved = !isNullBlockIdx(disk_block);
        if ((source_tier == Tier::DEVICE && target_tier == Tier::HOST)
            || (source_tier == Tier::HOST && target_tier == Tier::DEVICE)) {
            return device_resolved && host_resolved;
        }
        if ((source_tier == Tier::HOST && target_tier == Tier::DISK)
            || (source_tier == Tier::DISK && target_tier == Tier::HOST)) {
            return device_blocks.empty() && host_resolved && disk_resolved;
        }
        if ((source_tier == Tier::DEVICE && target_tier == Tier::DISK)
            || (source_tier == Tier::DISK && target_tier == Tier::DEVICE)) {
            return device_resolved && disk_resolved;
        }
        return false;
    }

    std::string debugString() const {
        std::string device_blocks_str;
        for (size_t i = 0; i < device_blocks.size(); ++i) {
            device_blocks_str += (i == 0 ? "" : ",") + std::to_string(device_blocks[i]);
        }
        return "TransferDescriptor{group_set_id=" + std::to_string(group_set_id) + ", direction="
               + tierName(source_tier) + "->" + tierName(target_tier) + ", device_blocks=[" + device_blocks_str
               + "], host_block=" + std::to_string(host_block) + ", disk_block=" + std::to_string(disk_block) + "}";
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
};

}  // namespace rtp_llm
