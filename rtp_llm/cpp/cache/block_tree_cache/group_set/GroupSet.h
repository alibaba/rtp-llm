#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct TreeNode;

struct MultiNodeResource {
    size_t                                                       group_set_id{0};
    Tier                                                         tier{Tier::DEVICE};
    std::vector<std::pair<TreeNode*, std::vector<BlockIdxType>>> node_blocks;
};

class MatchValidator {
public:
    virtual ~MatchValidator()                               = default;
    virtual bool validate(const GroupSetResource& resource) = 0;
};

// Layout of one packed HOST/DISK record for one GroupSet block on this rank.
// The payload combines one block from each member pool, including all of that
// member's layer buffers (KV and any scales). Factory computes this plan once
// and shares it with GroupSet, pool creation and memory/disk capacity budgeting.
// All sizes below are bytes per record, not the size of the whole pool.
//
// payload_bytes: Packed bytes before CRC; 0 derives logical sizes from topology, otherwise uses exact physical sizes.
// storage_bytes: Record bytes (payload or alignUp(payload_bytes + 4, 16) with CRC); 0 uses the resolved payload size.
// pool_stride: Distance between HOST/DISK slots, aligned to 4096 for direct disk I/O or 16 otherwise.
// crc_enabled: Whether transfers encode and verify a CRC32C checksum for each record.
//
// Example: member A contributes 2048 bytes and member B contributes 2048 bytes.
// Packing their buffers produces a 4096-byte payload (diagrams not to scale).
//
// Field order: {payload_bytes, storage_bytes, pool_stride, crc_enabled}.
// CRC off: BackingLayout{4096, 4096, 4096, false}
//   0                          2048                       4096
//   +--------------------------+--------------------------+ next record
//   |     member A: 2048 B     |     member B: 2048 B     |
//   +--------------------------+--------------------------+
//   payload_bytes = storage_bytes = pool_stride = 4096
//
// CRC on, HOST only or buffered disk I/O: BackingLayout{4096, 4112, 4112, true}
//   0                     4096          4108        4112
//   +---------------------+-------------+-----------+
//   |   payload: 4096 B   |  pad: 12 B  |  CRC: 4 B | next record
//   +---------------------+-------------+-----------+
//   |<-- payload_bytes -->|
//   |<-------- storage_bytes = pool_stride -------->|
//
// CRC on, direct disk I/O (both HOST/DISK slots): BackingLayout{4096, 4112, 8192, true}
//   0                     4096          4108        4112                    8192
//   +---------------------+-------------+-----------+-----------------------+
//   |   payload: 4096 B   |  pad: 12 B  |  CRC: 4 B |  pool padding: 4080 B | next record
//   +---------------------+-------------+-----------+-----------------------+
//   |<-- payload_bytes -->|
//   |<--------------- storage_bytes --------------->|
//   |<---------------------------- pool_stride ---------------------------->|
// CRC32C covers only the payload. The 4-byte checksum is stored at the END of
// the 16-byte-aligned encoded record; pool padding lies outside that record.
struct BackingLayout {
    size_t payload_bytes{0};
    size_t storage_bytes{0};
    size_t pool_stride{0};
    bool   crc_enabled{false};
};

class GroupSet {
public:
    GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
             std::shared_ptr<HostBlockPool>  host_pool,
             BlockTreeDiskBlockPoolPtr       disk_pool);

    virtual ~GroupSet() = default;

    void initialize(size_t                               group_set_id,
                    std::shared_ptr<const CacheTopology> topology,
                    std::vector<std::string>             group_tags,
                    BackingLayout                        backing_layout = {});

    size_t groupSetId() const {
        return group_set_id_;
    }
    const std::vector<std::string>& groupTags() const {
        return group_tags_;
    }
    std::shared_ptr<const CacheTopology> topologyPtr() const {
        return topology_;
    }
    const GroupBase& group(std::string_view tag) const {
        return topology_->group(tag);
    }
    size_t payloadBytes() const {
        return payload_bytes_;
    }
    bool usesPhysicalPayloadGeometry() const {
        return uses_physical_payload_geometry_;
    }
    // Storage adds a per-backing footer after the physical payload.
    size_t storageBytes() const {
        return storage_bytes_;
    }
    bool crcEnabled() const {
        return enable_crc_;
    }
    CacheGroupType groupType() const {
        return group(groupTags().front()).policy.group_type;
    }
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual size_t computeReuseBlockCount(size_t matched_block_count) const = 0;

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    std::shared_ptr<HostBlockPool> hostPool() const {
        return host_pool_;
    }
    std::shared_ptr<BlockTreeDiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    void referenceBlocks(const MultiNodeResource& resource) const;
    void unreferenceBlocks(const MultiNodeResource& resource) const;
    void referenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const;
    void unreferenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const;

    BlockIdxType               allocateSingleBlock(Tier tier, BlockTreeRefType ref_type);
    std::optional<BlockIdList> allocateBlocks(size_t n, Tier tier, BlockTreeRefType ref_type);
    void                       releaseBlocks(Tier tier, const BlockIdList& blocks, BlockTreeRefType ref_type) const;
    void                       releaseSingleBlock(Tier tier, BlockIdxType block, BlockTreeRefType ref_type) const;

private:
    std::vector<DeviceBlockPoolPtr>         device_pools_;
    std::shared_ptr<HostBlockPool>          host_pool_;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool_;
    size_t                                  group_set_id_{0};
    std::shared_ptr<const CacheTopology>    topology_;
    std::vector<std::string>                group_tags_;
    size_t                                  payload_bytes_{0};
    size_t                                  storage_bytes_{0};
    bool                                    enable_crc_{false};
    bool                                    uses_physical_payload_geometry_{false};
};

using GroupSetPtr = std::shared_ptr<GroupSet>;

}  // namespace rtp_llm
