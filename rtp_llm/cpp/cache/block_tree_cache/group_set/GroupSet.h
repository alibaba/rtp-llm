#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct TreeNode;
struct CacheTopology;
struct GroupBase;
class DeviceBlockPool;
class DiskBlockPool;
using DeviceBlockPoolPtr = std::shared_ptr<DeviceBlockPool>;
using DiskBlockPoolPtr = std::shared_ptr<DiskBlockPool>;

struct GroupSetPools {
    std::vector<std::shared_ptr<IBlockPool>> device;
    std::shared_ptr<HostBlockPool> host;
    std::shared_ptr<IBlockPool> disk;
};

struct GroupSetGeometry {
    CacheGroupType type{CacheGroupType::FULL};
    size_t page_tokens{1};
    size_t window_tokens{0};
};

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

class GroupSet {
public:
    explicit GroupSet(GroupSetPools pools);
    GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
             std::shared_ptr<HostBlockPool>  host_pool,
             DiskBlockPoolPtr                disk_pool);

    virtual ~GroupSet() = default;

    void initialize(size_t                               group_set_id,
                    std::shared_ptr<const CacheTopology> topology,
                    std::vector<std::string>             group_tags,
                    BackingLayout                        backing_layout = {});
    void initialize(size_t group_set_id, GroupSetGeometry geometry,
                    std::vector<std::string> group_tags, BackingLayout backing_layout = {});

    size_t groupSetId() const {
        return group_set_id_;
    }
    const std::vector<std::string>& groupTags() const {
        return group_tags_;
    }
    std::shared_ptr<const CacheTopology> topologyPtr() const {
        return topology_;
    }
    const GroupBase& group(std::string_view tag) const;
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
    size_t storageStrideBytes() const {
        return storage_stride_bytes_;
    }
    bool crcEnabled() const {
        return enable_crc_;
    }
    CacheGroupType groupType() const {
        return geometry_.type;
    }
    size_t pageTokens() const {
        return geometry_.page_tokens;
    }
    size_t windowTokens() const {
        return geometry_.window_tokens;
    }
    const GroupSetPools& pools() const {
        return pools_;
    }
    virtual std::unique_ptr<MatchValidator> createMatchValidator() = 0;

    virtual size_t computeReuseBlockCount(size_t matched_block_count) const = 0;

    const std::vector<DeviceBlockPoolPtr>& devicePools() const {
        return device_pools_;
    }
    std::shared_ptr<HostBlockPool> hostPool() const {
        return pools_.host;
    }
    std::shared_ptr<DiskBlockPool> diskPool() const {
        return disk_pool_;
    }

    void referenceBlocks(const MultiNodeResource& resource) const;
    void unreferenceBlocks(const MultiNodeResource& resource) const;
    void referenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const;
    void unreferenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const;
    bool canPublishDevice(const BlockIdList& blocks) const;

    BlockIdxType               allocateSingleBlock(Tier tier, BlockTreeRefType ref_type);
    std::optional<BlockIdList> allocateBlocks(size_t n, Tier tier, BlockTreeRefType ref_type);
    void                       releaseBlocks(Tier tier, const BlockIdList& blocks, BlockTreeRefType ref_type) const;
    void                       releaseSingleBlock(Tier tier, BlockIdxType block, BlockTreeRefType ref_type) const;

private:
    GroupSetPools                           pools_;
    GroupSetGeometry                        geometry_;
    // Typed handles for the RTP engine's buffer resolution.
    std::vector<DeviceBlockPoolPtr>         device_pools_;
    std::shared_ptr<DiskBlockPool>          disk_pool_;
    size_t                                  group_set_id_{0};
    std::shared_ptr<const CacheTopology>    topology_;
    std::vector<std::string>                group_tags_;
    size_t                                  payload_bytes_{0};
    size_t                                  storage_bytes_{0};
    size_t                                  storage_stride_bytes_{0};
    bool                                    enable_crc_{false};
    bool                                    uses_physical_payload_geometry_{false};
};

using GroupSetPtr = std::shared_ptr<GroupSet>;

}  // namespace rtp_llm
