#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"

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

class GroupSet {
public:
    GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
             std::shared_ptr<HostBlockPool>  host_pool,
             DiskBlockPoolPtr                disk_pool);

    virtual ~GroupSet() = default;

    void initialize(size_t                             group_set_id,
                    std::shared_ptr<const CacheConfig> cache_config,
                    std::vector<std::string>           group_tags,
                    BackingLayout                      backing_layout = {});

    size_t groupSetId() const {
        return group_set_id_;
    }
    const std::vector<std::string>& groupTags() const {
        return group_tags_;
    }
    std::shared_ptr<const CacheConfig> cacheConfig() const {
        return cache_config_;
    }
    const GroupBase& group(std::string_view tag) const {
        return cache_config_->group(tag);
    }
    const KVCacheSpec& groupSpec(std::string_view tag) const;
    // Local layer IDs for the tag's owning module (module-local, never union).
    std::vector<int> localLayerIdsForMember(std::string_view tag) const;
    size_t           payloadBytes() const {
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
    const std::shared_ptr<HostBlockPool>& hostPool() const {
        return host_pool_;
    }
    const std::shared_ptr<DiskBlockPool>& diskPool() const {
        return disk_pool_;
    }

    void referenceBlocks(const MultiNodeResource& resource) const;
    void unreferenceBlocks(const MultiNodeResource& resource) const;
    void referenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const;
    void unreferenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const;

    BlockIdxType               allocateSingleBlock(Tier tier, BlockTreeRefType ref_type);
    std::optional<BlockIdList> allocateBlocks(size_t n, Tier tier, BlockTreeRefType ref_type);
    void                       unreferenceBlocks(Tier tier, const BlockIdList& blocks, BlockTreeRefType ref_type) const;

private:
    std::vector<DeviceBlockPoolPtr>         device_pools_;
    std::shared_ptr<HostBlockPool>          host_pool_;
    std::shared_ptr<DiskBlockPool>          disk_pool_;
    size_t                                  group_set_id_{0};
    std::shared_ptr<const CacheConfig>      cache_config_;
    std::vector<std::string>                group_tags_;
    size_t                                  payload_bytes_{0};
    size_t                                  storage_bytes_{0};
    bool                                    enable_crc_{false};
    bool                                    uses_physical_payload_geometry_{false};
};

using GroupSetPtr = std::shared_ptr<GroupSet>;

void appendBlocks(std::vector<MultiNodeResource>& resources, size_t group_set_id, Tier tier, const BlockIdList& blocks);

void referenceBlocks(const std::vector<GroupSetPtr>& group_sets, const std::vector<MultiNodeResource>& resources);
void unreferenceBlocks(const std::vector<GroupSetPtr>& group_sets, const std::vector<MultiNodeResource>& resources);
void referenceBlocks(const std::vector<GroupSetPtr>&       group_sets,
                     const std::vector<MultiNodeResource>& resources,
                     BlockTreeRefType                      ref_type);
void unreferenceBlocks(const std::vector<GroupSetPtr>&       group_sets,
                       const std::vector<MultiNodeResource>& resources,
                       BlockTreeRefType                      ref_type);

}  // namespace rtp_llm
