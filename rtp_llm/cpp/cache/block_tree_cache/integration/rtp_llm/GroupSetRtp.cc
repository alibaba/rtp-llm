#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"

#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/PrefixCoverage.h"

namespace rtp_llm {

GroupSet::GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
                   std::shared_ptr<HostBlockPool> host_pool, DiskBlockPoolPtr disk_pool):
    GroupSet(GroupSetPools{{device_pools.begin(), device_pools.end()}, host_pool, disk_pool}) {
    device_pools_ = std::move(device_pools);
    disk_pool_ = std::move(disk_pool);
}

void GroupSet::initialize(size_t group_set_id, std::shared_ptr<const CacheTopology> topology,
                          std::vector<std::string> group_tags, BackingLayout backing_layout) {
    if (!topology || group_tags.empty()) {
        throw std::invalid_argument("RTP GroupSet requires a topology and member tags");
    }
    const bool physical = backing_layout.payload_bytes > 0;
    for (const auto& tag : group_tags) {
        (void)topology->group(tag);
        if (!physical) {
            const size_t bytes = topology->blockSizeBytesForGroup(tag);
            if (bytes > std::numeric_limits<size_t>::max() - backing_layout.payload_bytes) {
                throw std::overflow_error("GroupSet payload exceeds the size limit: " + tag);
            }
            backing_layout.payload_bytes += bytes;
        }
    }
    if (!backing_layout.pool_stride) {
        backing_layout.pool_stride = disk_pool_ ? disk_pool_->strideBytes()
                                                  : pools_.host ? pools_.host->strideBytes() : 0;
    }

    const auto& model_group = topology->group(group_tags.front());
    const CacheGroupType type = model_group.policy.group_type;
    const auto window = static_cast<size_t>(model_group.policy.sliding_window_size);
    // The coordinator supplies one canonical row per group page, including CP projections.
    const size_t window_rows = type == CacheGroupType::SWA && window > 0
        ? reusableBlockCount(type, model_group.cacheKeyTokenStride(), window, std::numeric_limits<size_t>::max())
        : 0;
    initialize(group_set_id, GroupSetGeometry{type, 1, window_rows}, std::move(group_tags), backing_layout);
    uses_physical_payload_geometry_ = physical;
    topology_ = std::move(topology);
}

const GroupBase& GroupSet::group(std::string_view tag) const {
    assert(topology_ && "RTP group spec requires a configured topology");
    return topology_->group(tag);
}

SWAGroupSet::SWAGroupSet(size_t sliding_window_size, size_t seq_size_per_block,
                         std::vector<DeviceBlockPoolPtr> device_pools, std::shared_ptr<HostBlockPool> host_pool,
                         DiskBlockPoolPtr disk_pool):
    GroupSet(std::move(device_pools), std::move(host_pool), std::move(disk_pool)),
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {}

}  // namespace rtp_llm
