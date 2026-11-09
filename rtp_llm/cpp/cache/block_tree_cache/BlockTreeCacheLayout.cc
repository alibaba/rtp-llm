#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {
namespace {

size_t checkedProduct(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("BlockTree transfer workspace size overflow");
    return a * b;
}

bool compatible(const CacheConfig& config, const std::string& lhs_tag, const std::string& rhs_tag) {
    const auto& lhs = config.topology().group(lhs_tag);
    const auto& rhs = config.topology().group(rhs_tag);
    return lhs.policy.group_type == rhs.policy.group_type && lhs.seqSizePerBlock() == rhs.seqSizePerBlock()
           && lhs.policy.cp_mapping == rhs.policy.cp_mapping
           && (lhs.policy.group_type != CacheGroupType::SWA
               || lhs.policy.sliding_window_size == rhs.policy.sliding_window_size)
           && ((lhs.policy.group_type != CacheGroupType::SWA && lhs.policy.group_type != CacheGroupType::LINEAR)
               || lhs.policy.active_tail_blocks == rhs.policy.active_tail_blocks);
}

size_t positive(int64_t value, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(value > 0 && static_cast<uint64_t>(value) <= std::numeric_limits<size_t>::max(),
                            "%s must be positive and fit in size_t",
                            name);
    return static_cast<size_t>(value);
}

}  // namespace

bool buildBlockTreeGroupMembers(const CacheConfig& config, std::vector<std::vector<std::string>>& members) {
    members.clear();
    for (const auto& group : config.topology().groups()) {
        if (!group.policy.enable_prefix_reuse)
            continue;
        const auto type = group.policy.group_type;
        if (type != CacheGroupType::FULL && type != CacheGroupType::SWA && type != CacheGroupType::LINEAR) {
            RTP_LLM_LOG_ERROR("unsupported reusable cache group type for tag=%s", group.tag.c_str());
            return false;
        }
        auto it = std::find_if(members.begin(), members.end(), [&](const auto& tags) {
            return config.topology().group(tags.front()).policy.group_type == type;
        });
        if (it == members.end()) {
            members.push_back({group.tag});
        } else {
            if (!compatible(config, it->front(), group.tag)) {
                RTP_LLM_LOG_ERROR("incompatible BlockTree reuse coordinates for tags=%s and %s",
                                  it->front().c_str(),
                                  group.tag.c_str());
                return false;
            }
            it->push_back(group.tag);
        }
    }
    return true;
}

BackingLayout computeBlockTreeBackingLayout(const CacheConfig&              config,
                                            const std::vector<std::string>& members,
                                            bool                            enable_crc,
                                            size_t                          alignment) {
    BackingLayout layout;
    for (const auto& tag : members) {
        const auto bytes = config.blockSizeBytesForGroup(tag);
        RTP_LLM_CHECK_WITH_INFO(bytes > 0 && bytes <= std::numeric_limits<size_t>::max() - layout.payload_bytes,
                                "invalid group set payload for tag=%s",
                                tag.c_str());
        layout.payload_bytes += bytes;
        enable_crc = enable_crc && config.topology().group(tag).policy.memory_placement == CacheMemoryPlacement::DEVICE;
    }
    layout.crc_enabled   = enable_crc;
    layout.storage_bytes = enable_crc ? CrcCopyWorkspace::encodedBytes(layout.payload_bytes) : layout.payload_bytes;
    RTP_LLM_CHECK_WITH_INFO(alignment > 0
                                && layout.storage_bytes <= std::numeric_limits<size_t>::max() - (alignment - 1),
                            "BlockTreeCache pool stride overflow");
    layout.pool_stride = (layout.storage_bytes + alignment - 1) / alignment * alignment;
    return layout;
}

size_t asyncTransferBatchLimit(size_t configured_batch, size_t staging_blocks, bool has_device_disk_route) {
    if (!configured_batch)
        throw std::invalid_argument("invalid async transfer batch size");
    if (!has_device_disk_route)
        return configured_batch;
    if (staging_blocks < 2 || staging_blocks % 2)
        throw std::invalid_argument("invalid async staging budget");
    // Each direction gets N/2 slots, shared by all cache group types.
    return std::min(configured_batch, staging_blocks / 2);
}

size_t crcTransferWorkspaceCount(size_t workers, size_t slots_per_loop, bool async) {
    if (!workers || (async && !slots_per_loop))
        throw std::invalid_argument("invalid CRC workspace count");
    return checkedProduct(checkedProduct(2, workers), async ? slots_per_loop : 1);
}

size_t blockTreeTransferDeviceMemory(const CacheConfig& config, const KVCacheConfig& options) {
    if (!options.enable_crc || (!options.enable_memory_cache && !options.enable_disk_cache)
        || !CrcCopyWorkspace::available())
        return 0;
    std::vector<std::vector<std::string>> members;
    RTP_LLM_CHECK_WITH_INFO(buildBlockTreeGroupMembers(config, members), "invalid BlockTree group aggregation");
    bool   async       = options.btc_transfer_engine_async;
    size_t max_payload = 0, max_tiles = 0;
    for (const auto& tags : members) {
        const auto layout = computeBlockTreeBackingLayout(config, tags, true, 16);
        if (!layout.crc_enabled) {
            async = false;  // A plain device route makes the whole engine use legacy.
            continue;
        }
        size_t tiles = 0;
        for (const auto& tag : tags)
            for (const auto layer : config.layerIdsForGroup(tag)) {
                const auto& physical = config.physicalGroupForLayer(layer, tag);
                tiles += (physical.kvBlockStrideBytes() > 0) + (physical.kvScaleStrideBytes() > 0);
            }
        max_payload = std::max(max_payload, layout.payload_bytes);
        max_tiles   = std::max(max_tiles, tiles);
    }
    if (!max_payload)
        return 0;
    size_t batch = positive(options.memory_cache_max_descriptors_per_transfer_batch, "transfer batch size");
    if (async)
        batch = asyncTransferBatchLimit(
            batch,
            options.enable_disk_cache ? positive(options.disk_cache_staging_block_count, "disk staging blocks") : 0,
            options.enable_disk_cache);
    const auto count = crcTransferWorkspaceCount(positive(options.block_tree_transfer_worker_count, "transfer workers"),
                                                 async ? positive(options.btc_crc_slot_count, "CRC slots") : 1,
                                                 async);
    const auto usage = CrcCopyWorkspace::memoryUsage(batch, max_payload, checkedProduct(batch, max_tiles));
    return checkedProduct(count, usage.device_bytes);
}

}  // namespace rtp_llm
