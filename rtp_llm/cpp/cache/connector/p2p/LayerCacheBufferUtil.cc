#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include <algorithm>
#include <exception>
#include <utility>

namespace rtp_llm {
namespace {
ErrorInfo conversionError(const std::string& message) {
    return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                     "LayerCacheBuffer conversion failed: " + message);
}

std::string bufferContext(const LayerCacheBuffer& buffer, int64_t key, int32_t block_id) {
    return "layer=" + std::to_string(buffer.getLayerId()) + " tag=" + buffer.cacheTag()
           + " cache_key=" + std::to_string(key) + " block_id=" + std::to_string(block_id);
}
}  // namespace

ErrorResult<std::vector<std::shared_ptr<LayerCacheBuffer>>> LayerCacheBufferUtil::convert(KVCacheResource&     resource,
                                                                                          const CacheTopology& topology,
                                                                                          int start_block_idx,
                                                                                          int block_count,
                                                                                          int cp_rank,
                                                                                          int cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;
    for (const auto& layer : topology.layers()) {
        auto buffers = convertLayer(resource, topology, layer.layer_id, start_block_idx, block_count, cp_rank, cp_size);
        if (!buffers.ok()) {
            return buffers.status();
        }
        for (auto& buffer : buffers.value()) {
            result.push_back(std::move(buffer));
        }
    }
    return std::move(result);
}

ErrorResult<std::vector<std::shared_ptr<LayerCacheBuffer>>>
LayerCacheBufferUtil::convertLayer(KVCacheResource&     resource,
                                   const CacheTopology& topology,
                                   int                  layer_id,
                                   int                  start_block_idx,
                                   int                  block_count,
                                   int                  cp_rank,
                                   int                  cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= topology.layers().size()) {
        return conversionError("layer=" + std::to_string(layer_id) + ": invalid layer");
    }
    for (const auto& group : topology.groupsForLayer(layer_id)) {
        auto buffer = convertLayerTag(resource, group.get(), layer_id, start_block_idx, block_count, cp_rank, cp_size);
        if (!buffer.ok()) {
            return buffer.status();
        }
        if (buffer.value()) {
            result.push_back(std::move(buffer.value()));
        }
    }
    return std::move(result);
}

ErrorResult<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convertLayerTag(KVCacheResource& resource,
                                                                                     const GroupBase& group,
                                                                                     int              layer_id,
                                                                                     int              start_block_idx,
                                                                                     int              block_count,
                                                                                     int              cp_rank,
                                                                                     int              cp_size) {
    const size_t      count   = resource.cacheKeys().size();
    const std::string context = "layer=" + std::to_string(layer_id) + " tag=" + group.tag;
    if (cp_size <= 0 || cp_rank < 0 || cp_rank >= cp_size || start_block_idx < 0 || block_count < -1
        || static_cast<size_t>(start_block_idx) > count) {
        return conversionError(context + ": invalid range or CP rank/size");
    }
    size_t begin = static_cast<size_t>(start_block_idx);
    if (block_count > 0 && static_cast<size_t>(block_count) > count - begin) {
        return conversionError(context + ": requested block range exceeds cache keys");
    }
    const size_t end = block_count >= 0 ? begin + static_cast<size_t>(block_count) : count;
    if (group.policy.active_tail_blocks > 0) {
        const size_t tail = static_cast<size_t>(group.policy.active_tail_blocks);
        begin             = std::max(begin, end > tail ? end - tail : 0);
    }
    std::vector<size_t> positions;
    for (size_t pos = begin; pos < end; ++pos) {
        // Select ownership first. A block outside this CP projection is not missing.
        if (CPSlotMapper::physicalBlockPosition(group.policy, pos, count, cp_rank, cp_size)) {
            positions.push_back(pos);
        }
    }
    return convertLayerTagForRoute(resource, group, layer_id, positions, cp_rank, cp_size);
}

ErrorResult<std::shared_ptr<LayerCacheBuffer>>
LayerCacheBufferUtil::convertLayerTagForRoute(KVCacheResource&           resource,
                                              const GroupBase&           group,
                                              int                        layer_id,
                                              const std::vector<size_t>& logical_positions,
                                              int                        cp_rank,
                                              int                        cp_size) {
    const std::string context = "layer=" + std::to_string(layer_id) + " tag=" + group.tag;
    if (cp_size <= 0 || cp_rank < 0 || cp_rank >= cp_size) {
        return conversionError(context + ": invalid CP rank/size");
    }
    if (logical_positions.empty()) {
        return std::shared_ptr<LayerCacheBuffer>{};
    }
    try {
        const auto& keys   = resource.cacheKeys();
        const auto& blocks = resource.blocksForLayer(layer_id, group.tag);
        auto        buffer = std::make_shared<LayerCacheBuffer>(layer_id, group.tag);
        // Route positions already include planner tail/window filtering. Never filter again here.
        for (size_t pos : logical_positions) {
            const std::string pos_context = context + " logical_pos=" + std::to_string(pos);
            if (pos >= keys.size()) {
                return conversionError(pos_context
                                       + " cache_key=<missing> block_id=<missing>: logical position out of range");
            }
            const auto physical = CPSlotMapper::physicalBlockPosition(group.policy, pos, keys.size(), cp_rank, cp_size);
            const std::string key_context = pos_context + " cache_key=" + std::to_string(keys[pos]);
            if (!physical || *physical >= blocks.size()) {
                return conversionError(key_context + " block_id=<missing> physical_pos="
                                       + (physical ? std::to_string(*physical) : "<missing>")
                                       + ": required physical block is missing");
            }
            if (isNullBlockIdx(blocks[*physical]) || blocks[*physical] < 0) {
                return conversionError(key_context + " block_id=" + std::to_string(blocks[*physical])
                                       + ": null/invalid block");
            }
            if (buffer->blockIdMap().count(keys[pos])) {
                return conversionError(bufferContext(*buffer, keys[pos], blocks[*physical]) + ": duplicate cache key");
            }
            buffer->addBlockId(keys[pos], blocks[*physical]);
        }
        if (buffer->blockIdMap().size() != logical_positions.size()) {
            return conversionError(context + ": incomplete route key set");
        }
        return std::move(buffer);
    } catch (const std::exception& error) {
        return conversionError(context + ": " + error.what());
    }
}

ErrorResult<std::vector<std::shared_ptr<LayerCacheBuffer>>>
LayerCacheBufferUtil::convertTagForRoute(KVCacheResource&           resource,
                                         const CacheTopology&       topology,
                                         const std::string&         cache_tag,
                                         const std::vector<size_t>& positions,
                                         int                        cp_rank,
                                         int                        cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;
    if (positions.empty()) {
        return std::move(result);
    }
    const auto group = std::find_if(
        topology.groups().begin(), topology.groups().end(), [&](const auto& entry) { return entry.tag == cache_tag; });
    if (group == topology.groups().end() || group->layer_ids.empty()) {
        return conversionError("tag=" + cache_tag + ": non-empty route has no topology layers");
    }
    for (int layer_id : group->layer_ids) {
        auto buffer = convertLayerTagForRoute(resource, *group, layer_id, positions, cp_rank, cp_size);
        if (!buffer.ok()) {
            return buffer.status();
        }
        result.push_back(std::move(buffer.value()));
    }
    return std::move(result);
}

ErrorResult<transfer::KeyBlockInfoMap>
LayerCacheBufferUtil::buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                         const std::shared_ptr<LayerCacheBuffer>&    buffer,
                                         int                                         partition_count,
                                         int                                         partition_id) {
    if (!converter || !buffer) {
        return conversionError("layer/tag/cache_key/block_id=<missing>: null converter or layer buffer");
    }
    const std::string context = "layer=" + std::to_string(buffer->getLayerId()) + " tag=" + buffer->cacheTag();
    if (partition_count <= 0 || partition_id < 0 || partition_id >= partition_count) {
        return conversionError(context + ": invalid head partition count=" + std::to_string(partition_count)
                               + " id=" + std::to_string(partition_id));
    }
    if (buffer->blockIdMap().empty()) {
        return conversionError(context + ": empty required block map");
    }
    transfer::KeyBlockInfoMap result;
    for (const auto& [key, block_id] : buffer->blockIdMap()) {
        const auto key_context = bufferContext(*buffer, key, block_id);
        if (block_id < 0 || isNullBlockIdx(block_id)) {
            return conversionError(key_context + ": null/invalid block");
        }
        std::vector<BlockInfo> parts;
        try {
            parts = converter->convertIndexToBuffer(
                buffer->getLayerId(), buffer->cacheTag(), block_id, partition_count, partition_id);
        } catch (const std::exception& error) {
            return conversionError(key_context + ": " + error.what());
        }
        if (parts.empty()) {
            return conversionError(key_context + ": converter returned no block info");
        }
        for (const auto& part : parts) {
            if (!part.addr || part.size_bytes == 0) {
                return conversionError(key_context + ": converter returned an invalid block info");
            }
        }
        transfer::KeyBlockInfo info;
        info.cache_key = key;
        info.blocks    = std::move(parts);
        result.emplace(key, std::make_shared<const transfer::KeyBlockInfo>(std::move(info)));
    }
    return std::move(result);
}

ErrorResult<transfer::KeyBlockInfoMap>
LayerCacheBufferUtil::buildKeyBlockInfosSliced(const std::shared_ptr<LayerBlockConverter>& converter,
                                               const std::shared_ptr<LayerCacheBuffer>&    buffer,
                                               int                                         partition_count,
                                               int                                         partition_id,
                                               const SliceSpec&                            slice,
                                               size_t                                      payload_bytes) {
    const std::string context =
        buffer ? "layer=" + std::to_string(buffer->getLayerId()) + " tag=" + buffer->cacheTag() : "layer/tag=<missing>";
    if (slice.count <= 0 || slice.index < 0 || slice.index >= slice.count
        || (slice.mode != CpBlockSliceMode::NONE && slice.mode != CpBlockSliceMode::EQUAL_BYTES
            && slice.mode != CpBlockSliceMode::PAYLOAD_BYTES)
        || (slice.mode == CpBlockSliceMode::NONE && slice.count != 1) || (slice.count > 1 && partition_count != 1)) {
        return conversionError(context
                               + ": invalid/incompatible slice mode=" + std::to_string(static_cast<int>(slice.mode))
                               + " count=" + std::to_string(slice.count) + " index=" + std::to_string(slice.index)
                               + " partition_count=" + std::to_string(partition_count));
    }
    auto blocks = buildKeyBlockInfos(converter, buffer, partition_count, partition_id);
    if (!blocks.ok() || slice.mode == CpBlockSliceMode::NONE || slice.count == 1) {
        return blocks;
    }
    const size_t              count = static_cast<size_t>(slice.count);
    const size_t              index = static_cast<size_t>(slice.index);
    transfer::KeyBlockInfoMap result;
    for (const auto& [key, info] : blocks.value()) {
        const auto key_context = bufferContext(*buffer, key, buffer->getBlockId(key));
        if (info->blocks.size() != 1) {
            return conversionError(key_context + ": expected one block part for CP slice");
        }
        auto         part  = info->blocks.front();
        const size_t bytes = slice.mode == CpBlockSliceMode::PAYLOAD_BYTES ? payload_bytes : part.size_bytes;
        if (bytes == 0 || bytes > part.size_bytes || bytes % count != 0) {
            return conversionError(key_context + ": invalid slice bytes=" + std::to_string(bytes) + " block_bytes="
                                   + std::to_string(part.size_bytes) + " count=" + std::to_string(count));
        }
        const size_t size = bytes / count;
        // index < count and bytes <= block size guarantee both operations cannot overflow.
        const size_t offset = size * index;
        if (offset > part.size_bytes || size > part.size_bytes - offset) {
            return conversionError(key_context + ": slice out of bounds");
        }
        part.addr       = static_cast<char*>(part.addr) + offset;
        part.size_bytes = size;
        transfer::KeyBlockInfo sliced;
        sliced.cache_key = key;
        sliced.blocks.push_back(part);
        result.emplace(key, std::make_shared<const transfer::KeyBlockInfo>(std::move(sliced)));
    }
    return std::move(result);
}
}  // namespace rtp_llm
