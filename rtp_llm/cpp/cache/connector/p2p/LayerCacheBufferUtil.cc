#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <algorithm>

namespace rtp_llm {

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(KVCacheResource&     resource,
                                                                             const CacheTopology& topology,
                                                                             int                  start_block_idx,
                                                                             int                  block_count,
                                                                             int                  cp_rank,
                                                                             int                  cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;
    for (const auto& layer : topology.layers()) {
        auto layer_buffers =
            convertLayer(resource, topology, layer.layer_id, start_block_idx, block_count, cp_rank, cp_size);
        result.insert(result.end(), layer_buffers.begin(), layer_buffers.end());
    }
    return result;
}

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convertLayer(
    KVCacheResource& resource,
    const CacheTopology& topology,
    int layer_id,
    int start_block_idx,
    int block_count,
    int cp_rank,
    int cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;
    for (const auto& group_ref : topology.groupsForLayer(layer_id)) {
        auto buffer =
            convertLayerTag(resource, group_ref.get(), layer_id, start_block_idx, block_count, cp_rank, cp_size);
        if (buffer) {
            result.push_back(std::move(buffer));
        }
    }
    return result;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayerTag(KVCacheResource& resource,
                                                                        const GroupBase&  group,
                                                                        int               layer_id,
                                                                        int               start_block_idx,
                                                                        int               block_count,
                                                                        int               cp_rank,
                                                                        int               cp_size) {
    const auto& cache_keys = resource.cacheKeys();
    if (start_block_idx < 0 || block_count == 0 || block_count < -1
        || static_cast<size_t>(start_block_idx) >= cache_keys.size()) {
        return nullptr;
    }

    const size_t logical_begin = static_cast<size_t>(start_block_idx);
    const size_t logical_end = block_count > 0 ?
                                   std::min(cache_keys.size(), logical_begin + static_cast<size_t>(block_count)) :
                                   cache_keys.size();
    const auto& block_ids = resource.blocksForLayer(layer_id, group.tag);
    auto buffer = std::make_shared<LayerCacheBuffer>(layer_id, group.tag);
    for (size_t logical_pos = logical_begin; logical_pos < logical_end; ++logical_pos) {
        const auto physical_pos =
            CPSlotMapper::physicalBlockPosition(group.policy, logical_pos, cache_keys.size(), cp_rank, cp_size);
        if (!physical_pos || *physical_pos >= block_ids.size() || isNullBlockIdx(block_ids[*physical_pos])) {
            continue;
        }
        buffer->addBlockId(cache_keys[logical_pos], block_ids[*physical_pos]);
    }
    return buffer->blockIdMap().empty() ? nullptr : buffer;
}

transfer::KeyBlockInfoMap
LayerCacheBufferUtil::buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                         const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                         int                                         partition_count,
                                         int                                         partition_id) {
    transfer::KeyBlockInfoMap key_block_infos;
    int                       layer_id = layer_cache_buffer->getLayerId();

    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto block_infos = converter->convertIndexToBuffer(
            layer_id, layer_cache_buffer->cacheTag(), block_id, partition_count, partition_id);

        transfer::KeyBlockInfo kbi;
        kbi.cache_key              = cache_key;
        kbi.blocks                 = std::move(block_infos);
        key_block_infos[cache_key] = std::make_shared<const transfer::KeyBlockInfo>(std::move(kbi));
    }
    return key_block_infos;
}

}  // namespace rtp_llm
