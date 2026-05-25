#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(
    KVCacheResource& resource, int batch_id, int start_block_idx, int block_count, int cp_rank, int cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    const auto& layer_block_ids = resource.layerBlocks();
    for (size_t i = 0; i < layer_block_ids.size(); ++i) {
        auto layer_cache_buffer = convertLayer(resource, batch_id, i, start_block_idx, block_count, cp_rank, cp_size);
        if (layer_cache_buffer) {
            layer_cache_buffers.push_back(layer_cache_buffer);
        }
    }
    return layer_cache_buffers;
}

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convertWithRegions(
    KVCacheResource&                     resource,
    int                                  batch_id,
    const std::vector<std::vector<int>>& layer_region_to_group_id,
    const std::vector<CacheGroupType>&   group_types,
    int                                  start_block_idx,
    int                                  block_count) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;

    for (size_t layer_id = 0; layer_id < layer_region_to_group_id.size(); ++layer_id) {
        const auto& regions = layer_region_to_group_id[layer_id];
        for (size_t r = 0; r < regions.size(); ++r) {
            int group_id = regions[r];
            if (group_id < 0) {
                continue;
            }
            auto region_name = static_cast<KVCacheRegionName>(r);
            auto buf = convertLayerRegion(resource, batch_id, layer_id, region_name, start_block_idx, block_count);
            if (buf) {
                result.push_back(buf);
            }
        }
    }
    return result;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(KVCacheResource& resource,
                                                                     int              batch_id,
                                                                     int              layer_id,
                                                                     int              start_block_idx,
                                                                     int              block_count,
                                                                     int              cp_rank,
                                                                     int              cp_size) {
    const auto& layer_block_ids = resource.layerBlocks();
    const auto& cache_keys      = resource.cacheKeys();

    if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_block_ids.size()) {
        RTP_LLM_LOG_WARNING("invalid layer_id %d, total layers: %zu", layer_id, layer_block_ids.size());
        return nullptr;
    }

    if (start_block_idx < 0 || block_count == 0 || block_count < -1) {
        RTP_LLM_LOG_WARNING("invalid start_block_idx %d, block_count %d", start_block_idx, block_count);
        return nullptr;
    }

    if (cp_size < 1 || cp_rank < 0 || cp_rank >= cp_size) {
        RTP_LLM_LOG_WARNING("invalid cp_rank/cp_size: cp_rank=%d cp_size=%d", cp_rank, cp_size);
        return nullptr;
    }

    const auto& block_ids = layer_block_ids[layer_id]->blocks();
    const int local_to_logical_stride = cp_size;
    const int local_to_logical_offset = cp_rank;
    const int max_local_blocks_for_keys =
        cp_size > 1 ?
            static_cast<int>((cache_keys.size() > static_cast<size_t>(cp_rank) ? cache_keys.size() - cp_rank : 0)
                             + cp_size - 1)
                / cp_size :
            static_cast<int>(cache_keys.size());
    int actual_block_count =
        static_cast<int>(std::min(block_ids.size(), static_cast<size_t>(max_local_blocks_for_keys)));
    if (start_block_idx >= actual_block_count) {
        RTP_LLM_LOG_WARNING("start_block_idx %d >= actual_block_count %d", start_block_idx, actual_block_count);
        return nullptr;
    }
    int block_ids_size = (block_count > 0) ? std::min(block_count, actual_block_count - start_block_idx) :
                                             (actual_block_count - start_block_idx);
    if (block_ids_size <= 0) {
        RTP_LLM_LOG_WARNING("block_ids_size %d", block_ids_size);
        return nullptr;
    }

    auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
    for (size_t i = 0; i < block_ids_size; ++i) {
        const int local_idx   = start_block_idx + static_cast<int>(i);
        const int logical_idx = local_to_logical_offset + local_idx * local_to_logical_stride;
        if (logical_idx < 0 || static_cast<size_t>(logical_idx) >= cache_keys.size()) {
            RTP_LLM_LOG_WARNING("logical_idx %d out of cache_keys range %zu (cp_rank=%d cp_size=%d)",
                                logical_idx,
                                cache_keys.size(),
                                cp_rank,
                                cp_size);
            break;
        }
        int     block_id = block_ids[local_idx];
        int64_t key      = cache_keys[logical_idx];
        layer_cache_buffer->addBlockId(key, block_id);
    }

    return layer_cache_buffer;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayerRegion(KVCacheResource&  resource,
                                                                           int               batch_id,
                                                                           int               layer_id,
                                                                           KVCacheRegionName region_name,
                                                                           int               start_block_idx,
                                                                           int               block_count) {
    const auto& block_indices = resource.blocks(layer_id, region_name);
    const auto& cache_keys    = resource.cacheKeys();

    if (start_block_idx < 0 || block_count == 0 || block_count < -1) {
        RTP_LLM_LOG_WARNING("convertLayerRegion: invalid start_block_idx %d, block_count %d", start_block_idx, block_count);
        return nullptr;
    }

    int actual_block_count = static_cast<int>(std::min(block_indices.size(), cache_keys.size()));
    if (start_block_idx >= actual_block_count) {
        return nullptr;
    }
    int size = (block_count > 0) ? std::min(block_count, actual_block_count - start_block_idx) :
                                   (actual_block_count - start_block_idx);
    if (size <= 0) {
        return nullptr;
    }

    auto buf = std::make_shared<LayerCacheBuffer>(layer_id, region_name);
    for (int i = 0; i < size; ++i) {
        buf->addBlockId(cache_keys[start_block_idx + i], block_indices[start_block_idx + i]);
    }
    return buf;
}

transfer::KeyBlockInfoMap
LayerCacheBufferUtil::buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                         const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                         int                                         partition_count,
                                         int                                         partition_id) {
    transfer::KeyBlockInfoMap key_block_infos;
    int                       layer_id    = layer_cache_buffer->getLayerId();
    KVCacheRegionName         region_name = layer_cache_buffer->getRegionName();

    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        std::vector<BlockInfo> block_infos;
        if (region_name != KVCacheRegionName::DEFAULT) {
            block_infos = converter->convertIndexToBuffer(layer_id, region_name, block_id, partition_count, partition_id);
        } else {
            block_infos = converter->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
        }

        transfer::KeyBlockInfo kbi;
        kbi.cache_key              = cache_key;
        kbi.blocks                 = std::move(block_infos);
        key_block_infos[cache_key] = std::make_shared<const transfer::KeyBlockInfo>(std::move(kbi));
    }
    return key_block_infos;
}

}  // namespace rtp_llm
