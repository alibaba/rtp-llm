#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(
    KVCacheResource& resource, int batch_id, int start_block_idx, int block_count, int cp_rank, int cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    const auto& layer_group_block_ids = resource.layerGroupBlocks();
    for (size_t layer_id = 0; layer_id < layer_group_block_ids.size(); ++layer_id) {
        const auto& group_block_ids = layer_group_block_ids[layer_id];
        for (size_t group_id = 0; group_id < group_block_ids.size(); ++group_id) {
            if (!group_block_ids[group_id]) {
                continue;
            }
            auto layer_cache_buffer = convertLayer(resource,
                                                   batch_id,
                                                   static_cast<int>(layer_id),
                                                   static_cast<int>(group_id),
                                                   start_block_idx,
                                                   block_count,
                                                   cp_rank,
                                                   cp_size);
            if (layer_cache_buffer) {
                layer_cache_buffers.push_back(layer_cache_buffer);
            }
        }
    }
    return layer_cache_buffers;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(KVCacheResource& resource,
                                                                     int              batch_id,
                                                                     int              layer_id,
                                                                     int              start_block_idx,
                                                                     int              block_count,
                                                                     int              cp_rank,
                                                                     int              cp_size) {
    const auto& layer_group_block_ids = resource.layerGroupBlocks();

    if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_group_block_ids.size()) {
        RTP_LLM_LOG_WARNING("invalid layer_id %d, total layers: %zu", layer_id, layer_group_block_ids.size());
        return nullptr;
    }

    // block_count == 0 is never valid; block_count < -1 is an undefined sentinel
    if (start_block_idx < 0 || block_count == 0 || block_count < -1) {
        RTP_LLM_LOG_WARNING("invalid start_block_idx %d, block_count %d", start_block_idx, block_count);
        return nullptr;
    }

    if (cp_size < 1 || cp_rank < 0 || cp_rank >= cp_size) {
        RTP_LLM_LOG_WARNING("invalid cp_rank/cp_size: cp_rank=%d cp_size=%d", cp_rank, cp_size);
        return nullptr;
    }

    const auto& group_block_ids   = layer_group_block_ids[static_cast<size_t>(layer_id)];
    int         selected_group_id = -1;
    for (size_t group_id = 0; group_id < group_block_ids.size(); ++group_id) {
        if (group_block_ids[group_id]) {
            if (selected_group_id >= 0) {
                RTP_LLM_LOG_WARNING("convertLayer without group_id is ambiguous for layer_id %d", layer_id);
                return nullptr;
            }
            selected_group_id = static_cast<int>(group_id);
        }
    }
    if (selected_group_id < 0) {
        RTP_LLM_LOG_WARNING("no block ids found for layer_id %d", layer_id);
        return nullptr;
    }
    return convertLayer(
        resource, batch_id, layer_id, selected_group_id, start_block_idx, block_count, cp_rank, cp_size);
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(KVCacheResource& resource,
                                                                     int              batch_id,
                                                                     int              layer_id,
                                                                     int              group_id,
                                                                     int              start_block_idx,
                                                                     int              block_count,
                                                                     int              cp_rank,
                                                                     int              cp_size) {
    (void)batch_id;
    const auto& layer_group_block_ids = resource.layerGroupBlocks();
    const auto& cache_keys            = resource.cacheKeys();

    if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_group_block_ids.size()) {
        RTP_LLM_LOG_WARNING("invalid layer_id %d, total layers: %zu", layer_id, layer_group_block_ids.size());
        return nullptr;
    }
    if (group_id < 0 || static_cast<size_t>(group_id) >= layer_group_block_ids[static_cast<size_t>(layer_id)].size()
        || !layer_group_block_ids[static_cast<size_t>(layer_id)][static_cast<size_t>(group_id)]) {
        RTP_LLM_LOG_WARNING("invalid group_id %d for layer_id %d", group_id, layer_id);
        return nullptr;
    }

    // block_count == 0 is never valid; block_count < -1 is an undefined sentinel
    if (start_block_idx < 0 || block_count == 0 || block_count < -1) {
        RTP_LLM_LOG_WARNING("invalid start_block_idx %d, block_count %d", start_block_idx, block_count);
        return nullptr;
    }

    if (cp_size < 1 || cp_rank < 0 || cp_rank >= cp_size) {
        RTP_LLM_LOG_WARNING("invalid cp_rank/cp_size: cp_rank=%d cp_size=%d", cp_rank, cp_size);
        return nullptr;
    }

    const auto& block_ids =
        layer_group_block_ids[static_cast<size_t>(layer_id)][static_cast<size_t>(group_id)]->blocks();
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

    auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id, group_id);
    for (int i = 0; i < block_ids_size; ++i) {
        const int local_idx   = start_block_idx + i;
        const int logical_idx = cp_rank + local_idx * cp_size;
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

transfer::KeyBlockInfoMap
LayerCacheBufferUtil::buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                         const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                         int                                         partition_count,
                                         int                                         partition_id) {
    transfer::KeyBlockInfoMap key_block_infos;
    int                       layer_id = layer_cache_buffer->getLayerId();

    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto block_infos = converter->convertIndexToBuffer(
            layer_id, layer_cache_buffer->getGroupId(), block_id, partition_count, partition_id);

        transfer::KeyBlockInfo kbi;
        kbi.cache_key              = cache_key;
        kbi.blocks                 = std::move(block_infos);
        key_block_infos[cache_key] = std::make_shared<const transfer::KeyBlockInfo>(std::move(kbi));
    }
    return key_block_infos;
}

}  // namespace rtp_llm
