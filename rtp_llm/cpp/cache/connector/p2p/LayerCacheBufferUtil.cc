#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<std::shared_ptr<LayerCacheBuffer>>
LayerCacheBufferUtil::convert(KVCacheResource& resource, int batch_id, int start_block_idx, int block_count) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    const auto& layer_block_ids = resource.layerBlocks();
    for (size_t i = 0; i < layer_block_ids.size(); ++i) {
        auto layer_cache_buffer = convertLayer(resource, batch_id, i, start_block_idx, block_count);
        if (layer_cache_buffer) {
            layer_cache_buffers.push_back(layer_cache_buffer);
        }
    }
    return layer_cache_buffers;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(
    KVCacheResource& resource, int batch_id, int layer_id, int start_block_idx, int block_count) {
    const auto& layer_block_ids = resource.layerBlocks();
    const auto& cache_keys      = resource.cacheKeys();

    if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_block_ids.size()) {
        RTP_LLM_LOG_WARNING("invalid layer_id %d, total layers: %zu", layer_id, layer_block_ids.size());
        return nullptr;
    }

    // block_count == 0 is never valid; block_count < -1 is an undefined sentinel
    if (start_block_idx < 0 || block_count == 0 || block_count < -1) {
        RTP_LLM_LOG_WARNING("invalid start_block_idx %d, block_count %d", start_block_idx, block_count);
        return nullptr;
    }

    const auto& block_ids = layer_block_ids[layer_id]->blocks();
    // Use signed arithmetic throughout to avoid unsigned underflow when start_block_idx >= actual_block_count
    int actual_block_count = static_cast<int>(std::min(block_ids.size(), cache_keys.size()));
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
        int     block_id = block_ids[start_block_idx + i];
        int64_t key      = cache_keys[start_block_idx + i];
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
        auto block_infos = converter->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);

        transfer::KeyBlockInfo kbi;
        kbi.cache_key              = cache_key;
        kbi.blocks                 = std::move(block_infos);
        key_block_infos[cache_key] = std::make_shared<const transfer::KeyBlockInfo>(std::move(kbi));
    }
    return key_block_infos;
}

}  // namespace rtp_llm
