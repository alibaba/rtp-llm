#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBufferUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<std::shared_ptr<LayerCacheBuffer>>
LayerCacheBufferUtil::convert(KVCacheResource& resource, int batch_id, int start_block_idx, int block_count) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    const auto& layer_block_ids = resource.layerBlockIds();
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
    const auto& layer_block_ids = resource.layerBlockIds();
    const auto& cache_keys      = resource.cacheKeys();

    // 检查层 ID 是否有效
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_block_ids.size()) {
        RTP_LLM_LOG_WARNING(
            "LayerCacheBufferUtil::convert: invalid layer_id %d, total layers: %zu", layer_id, layer_block_ids.size());
        return nullptr;
    }

    if (start_block_idx < 0 || block_count == 0) {
        RTP_LLM_LOG_WARNING(
            "LayerCacheBufferUtil::convert: invalid start_block_idx %d, block_count %d", start_block_idx, block_count);
        return nullptr;
    }

    const auto& block_ids          = layer_block_ids[layer_id]->blocks();
    auto        actual_block_count = std::min(block_ids.size(), cache_keys.size());
    auto        block_ids_size = block_count > 0 ? std::min(block_count, int(actual_block_count - start_block_idx)) :
                                                   (int(actual_block_count - start_block_idx));
    if (block_ids_size <= 0) {
        RTP_LLM_LOG_WARNING("LayerCacheBufferUtil::convert: block_ids_size %d", block_ids_size);
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

}  // namespace rtp_llm
