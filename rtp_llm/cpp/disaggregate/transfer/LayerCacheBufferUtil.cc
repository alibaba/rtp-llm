#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBufferUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(KVCacheResourceV1& resource,
                                                                             int                batch_id) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    const auto& layer_block_ids = resource.layerBlockIds();
    const auto& cache_keys      = resource.cacheKeys();
    RTP_LLM_LOG_INFO("LayerCacheBufferUtil::convert: layer_block_ids size: %zu, cache_keys size: %zu",
                     layer_block_ids.size(),
                     cache_keys.size());

    for (size_t i = 0; i < layer_block_ids.size(); ++i) {
        auto layer_cache_buffer = convert(resource, batch_id, i);
        if (layer_cache_buffer) {
            layer_cache_buffers.push_back(layer_cache_buffer);
        }
    }
    return layer_cache_buffers;
}

std::shared_ptr<LayerCacheBuffer>
LayerCacheBufferUtil::convert(KVCacheResourceV1& resource, int batch_id, int layer_id) {
    const auto& layer_block_ids = resource.layerBlockIds();
    const auto& cache_keys      = resource.cacheKeys();

    // 检查层 ID 是否有效
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= layer_block_ids.size()) {
        RTP_LLM_LOG_WARNING(
            "LayerCacheBufferUtil::convert: invalid layer_id %d, total layers: %zu", layer_id, layer_block_ids.size());
        return nullptr;
    }

    // 创建 LayerCacheBuffer
    auto        layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
    const auto& block_ids          = layer_block_ids[layer_id]->block_indices;
    auto        block_ids_size     = std::min(block_ids.size(), cache_keys.size());

    // 将 block_ids 和 cache_keys 配对添加到 LayerCacheBuffer
    // 注意：cache_keys 可能与 block_ids 的数量不一致，需要确保索引不越界
    for (size_t i = 0; i < block_ids_size; ++i) {
        int     block_id = block_ids[i];
        int64_t key      = cache_keys[i];
        layer_cache_buffer->addBlockId(key, block_id);
    }

    return layer_cache_buffer;
}

}  // namespace rtp_llm
