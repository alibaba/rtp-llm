#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include <algorithm>
#include <optional>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

struct TransferWindow {
    const BlockIndicesType* block_ids;
    const CacheKeysType*    cache_keys;
    size_t                  begin;
    size_t                  count;
    size_t                  key_first;
    size_t                  key_stride;
};

std::optional<TransferWindow> getTransferWindow(const KVCacheResource& resource,
                                                int                    layer_id,
                                                const std::string&     cache_tag,
                                                int                    start_block_idx,
                                                int                    block_count,
                                                size_t                 key_first,
                                                size_t                 key_stride) {
    if (start_block_idx < 0 || block_count == 0 || block_count < -1 || key_stride == 0 || key_first >= key_stride) {
        RTP_LLM_LOG_WARNING(
            "invalid tagged cache conversion arguments for layer=%d tag=%s", layer_id, cache_tag.c_str());
        return std::nullopt;
    }

    const auto& cache_keys                = resource.cacheKeys();
    const auto& block_ids                 = resource.blocksForLayer(layer_id, cache_tag);
    const size_t max_local_blocks_for_keys =
        cache_keys.size() > key_first ? 1 + (cache_keys.size() - 1 - key_first) / key_stride : 0;
    const size_t actual_block_count = std::min(block_ids.size(), max_local_blocks_for_keys);
    const auto   begin              = static_cast<size_t>(start_block_idx);
    if (begin >= actual_block_count) {
        return std::nullopt;
    }

    const size_t remaining = actual_block_count - begin;
    const size_t count     = block_count > 0 ? std::min(static_cast<size_t>(block_count), remaining) : remaining;
    if (count == 0) {
        return std::nullopt;
    }
    return TransferWindow{&block_ids, &cache_keys, begin, count, key_first, key_stride};
}

}  // namespace

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(
    KVCacheResource& resource, int batch_id, int start_block_idx, int block_count, size_t key_first, size_t key_stride) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    for (int layer_id = 0; layer_id < resource.layerNum(); ++layer_id) {
        for (const auto& tag : resource.groupTagsForLayer(layer_id)) {
            auto buffer =
                convertLayer(resource, batch_id, layer_id, tag, start_block_idx, block_count, key_first, key_stride);
            if (buffer) {
                layer_cache_buffers.push_back(std::move(buffer));
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
                                                                     size_t           key_first,
                                                                     size_t           key_stride) {
    return convertLayer(resource,
                        batch_id,
                        layer_id,
                        resource.soleGroupTagForLayer(layer_id),
                        start_block_idx,
                        block_count,
                        key_first,
                        key_stride);
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(KVCacheResource&   resource,
                                                                     int                batch_id,
                                                                     int                layer_id,
                                                                     const std::string& cache_tag,
                                                                     int                start_block_idx,
                                                                     int                block_count,
                                                                     size_t             key_first,
                                                                     size_t             key_stride) {
    (void)batch_id;
    const auto window =
        getTransferWindow(resource, layer_id, cache_tag, start_block_idx, block_count, key_first, key_stride);
    if (!window.has_value()) {
        return nullptr;
    }

    auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id, cache_tag);
    for (size_t i = 0; i < window->count; ++i) {
        const size_t local_idx   = window->begin + i;
        const size_t logical_idx = window->key_first + local_idx * window->key_stride;
        if (logical_idx >= window->cache_keys->size()) {
            break;
        }
        if (isNullBlockIdx(window->block_ids->at(local_idx))) {
            continue;
        }
        layer_cache_buffer->addBlockId(window->cache_keys->at(logical_idx), window->block_ids->at(local_idx));
    }
    return layer_cache_buffer->blockIdMap().empty() ? nullptr : layer_cache_buffer;
}

bool LayerCacheBufferUtil::hasTransferableBlocks(const KVCacheResource& resource,
                                                 int                    layer_id,
                                                 const std::string&     cache_tag,
                                                 int                    start_block_idx,
                                                 int                    block_count,
                                                 size_t                 key_first,
                                                 size_t                 key_stride) {
    const auto window =
        getTransferWindow(resource, layer_id, cache_tag, start_block_idx, block_count, key_first, key_stride);
    if (!window.has_value()) {
        return false;
    }
    for (size_t i = 0; i < window->count; ++i) {
        const size_t local_idx   = window->begin + i;
        const size_t logical_idx = window->key_first + local_idx * window->key_stride;
        if (logical_idx >= window->cache_keys->size()) {
            break;
        }
        if (!isNullBlockIdx(window->block_ids->at(local_idx))) {
            return true;
        }
    }
    return false;
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
