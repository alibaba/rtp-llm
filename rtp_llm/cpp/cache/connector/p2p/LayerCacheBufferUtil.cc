#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

bool validRangeArguments(
    int layer_id, std::string_view tag, int start_key_ordinal, int key_count, int cp_rank, int cp_size) {
    if (start_key_ordinal < 0 || key_count == 0 || key_count < -1 || cp_size < 1 || cp_rank < 0 || cp_rank >= cp_size) {
        RTP_LLM_LOG_WARNING("invalid tagged cache conversion arguments for layer=%d tag=%.*s",
                            layer_id,
                            static_cast<int>(tag.size()),
                            tag.data());
        return false;
    }
    return true;
}

void validateGroupPacking(const CacheConfig&     config,
                          const KVCacheResource& resource,
                          int                    layer_id,
                          std::string_view       tag) {
    RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "P2P transfer requires a non-empty cache group tag");
    const auto& group = config.groupForLayer(layer_id, tag);
    RTP_LLM_CHECK_WITH_INFO(config.seq_size_per_block > 0 && group.seqSizePerBlock() > 0
                                && group.seqSizePerBlock() % config.seq_size_per_block == 0,
                            "P2P transfer tag=%.*s has invalid global/physical spans=%zu/%zu",
                            static_cast<int>(tag.size()),
                            tag.data(),
                            config.seq_size_per_block,
                            group.seqSizePerBlock());
    RTP_LLM_CHECK_WITH_INFO(group.kernelSeqSizePerBlock() > 0
                                && group.seqSizePerBlock() % group.kernelSeqSizePerBlock() == 0,
                            "P2P transfer tag=%.*s has invalid physical/kernel spans=%zu/%zu",
                            static_cast<int>(tag.size()),
                            tag.data(),
                            group.seqSizePerBlock(),
                            group.kernelSeqSizePerBlock());

    // Validate layer/tag ownership without duplicating group geometry in the
    // request-owned physical binding sequence.
    resource.blockIdsForLayer(layer_id, tag);
}

template<typename Visitor>
bool visitSelectedBlocks(const CacheConfig&     config,
                         const KVCacheResource& resource,
                         int                    layer_id,
                         std::string_view       tag,
                         int                    start_key_ordinal,
                         int                    key_count,
                         int                    cp_rank,
                         int                    cp_size,
                         Visitor&&              visitor) {
    if (!validRangeArguments(layer_id, tag, start_key_ordinal, key_count, cp_rank, cp_size)) {
        return false;
    }
    validateGroupPacking(config, resource, layer_id, tag);

    const auto& group      = config.groupForLayer(layer_id, tag);
    const auto& block_ids  = resource.blockIdsForLayer(layer_id, tag);
    const auto& cache_keys = resource.cacheKeys();
    const auto  mapping    = cp_size > 1 ? group.policy.cp_mapping : CpBlockMappingMode::NONE;
    const auto  world_size = static_cast<size_t>(cp_size);
    const auto  rank       = static_cast<size_t>(cp_rank);

    const size_t local_block_count = block_ids.blocks().size();
    size_t       physical_capacity = local_block_count;
    if (mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN && local_block_count > 0) {
        physical_capacity = rank + (local_block_count - 1) * world_size + 1;
    } else if (mapping == CpBlockMappingMode::COMPACT_LAST_RANK) {
        physical_capacity = local_block_count * world_size;
    }

    const size_t keys_per_physical_block = group.seqSizePerBlock() / config.seq_size_per_block;
    const size_t available_key_count     = std::min(cache_keys.size(), physical_capacity * keys_per_physical_block);
    const size_t key_begin               = static_cast<size_t>(start_key_ordinal);
    if (key_begin >= available_key_count) {
        return false;
    }
    const size_t remaining = available_key_count - key_begin;
    const size_t count     = key_count > 0 ? std::min(static_cast<size_t>(key_count), remaining) : remaining;
    if (count == 0) {
        return false;
    }

    const size_t key_end        = key_begin + count;
    const size_t physical_begin = key_begin / keys_per_physical_block;
    const size_t physical_end   = (key_end + keys_per_physical_block - 1) / keys_per_physical_block;
    const size_t last_key       = key_end - 1;
    bool         found          = false;

    const auto visit_physical = [&](size_t physical_position) {
        if (mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
            RTP_LLM_CHECK_WITH_INFO(physical_position % world_size == rank,
                                    "P2P transfer physical block=%zu is not owned by rank=%zu tag=%.*s",
                                    physical_position,
                                    rank,
                                    static_cast<int>(tag.size()),
                                    tag.data());
        }
        const size_t local_position =
            mapping == CpBlockMappingMode::NONE ? physical_position : physical_position / world_size;
        RTP_LLM_CHECK_WITH_INFO(local_position < block_ids.blocks().size(),
                                "P2P transfer physical block=%zu maps past tag=%.*s local blocks=%zu",
                                physical_position,
                                static_cast<int>(tag.size()),
                                tag.data(),
                                block_ids.blocks().size());
        const size_t global_key = std::min((physical_position + 1) * keys_per_physical_block - 1, last_key);
        RTP_LLM_CHECK_WITH_INFO(global_key < cache_keys.size(),
                                "P2P transfer key ordinal=%zu is past request cache keys=%zu",
                                global_key,
                                cache_keys.size());
        const BlockIdxType block_id = block_ids.blocks()[local_position];
        if (!isNullBlockIdx(block_id)) {
            visitor(cache_keys[global_key], block_id);
            found = true;
        }
    };

    if (mapping == CpBlockMappingMode::COMPACT_LAST_RANK) {
        const size_t compact_begin  = physical_begin / world_size;
        const size_t compact_end    = (physical_end + world_size - 1) / world_size;
        size_t       selected_begin = compact_begin;
        if (group.policy.group_type != CacheGroupType::FULL) {
            const size_t tail =
                group.policy.active_tail_blocks > 0 ? static_cast<size_t>(group.policy.active_tail_blocks) : 1;
            selected_begin = compact_end > tail ? std::max(compact_begin, compact_end - tail) : compact_begin;
        }
        for (size_t compact_position = selected_begin; compact_position < compact_end; ++compact_position) {
            visit_physical(std::min((compact_position + 1) * world_size - 1, physical_end - 1));
        }
        return found;
    }

    size_t selected_begin = physical_begin;
    if (group.policy.group_type != CacheGroupType::FULL) {
        const size_t tail =
            group.policy.active_tail_blocks > 0 ? static_cast<size_t>(group.policy.active_tail_blocks) : 1;
        selected_begin = physical_end > tail ? std::max(physical_begin, physical_end - tail) : physical_begin;
    }
    for (size_t physical_position = selected_begin; physical_position < physical_end; ++physical_position) {
        if (mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN && physical_position % world_size != rank) {
            continue;
        }
        visit_physical(physical_position);
    }
    return found;
}

}  // namespace

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(const CacheConfig& config,
                                                                             KVCacheResource&   resource,
                                                                             int                batch_id,
                                                                             int                start_key_ordinal,
                                                                             int                key_count,
                                                                             int                cp_rank,
                                                                             int                cp_size) {
    return convert(config, resource, batch_id, start_key_ordinal, key_count, cp_rank, cp_size, nullptr);
}

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convert(const CacheConfig&  config,
                                                                             KVCacheResource&    resource,
                                                                             int                 batch_id,
                                                                             int                 start_key_ordinal,
                                                                             int                 key_count,
                                                                             int                 cp_rank,
                                                                             int                 cp_size,
                                                                             ConversionObserver* observer) {
    for (const auto& [tag, block_ids] : resource.blocksByGroup()) {
        (void)block_ids;
        RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "P2P transfer requires a non-empty cache group tag");
        config.group(tag);
    }
    std::vector<std::vector<std::string>> sorted_tags_by_layer;
    sorted_tags_by_layer.reserve(static_cast<size_t>(resource.layerNum()));
    for (int layer_id = 0; layer_id < resource.layerNum(); ++layer_id) {
        auto sorted_tags = resource.groupTagsForLayer(layer_id);
        std::sort(sorted_tags.begin(), sorted_tags.end());
        RTP_LLM_CHECK_WITH_INFO(std::adjacent_find(sorted_tags.begin(), sorted_tags.end()) == sorted_tags.end(),
                                "P2P transfer layer=%d contains duplicate cache group tags",
                                layer_id);
        sorted_tags_by_layer.push_back(std::move(sorted_tags));
    }
    // Whole-set validation: no output object exists while every selected
    // tag/layer/range/CP mapping is checked through the final packing path.
    for (int layer_id = 0; layer_id < resource.layerNum(); ++layer_id) {
        for (const auto& tag : sorted_tags_by_layer[static_cast<size_t>(layer_id)]) {
            visitSelectedBlocks(config,
                                resource,
                                layer_id,
                                tag,
                                start_key_ordinal,
                                key_count,
                                cp_rank,
                                cp_size,
                                [](CacheKeyType, BlockIdxType) {});
        }
    }

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (int layer_id = 0; layer_id < resource.layerNum(); ++layer_id) {
        for (const auto& tag : sorted_tags_by_layer[static_cast<size_t>(layer_id)]) {
            auto buffer = convertLayer(
                config, resource, batch_id, layer_id, tag, start_key_ordinal, key_count, cp_rank, cp_size, observer);
            if (buffer) {
                layer_cache_buffers.push_back(std::move(buffer));
                if (observer != nullptr) {
                    observer->onLayerCacheBufferPublished();
                }
            }
        }
    }
    return layer_cache_buffers;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(const CacheConfig& config,
                                                                     KVCacheResource&   resource,
                                                                     int                batch_id,
                                                                     int                layer_id,
                                                                     std::string_view   tag,
                                                                     int                start_key_ordinal,
                                                                     int                key_count,
                                                                     int                cp_rank,
                                                                     int                cp_size) {
    return convertLayer(
        config, resource, batch_id, layer_id, tag, start_key_ordinal, key_count, cp_rank, cp_size, nullptr);
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(const CacheConfig&  config,
                                                                     KVCacheResource&    resource,
                                                                     int                 batch_id,
                                                                     int                 layer_id,
                                                                     std::string_view    tag,
                                                                     int                 start_key_ordinal,
                                                                     int                 key_count,
                                                                     int                 cp_rank,
                                                                     int                 cp_size,
                                                                     ConversionObserver* observer) {
    (void)batch_id;
    if (!validRangeArguments(layer_id, tag, start_key_ordinal, key_count, cp_rank, cp_size)) {
        return nullptr;
    }
    validateGroupPacking(config, resource, layer_id, tag);
    auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id, std::string(tag));
    if (observer != nullptr) {
        observer->onLayerCacheBufferConstructed();
    }
    visitSelectedBlocks(
        config,
        resource,
        layer_id,
        tag,
        start_key_ordinal,
        key_count,
        cp_rank,
        cp_size,
        [&](CacheKeyType cache_key, BlockIdxType block_id) { layer_cache_buffer->addBlockId(cache_key, block_id); });
    return layer_cache_buffer->blockIdMap().empty() ? nullptr : layer_cache_buffer;
}

bool LayerCacheBufferUtil::hasTransferableBlocks(const CacheConfig&     config,
                                                 const KVCacheResource& resource,
                                                 int                    layer_id,
                                                 std::string_view       tag,
                                                 int                    start_key_ordinal,
                                                 int                    key_count,
                                                 int                    cp_rank,
                                                 int                    cp_size) {
    return visitSelectedBlocks(config,
                               resource,
                               layer_id,
                               tag,
                               start_key_ordinal,
                               key_count,
                               cp_rank,
                               cp_size,
                               [](CacheKeyType, BlockIdxType) {});
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
