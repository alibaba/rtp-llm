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

    size_t       logical_begin = static_cast<size_t>(start_block_idx);
    const size_t logical_end   = block_count > 0 ?
                                     std::min(cache_keys.size(), logical_begin + static_cast<size_t>(block_count)) :
                                     cache_keys.size();
    if (group.policy.active_tail_blocks > 0) {
        const size_t tail_count = static_cast<size_t>(group.policy.active_tail_blocks);
        logical_begin           = std::max(logical_begin, logical_end > tail_count ? logical_end - tail_count : 0);
    }
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

std::shared_ptr<LayerCacheBuffer>
LayerCacheBufferUtil::convertLayerTagForRoute(KVCacheResource&           resource,
                                              const GroupBase&           group,
                                              int                        layer_id,
                                              const std::vector<size_t>& logical_positions,
                                              int                        cp_rank,
                                              int                        cp_size) {
    const auto& cache_keys = resource.cacheKeys();
    if (logical_positions.empty() || cache_keys.empty()) {
        return nullptr;
    }
    const auto& block_ids = resource.blocksForLayer(layer_id, group.tag);
    auto        buffer    = std::make_shared<LayerCacheBuffer>(layer_id, group.tag);

    // 注意：这里**不**施加 group.policy.active_tail_blocks —— logical_positions 已由
    // KVCacheTransferPlanner::resolveKeys 按编排层统一算出的 tail_count 裁剪过。
    for (size_t logical_pos : logical_positions) {
        if (logical_pos >= cache_keys.size()) {
            continue;
        }
        const auto physical_pos =
            CPSlotMapper::physicalBlockPosition(group.policy, logical_pos, cache_keys.size(), cp_rank, cp_size);
        if (!physical_pos || *physical_pos >= block_ids.size() || isNullBlockIdx(block_ids[*physical_pos])) {
            continue;
        }
        buffer->addBlockId(cache_keys[logical_pos], block_ids[*physical_pos]);
    }
    return buffer->blockIdMap().empty() ? nullptr : buffer;
}

std::vector<std::shared_ptr<LayerCacheBuffer>>
LayerCacheBufferUtil::convertTagForRoute(KVCacheResource&           resource,
                                         const CacheTopology&       topology,
                                         const std::string&         cache_tag,
                                         const std::vector<size_t>& logical_positions,
                                         int                        cp_rank,
                                         int                        cp_size) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> result;
    if (logical_positions.empty()) {
        return result;
    }
    const auto& group = topology.group(cache_tag);
    for (int layer_id : group.layer_ids) {
        auto buffer =
            convertLayerTagForRoute(resource, group, layer_id, logical_positions, cp_rank, cp_size);
        if (buffer) {
            result.push_back(std::move(buffer));
        }
    }
    return result;
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

transfer::KeyBlockInfoMap
LayerCacheBufferUtil::buildKeyBlockInfosSliced(const std::shared_ptr<LayerBlockConverter>& converter,
                                               const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                               int                                         partition_count,
                                               int                                         partition_id,
                                               const SliceSpec&                            slice,
                                               size_t                                      k_block_payload_bytes) {
    if (slice.mode == CpBlockSliceMode::NONE || slice.count <= 1) {
        return buildKeyBlockInfos(converter, layer_cache_buffer, partition_count, partition_id);
    }

    transfer::KeyBlockInfoMap key_block_infos;
    const int                 layer_id    = layer_cache_buffer->getLayerId();
    const size_t              slice_count = static_cast<size_t>(slice.count);
    const size_t              slice_index = static_cast<size_t>(std::max(0, slice.index));

    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto block_infos = converter->convertIndexToBuffer(
            layer_id, layer_cache_buffer->cacheTag(), block_id, partition_count, partition_id);
        // sliceBlockForPeer 的前提：被切的 block 必须是单一 BlockInfo。planner 的 Step 1
        // 已用「cp_slice 与 head 分片互斥」保证了这一点；这里再兜一次，不满足就跳过该 key，
        // 让上层因键集不完整而失败，而不是切出错误的字节区间。
        if (block_infos.size() != 1) {
            RTP_LLM_LOG_WARNING("buildKeyBlockInfosSliced: expected 1 block part for cache_key %ld, got %zu; "
                                "cp_slice must not be combined with head partitioning",
                                cache_key,
                                block_infos.size());
            continue;
        }
        auto& block = block_infos[0];

        // 分母必须与 CPSlotMapper::sliceBlockForPeer 一致。
        size_t slice_bytes = 0;
        if (slice.mode == CpBlockSliceMode::PAYLOAD_BYTES) {
            if (k_block_payload_bytes == 0 || k_block_payload_bytes % slice_count != 0) {
                RTP_LLM_LOG_WARNING("buildKeyBlockInfosSliced: payload %zu not divisible by slice count %zu",
                                    k_block_payload_bytes,
                                    slice_count);
                continue;
            }
            slice_bytes = k_block_payload_bytes / slice_count;
        } else {
            if (block.size_bytes % slice_count != 0) {
                RTP_LLM_LOG_WARNING("buildKeyBlockInfosSliced: block bytes %zu not divisible by slice count %zu",
                                    block.size_bytes,
                                    slice_count);
                continue;
            }
            slice_bytes = block.size_bytes / slice_count;
        }
        const size_t slice_offset = slice_bytes * slice_index;
        if (block.addr == nullptr || slice_offset + slice_bytes > block.size_bytes) {
            RTP_LLM_LOG_WARNING("buildKeyBlockInfosSliced: slice [%zu,+%zu) out of block bytes %zu",
                                slice_offset,
                                slice_bytes,
                                block.size_bytes);
            continue;
        }
        block.addr       = static_cast<char*>(block.addr) + slice_offset;
        block.size_bytes = slice_bytes;

        transfer::KeyBlockInfo kbi;
        kbi.cache_key              = cache_key;
        kbi.blocks                 = std::move(block_infos);
        key_block_infos[cache_key] = std::make_shared<const transfer::KeyBlockInfo>(std::move(kbi));
    }
    return key_block_infos;
}

}  // namespace rtp_llm
