#include "rtp_llm/cpp/cache/KVCacheAllocator.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/SWAKVCacheGroup.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

bool KVCacheAllocator::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "init failed");

    // NOTE: the reservable block count depends on initialized block pools and must be queried after `doInit()`.
    const int64_t reserve_ratio = reserve_block_ratio_;
    if (reserve_ratio > 0) {
        const size_t available_blocks = reservableAvailableBlocksNum();
        const size_t reserve_blocks = static_cast<size_t>(reserve_ratio) * available_blocks / static_cast<size_t>(100);
        reserve_block_num_          = reserve_blocks;
        RTP_LLM_LOG_INFO("KVCacheAllocator set reserve blocks: ratio=%ld%% reserve_blocks=%zu available_blocks=%zu",
                         reserve_ratio,
                         reserve_blocks,
                         available_blocks);
    } else {
        reserve_block_num_ = 0;
    }

    return true;
}

MallocResult KVCacheAllocator::initMalloc(const MallocInfo& malloc_info) {
    auto init_result = initMallocForCommonLen(malloc_info);
    if (!init_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return init_result;
    }

    auto incr_result = incrMalloc(malloc_info);
    if (!incr_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return incr_result;
    } else {
        if (metrics_reporter_ && malloc_info.enable_device_cache) {
            int64_t device_input_length = 0;
            if (malloc_info.batch_kv_cache_resource) {
                for (const auto& group : config_.topology().groups()) {
                    const auto key_count = malloc_info.batch_kv_cache_resource->cacheKeys(0, group.tag).size();
                    device_input_length =
                        std::max(device_input_length, static_cast<int64_t>(key_count * group.seq_size_per_block));
                }
            }

            if (device_input_length > 0) {
                RtpLLMDeviceCacheReuseMetricsCollector collector;
                collector.match_cost_time_us    = init_result.match_cost_time_us;
                collector.device_input_length   = device_input_length;
                collector.device_reuse_length   = init_result.reuse_len;
                collector.device_cache_hit_rate = static_cast<float>(static_cast<int64_t>(collector.device_reuse_length)
                                                                     * 100 / collector.device_input_length);
                kmonitor::MetricsTags tags;
                metrics_reporter_->report<RtpLLMDeviceCacheReuseMetrics, RtpLLMDeviceCacheReuseMetricsCollector>(
                    &tags, &collector);
            }
        }
        return init_result;
    }
}

MallocResult KVCacheAllocator::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    if (malloc_info.batch_kv_cache_resource->curBlocksNum() == 0) {
        return initMalloc(malloc_info);
    } else {
        return incrMalloc(malloc_info);
    }
}

int KVCacheAllocator::estimateBatchPeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                  int                            seq_len,
                                                  int                            common_seq_len,
                                                  int                            remaining_tokens,
                                                  int                            reserve_step,
                                                  bool                           enable_reuse_cache,
                                                  int                            target_batch_size) const {
    if (!batch_kv_cache_resource || batch_kv_cache_resource->batchSize() == 0) {
        return 0;
    }

    const int current_batch_size = batch_kv_cache_resource->batchSize();
    const int target_width       = std::max(current_batch_size, target_batch_size);
    const int clamped_common_len = std::clamp(common_seq_len, 0, seq_len);

    // A fresh resource follows initMalloc's two phases. Each group estimates that exact sequence so Linear groups can
    // distinguish the shared common tail from every sequence's private suffix tail.
    if (batch_kv_cache_resource->curBlocksNum() == 0) {
        return estimateInitialBatchPeakNeedBlocks(
            seq_len, clamped_common_len, remaining_tokens, reserve_step, enable_reuse_cache, target_width);
    }

    // Initialized sequences have the same layout, and all subsequent growth is private per sequence.
    const int per_sequence_growth = estimatePeakNeedBlocks(
        batch_kv_cache_resource->cacheResource(0), seq_len, remaining_tokens, reserve_step, enable_reuse_cache);

    // updateKVBlock(copy_last_block=true) replaces the last block independently
    // for every non-empty tagged group on each forked sequence.
    const int expanded_sequences = target_width - current_batch_size;
    int       copied_group_count = 0;
    if (expanded_sequences > 0 && seq_len % seqSizePerBlock() != 0) {
        for (const auto& entry : batch_kv_cache_resource->cacheResource(0).groupResources()) {
            if (!entry.block_ids->blocks().empty()) {
                ++copied_group_count;
            }
        }
    }
    const int tail_copy_blocks = expanded_sequences * copied_group_count;
    return target_width * per_sequence_growth + tail_copy_blocks;
}

uint32_t KVCacheAllocator::convertToGlobalLayerId(size_t model_id, int local_layer_id) const {
    if (model_id == 0) {
        // main model: local_layer_id is the global layer id
        if (local_layer_id >= 0 && static_cast<size_t>(local_layer_id) < config_.layer_num) {
            return static_cast<uint32_t>(local_layer_id);
        }
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: local_layer_id=%d is invalid", local_layer_id);
        return std::numeric_limits<uint32_t>::max();
    }

    if (model_id > config_.mtp_sub_configs.size()) {
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: model_id=%zu out of range (mtp_sub_configs=%zu)",
                          model_id,
                          config_.mtp_sub_configs.size());
        return std::numeric_limits<uint32_t>::max();
    }

    const auto& sub = config_.mtp_sub_configs[model_id - 1];
    if (!sub) {
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: mtp_sub_configs[%zu] is null", model_id - 1);
        return std::numeric_limits<uint32_t>::max();
    }
    if (local_layer_id < 0 || static_cast<size_t>(local_layer_id) >= sub->layer_num) {
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: local_layer_id=%d is invalid", local_layer_id);
        return std::numeric_limits<uint32_t>::max();
    }

    return CacheConfig::mtpGlobalLayerId(
        config_.layer_num, static_cast<int>(model_id - 1), sub->layer_num, local_layer_id);
}

bool KVCacheAllocator::cpShardThisGroupForCapacity(std::string_view tag) const {
    const auto mapper = cpSlotMapper(tag);
    return mapper && mapper->isSharded() && mapper->blockRoundRobinGroup(config_, tag);
}

size_t KVCacheAllocator::logicalSeqSizePerBlockForCapacity(std::string_view tag) const {
    const auto mapper = cpSlotMapper(tag);
    if (mapper && mapper->isSharded()) {
        return mapper->logicalSeqSizePerBlock(config_, tag);
    }
    return config_.seqSizePerBlockForGroup(tag);
}

int KVCacheAllocator::cpEffectiveSeqLenForAlloc(std::string_view tag, int seq_len) const {
    const auto mapper = cpSlotMapper(tag);
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config_, tag, seq_len) : seq_len;
}

}  // namespace rtp_llm

namespace rtp_llm {
namespace {

// CP shard helpers: when mapper is null/passthrough, all helpers no-op.
inline CacheKeysType cpCanonicalCacheKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full) {
    return (mapper && mapper->isSharded()) ? mapper->canonicalCacheKeys(full) : full;
}

inline bool
cpBlockRoundRobinGroup(const std::shared_ptr<CPSlotMapper>& mapper, const CacheConfig& config, std::string_view tag) {
    return mapper && mapper->isSharded() && mapper->blockRoundRobinGroup(config, tag);
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const CacheConfig&                   config,
                                     std::string_view                     tag,
                                     int                                  seq_len) {
    return cpBlockRoundRobinGroup(mapper, config, tag) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) :
                                                         seq_len;
}

inline int cpLogicalSeqSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                    const CacheConfig&                   config,
                                    std::string_view                     tag,
                                    int                                  fallback) {
    return (mapper && mapper->isSharded()) ? static_cast<int>(mapper->logicalSeqSizePerBlock(config, tag)) : fallback;
}

BlockIndicesType validBlocksAfter(const BlockIndicesType& blocks, size_t begin) {
    BlockIndicesType valid;
    if (begin >= blocks.size()) {
        return valid;
    }
    valid.reserve(blocks.size() - begin);
    for (size_t i = begin; i < blocks.size(); ++i) {
        if (!isNullBlockIdx(blocks[i])) {
            valid.push_back(blocks[i]);
        }
    }
    return valid;
}

}  // namespace

bool KVCacheAllocator::skipReuseCacheGroup(std::string_view tag) const {
    const auto it = kv_cache_groups_.find(std::string(tag));
    return it != kv_cache_groups_.end() && !it->second->prefixReuseEnabled();
}

std::vector<std::string> KVCacheAllocator::independentEvictionTags() const {
    std::vector<std::string> tags;
    for (const auto& [tag, group] : kv_cache_groups_) {
        if (group->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            tags.push_back(tag);
        }
    }
    return tags;
}

bool KVCacheAllocator::cpCompactSwaGroup(std::string_view tag, const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && kv_cache_groups_.find(std::string(tag)) != kv_cache_groups_.end()
           && mapper->compactLastRankGroup(config_, tag);
}

int KVCacheAllocator::reuseTokens(BatchKVCacheResource& kv_resource, size_t max_reuse_tokens) {
    uint64_t                                          common_tokens = std::numeric_limits<uint64_t>::max();
    uint64_t                                          boundary_lcm  = 1;
    bool                                              saw_reusable  = false;
    std::unordered_map<std::string, BlockIndicesType> prefix_blocks;

    for (const auto& group : config_.topology().groups()) {
        const auto& tag = group.tag;
        if (skipReuseCacheGroup(tag)) {
            continue;
        }
        saw_reusable                 = true;
        const auto    cp_mapper      = cpSlotMapper(tag);
        const auto&   physical_keys  = kv_resource.cacheKeys(0, tag);
        CacheKeysType match_keys     = cpBlockRoundRobinGroup(cp_mapper, config_, tag) ?
                                           cpCanonicalCacheKeys(cp_mapper, physical_keys) :
                                           physical_keys;
        uint64_t      matched_tokens = 0;
        if (group.policy.group_type == CacheGroupType::FULL) {
            auto result    = kv_cache_groups_.at(tag)->match(match_keys);
            matched_tokens = result.block_indices.size()
                             * static_cast<uint64_t>(cpLogicalSeqSizeForGroup(
                                 cp_mapper, config_, tag, static_cast<int>(group.seq_size_per_block)));
            prefix_blocks.emplace(tag, std::move(result.block_indices));
        } else {
            const uint64_t match_span =
                cpLogicalSeqSizeForGroup(cp_mapper, config_, tag, static_cast<int>(group.seq_size_per_block));
            for (size_t pos = match_keys.size(); pos > 0; --pos) {
                auto result = kv_cache_groups_.at(tag)->matchSingleKey(match_keys[pos - 1]);
                if (!result.block_indices.empty()) {
                    matched_tokens = pos * match_span;
                    break;
                }
            }
        }
        common_tokens                = std::min(common_tokens, matched_tokens);
        const uint64_t boundary_span = cpBlockRoundRobinGroup(cp_mapper, config_, tag) ?
                                           static_cast<uint64_t>(cpLogicalSeqSizeForGroup(
                                               cp_mapper, config_, tag, static_cast<int>(group.seq_size_per_block))) :
                                           static_cast<uint64_t>(group.seq_size_per_block);
        const uint64_t gcd           = std::gcd(boundary_lcm, boundary_span);
        RTP_LLM_CHECK_WITH_INFO(boundary_lcm / gcd <= std::numeric_limits<uint64_t>::max() / boundary_span,
                                "prefix reuse physical boundary LCM overflow");
        boundary_lcm = boundary_lcm / gcd * boundary_span;
    }
    if (!saw_reusable || common_tokens == 0) {
        return 0;
    }
    common_tokens = std::min(common_tokens, static_cast<uint64_t>(max_reuse_tokens));
    common_tokens -= common_tokens % boundary_lcm;
    if (common_tokens == 0) {
        return 0;
    }

    std::unordered_map<std::string, BlockIdxType> boundary_blocks;
    while (common_tokens > 0) {
        bool all_boundaries_match = true;
        boundary_blocks.clear();
        for (const auto& group : config_.topology().groups()) {
            const auto& tag = group.tag;
            if (skipReuseCacheGroup(tag) || group.policy.group_type == CacheGroupType::FULL) {
                continue;
            }
            const auto    cp_mapper       = cpSlotMapper(tag);
            const size_t  retained_span   = cpBlockRoundRobinGroup(cp_mapper, config_, tag) ?
                                                static_cast<size_t>(cpLogicalSeqSizeForGroup(
                                                 cp_mapper, config_, tag, static_cast<int>(group.seq_size_per_block))) :
                                                group.seq_size_per_block;
            const size_t  physical_blocks = static_cast<size_t>(common_tokens / retained_span);
            const auto&   physical_keys   = kv_resource.cacheKeys(0, tag);
            CacheKeysType match_keys      = cpBlockRoundRobinGroup(cp_mapper, config_, tag) ?
                                                cpCanonicalCacheKeys(cp_mapper, physical_keys) :
                                                physical_keys;
            if (physical_blocks == 0 || physical_blocks > match_keys.size()) {
                all_boundaries_match = false;
                break;
            }
            auto tail = kv_cache_groups_.at(tag)->matchSingleKey(match_keys[physical_blocks - 1]);
            if (tail.block_indices.empty()) {
                all_boundaries_match = false;
                break;
            }
            boundary_blocks.emplace(tag, tail.block_indices.front());
        }
        if (all_boundaries_match) {
            break;
        }
        common_tokens -= boundary_lcm;
    }
    if (common_tokens == 0) {
        for (const auto& group : config_.topology().groups()) {
            kv_resource.mutableBlockIds(0, group.tag).resize(0);
        }
        RTP_LLM_LOG_WARNING("prefix reuse found no common materialized boundary; falling back to no reuse");
        return 0;
    }

    for (const auto& group : config_.topology().groups()) {
        const auto& tag = group.tag;
        auto&       ids = kv_resource.mutableBlockIds(0, tag);
        if (skipReuseCacheGroup(tag)) {
            ids.resize(0);
            continue;
        }
        const auto   cp_mapper       = cpSlotMapper(tag);
        const size_t retained_span   = cpBlockRoundRobinGroup(cp_mapper, config_, tag) ?
                                           static_cast<size_t>(cpLogicalSeqSizeForGroup(
                                             cp_mapper, config_, tag, static_cast<int>(group.seq_size_per_block))) :
                                           group.seq_size_per_block;
        const size_t physical_blocks = static_cast<size_t>(common_tokens / retained_span);
        if (group.policy.group_type == CacheGroupType::FULL) {
            auto blocks = std::move(prefix_blocks.at(tag));
            blocks.resize(std::min(blocks.size(), physical_blocks));
            ids.assign(std::move(blocks));
            continue;
        }
        ids.assign(BlockIndicesType(physical_blocks, NULL_BLOCK_IDX));
        ids.setAt(physical_blocks - 1, boundary_blocks.at(tag));
    }
    return static_cast<int>(common_tokens);
}

MallocResult KVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int    seq_len            = malloc_info.complete_token_ids->seqLength();
    const int    common_seq_len     = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    int64_t      match_cost_time_us = 0;
    const size_t reserve_blocks     = reserveBlocksNum();
    int          reuse_tokens       = 0;
    std::unordered_map<std::string, BlockIndicesType> referenced_blocks;

    if (malloc_info.enable_device_cache) {
        auto         begin_us         = currentTimeUs();
        const size_t max_reuse_tokens = seq_len > 0 ? static_cast<size_t>(seq_len - 1) : 0;
        reuse_tokens                  = reuseTokens(*kv_resource, max_reuse_tokens);
        match_cost_time_us            = currentTimeUs() - begin_us;

        for (const auto& entry : kv_resource->groupResources()) {
            const auto&      tag    = entry.tag;
            const auto&      blocks = entry.block_ids->blocks();
            BlockIndicesType valid;
            valid.reserve(blocks.size());
            for (auto b : blocks) {
                if (!isNullBlockIdx(b)) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                referenceBlocksInGroup(tag, valid);
                referenced_blocks.emplace(tag, std::move(valid));
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseTokenNum(static_cast<size_t>(reuse_tokens));
    }

    if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        rollbackInitMalloc(*kv_resource, referenced_blocks, {});
        return {false, 0};
    }

    std::unordered_map<std::string, size_t> original_sizes;
    for (const auto& entry : kv_resource->groupResources()) {
        original_sizes.emplace(entry.tag, entry.block_ids->blocksNum());
    }
    for (const auto& entry : kv_resource->groupResources()) {
        const auto& tag           = entry.tag;
        auto&       block_ids_0   = kv_resource->mutableBlockIds(0, tag);
        const int   group_seq_len = cpEffectiveSeqLenForGroup(cpSlotMapper(tag), config_, tag, common_seq_len);
        if (!kv_cache_groups_.at(tag)->malloc(block_ids_0, group_seq_len, malloc_info.reuse_cache, 0)) {
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (const auto& entry : kv_resource->groupResources()) {
            kv_cache_groups_.at(entry.tag)->reference(kv_resource->mutableBlockIds(b, entry.tag),
                                                      kv_resource->blocks(0, entry.tag));
        }
    }
    return {true, reuse_tokens, match_cost_time_us};
}

MallocResult KVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&     kv_resource  = malloc_info.batch_kv_cache_resource;
    const int batch_size   = kv_resource->batchSize();
    const int raw_seq_len  = malloc_info.incrSeqLen();
    const int reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::unordered_map<std::string, size_t>>              original_sizes(static_cast<size_t>(batch_size));
    std::vector<std::unordered_map<std::string, std::vector<size_t>>> backfilled_positions(
        static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& entry : kv_resource->groupResources(b)) {
            original_sizes[static_cast<size_t>(b)].emplace(entry.tag, entry.block_ids->blocksNum());
        }
    }

    bool        all_success  = true;
    int         failed_batch = -1;
    std::string failed_tag;
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& entry : kv_resource->groupResources(b)) {
            const auto& tag              = entry.tag;
            auto&       block_ids        = kv_resource->mutableBlockIds(b, tag);
            const int   group_seq_len    = cpEffectiveSeqLenForGroup(cpSlotMapper(tag), config_, tag, raw_seq_len);
            auto&       filled_positions = backfilled_positions[static_cast<size_t>(b)][tag];
            if (!kv_cache_groups_.at(tag)->malloc(
                    block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step, &filled_positions)) {
                all_success  = false;
                failed_batch = b;
                failed_tag   = tag;
                break;
            }
        }
        if (!all_success) {
            break;
        }
    }

    if (all_success) {
        if (!malloc_info.enable_remove_skipped_blocks) {
            return {true, 0};
        }
        for (int b = 0; b < batch_size; ++b) {
            for (const auto& entry : kv_resource->groupResources(b)) {
                kv_cache_groups_.at(entry.tag)->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, entry.tag), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (const auto& entry : kv_resource->groupResources(b)) {
            const auto&      tag              = entry.tag;
            auto&            block_ids        = kv_resource->mutableBlockIds(b, tag);
            const auto       original_size    = original_sizes[static_cast<size_t>(b)].at(tag);
            const auto&      filled_positions = backfilled_positions[static_cast<size_t>(b)][tag];
            const auto&      blocks           = block_ids.blocks();
            BlockIndicesType blocks_to_free;
            blocks_to_free.reserve(filled_positions.size() + blocks.size() - std::min(original_size, blocks.size()));
            for (size_t pos : filled_positions) {
                RTP_LLM_CHECK_WITH_INFO(pos < original_size && pos < blocks.size(),
                                        "invalid hybrid rollback backfill position=%zu original_size=%zu size=%zu",
                                        pos,
                                        original_size,
                                        blocks.size());
                if (!isNullBlockIdx(blocks[pos])) {
                    blocks_to_free.push_back(blocks[pos]);
                }
            }
            for (size_t pos = original_size; pos < blocks.size(); ++pos) {
                const auto block = blocks[pos];
                if (!isNullBlockIdx(block)) {
                    blocks_to_free.push_back(block);
                }
            }
            if (!blocks_to_free.empty()) {
                freeBlocksInGroup(tag, blocks_to_free);
            }
            for (size_t pos : filled_positions) {
                block_ids.setAt(pos, NULL_BLOCK_IDX);
            }
            block_ids.resize(original_size);
        }
    }
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d tag=%s", failed_batch, failed_tag.c_str());
    return {false, 0};
}

void KVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (const auto& entry : kv_cache_resource->groupResources(batch_id)) {
            kv_cache_groups_.at(entry.tag)->free(entry.block_ids->blocks());
        }
    }
    kv_cache_resource->clearBlocks();
}

void KVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!shared_block_cache_) {
        return;
    }

    const int batch_size = kv_cache_resource->batchSize();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        const size_t token_len = token_ids.size() - 1;

        for (const auto& entry : kv_cache_resource->groupResources(batch_id)) {
            const auto& tag = entry.tag;
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            auto& resource = kv_cache_resource->cacheResource(batch_id);
            resource.ensureLinearBlockDependencies(tag);
            const auto& physical_keys = resource.cacheKeys(tag);
            const auto  cp_mapper     = cpSlotMapper(tag);
            const bool  use_cp_keys =
                cpBlockRoundRobinGroup(cp_mapper, config_, tag) || cpCompactSwaGroup(tag, cp_mapper);
            CacheKeysType cp_keys = use_cp_keys ? cpCanonicalCacheKeys(cp_mapper, physical_keys) : CacheKeysType{};
            const CacheKeysType& src_keys = use_cp_keys ? cp_keys : physical_keys;
            if (src_keys.empty()) {
                continue;
            }
            BlockDependenciesType dependencies;
            dependencies.reserve(src_keys.size());
            const auto& physical_dependencies = resource.blockDependencies(tag);
            for (size_t i = 0; i < src_keys.size(); ++i) {
                const auto      key_it       = std::find(physical_keys.begin(), physical_keys.end(), src_keys[i]);
                const size_t    physical_pos = key_it == physical_keys.end() ?
                                                   i :
                                                   static_cast<size_t>(std::distance(physical_keys.begin(), key_it));
                BlockDependency dependency =
                    physical_pos < physical_dependencies.size() ?
                        physical_dependencies[physical_pos] :
                        BlockDependency{i > 0, i > 0 ? src_keys[i - 1] : 0, static_cast<uint32_t>(physical_pos)};
                dependency.has_parent = i > 0;
                dependency.parent_key = i > 0 ? src_keys[i - 1] : 0;
                dependencies.push_back(dependency);
            }
            const auto namespace_id =
                use_cp_keys ? SharedBlockCache::kGpuCpCanonicalNamespace : SharedBlockCache::kGpuLogicalNamespace;
            const int group_seq_size =
                cpLogicalSeqSizeForGroup(cp_mapper, config_, tag, kv_cache_groups_.at(tag)->seqSizePerBlock());
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(src_keys.size(), full_blocks_num);
            const auto&  blocks          = entry.block_ids->blocks();
            const size_t loop_end        = std::min(n, blocks.size());

            for (size_t pos = loop_end; pos > 0; --pos) {
                const size_t i = pos - 1;
                if (isNullBlockIdx(blocks[i])) {
                    continue;
                }
                std::vector<SharedBlockCache::UnifiedCacheItem::GroupBlock> group_blocks{
                    {tag, blocks[i], true, 0, dependencies[i].ordinal}};
                shared_block_cache_->put(
                    src_keys[i], group_blocks, insert_info.is_resident, namespace_id, dependencies[i]);
            }
        }
    }
}

std::shared_ptr<KVCacheResource> KVCacheAllocator::incrKVCacheRef(const KVCacheResource&  kvcache_resource,
                                                                  const CacheKeysByGroup& cache_keys_by_group,
                                                                  bool                    is_connector) {
    if (cache_keys_by_group.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(config_.topologyPtr());
    selected_resource->requestPrefix() = kvcache_resource.requestPrefix();

    bool selected_any = false;
    for (const auto& entry : kvcache_resource.groupResources()) {
        const auto& tag          = entry.tag;
        const auto  requested_it = cache_keys_by_group.find(tag);
        if (requested_it == cache_keys_by_group.end()) {
            continue;
        }
        std::optional<KVCacheResource> projected_resource;
        const KVCacheResource*         source_resource = &kvcache_resource;
        const auto                     cp_mapper       = cpSlotMapper(tag);
        if (cp_mapper && cp_mapper->isSharded() && cp_mapper->usesCpCanonicalKeys(config_, tag)
            && !kvcache_resource.cacheKeysAreCpCanonical(tag)) {
            projected_resource =
                cp_mapper->projectConnectorResource(kvcache_resource, config_, tag, requested_it->second);
            source_resource = &projected_resource.value();
        }
        std::unordered_map<CacheKeyType, size_t> key_to_pos;
        const auto&                              source_keys = source_resource->cacheKeys(tag);
        for (size_t pos = 0; pos < source_keys.size(); ++pos) {
            key_to_pos.emplace(source_keys[pos], pos);
        }
        CacheKeysType         selected_keys;
        BlockDependenciesType selected_dependencies;
        BlockIndicesType      selected_blocks;
        const auto&           source_blocks       = source_resource->blocks(tag);
        const auto&           source_dependencies = source_resource->blockDependencies(tag);
        for (const auto key : requested_it->second) {
            const auto pos_it = key_to_pos.find(key);
            if (pos_it == key_to_pos.end()) {
                continue;
            }
            const size_t pos                     = pos_it->second;
            const auto   block                   = pos < source_blocks.size() ? source_blocks[pos] : NULL_BLOCK_IDX;
            const bool   preserve_connector_tail = is_connector && !source_resource->lastBlockAligned(tag)
                                                 && pos + 1 == source_keys.size() && !selected_keys.empty();
            if ((isNullBlockIdx(block) || block <= 0) && !preserve_connector_tail) {
                continue;
            }
            selected_keys.push_back(key);
            selected_blocks.push_back(block);
            selected_dependencies.push_back(
                pos < source_dependencies.size() ?
                    source_dependencies[pos] :
                    BlockDependency{false, 0, static_cast<uint32_t>(selected_dependencies.size())});
        }
        if (selected_keys.empty()) {
            continue;
        }
        selected_resource->setCacheKeys(tag, std::move(selected_keys));
        selected_resource->setCacheKeysAreCpCanonical(tag, source_resource->cacheKeysAreCpCanonical(tag));
        selected_resource->setBlockDependencies(tag, std::move(selected_dependencies));
        BlockIndicesType valid;
        for (auto b : selected_blocks) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocksInGroup(tag, valid, is_connector);
        }
        selected_resource->mutableBlockIds(tag).assign(std::move(selected_blocks));
        selected_resource->setLastBlockAligned(tag, source_resource->lastBlockAligned(tag));
        selected_any = true;
    }
    return selected_any ? selected_resource : nullptr;
}

void KVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (const auto& entry : kvcache_resource.groupResources()) {
        BlockIndicesType valid;
        for (auto b : entry.block_ids->blocks()) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            freeBlocksInGroup(entry.tag, valid, is_connector);
        }
    }
}

bool KVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                     const std::vector<int>&        block_src_batch,
                                     bool                           copy_last_block,
                                     std::vector<GroupBlockIdPair>& block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }

    const int        old_batch_size = batch_kv_cache_resource->batchSize();
    const int        new_batch_size = static_cast<int>(block_src_batch.size());
    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx >= 0 && old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    std::unordered_map<std::string, int> new_blocks_num;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count > 1 && copy_last_block) {
            for (const auto& entry : batch_kv_cache_resource->groupResources(old_batch_idx)) {
                if (!entry.block_ids->blocks().empty()) {
                    new_blocks_num[entry.tag] += fork_count - 1;
                }
            }
        }
    }

    // Transfer request ownership from dropped batches before allocating new
    // blocks. This keeps the operation transactional while allowing net-feasible
    // drop-and-fork updates to succeed when the pool is otherwise full.
    std::unordered_map<std::string, BlockIndicesType>                      replacement_blocks;
    std::unordered_map<std::string, BlockIndicesType>                      allocated_replacements;
    std::unordered_map<std::string, std::unordered_map<BlockIdxType, int>> transferred_ref_counts;
    for (const auto& group_entry : batch_kv_cache_resource->groupResources()) {
        const auto&                           tag = group_entry.tag;
        std::unordered_set<BlockIdxType>      retained_blocks;
        std::unordered_map<BlockIdxType, int> dropped_block_counts;
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, tag)) {
                if (isNullBlockIdx(block) || block <= 0) {
                    continue;
                }
                if (batch_fork_count[old_batch_idx] == 0) {
                    ++dropped_block_counts[block];
                } else {
                    retained_blocks.insert(block);
                }
            }
        }

        auto&     replacements = replacement_blocks[tag];
        auto&     transferred  = transferred_ref_counts[tag];
        const int need         = new_blocks_num[tag];
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size && static_cast<int>(replacements.size()) < need;
             ++old_batch_idx) {
            if (batch_fork_count[old_batch_idx] != 0) {
                continue;
            }
            const auto& dropped = batch_kv_cache_resource->blocks(old_batch_idx, tag);
            if (dropped.empty()) {
                continue;
            }
            const auto block = dropped.back();
            if (!isNullBlockIdx(block) && block > 0 && dropped_block_counts[block] == 1 && !retained_blocks.count(block)
                && !transferred.count(block)) {
                replacements.push_back(block);
                transferred[block] = 1;
            }
        }
    }

    auto rollback_replacements = [&]() {
        for (auto& [tag, blocks] : allocated_replacements) {
            if (!blocks.empty()) {
                kv_cache_groups_.at(tag)->free(blocks);
                blocks.clear();
            }
        }
    };
    for (const auto& group_entry : batch_kv_cache_resource->groupResources()) {
        const auto& tag         = group_entry.tag;
        const int   need_blocks = new_blocks_num[tag];
        auto&       reserved    = replacement_blocks[tag];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            BlockIds    one_block;
            const bool  ok = kv_cache_groups_.at(tag)->malloc(one_block, kv_cache_groups_.at(tag)->seqSizePerBlock());
            const auto& blocks = one_block.blocks();
            if (ok && blocks.size() == 1 && !isNullBlockIdx(blocks.front())) {
                reserved.push_back(blocks.front());
                allocated_replacements[tag].push_back(blocks.front());
                continue;
            }
            if (!blocks.empty()) {
                allocated_replacements[tag].insert(allocated_replacements[tag].end(), blocks.begin(), blocks.end());
            }
            RTP_LLM_LOG_WARNING(
                "reserve replacement block failed for hybrid kv cache update, tag=%s need=%d reserved=%zu",
                tag.c_str(),
                need_blocks,
                reserved.size());
            rollback_replacements();
            return false;
        }
    }

    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        if (batch_fork_count[old_batch_idx] != 0) {
            continue;
        }
        for (const auto& entry : batch_kv_cache_resource->groupResources(old_batch_idx)) {
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[entry.tag];
            for (const auto block : entry.block_ids->blocks()) {
                if (isNullBlockIdx(block) || block <= 0) {
                    continue;
                }
                auto it = transferred.find(block);
                if (it != transferred.end() && it->second > 0) {
                    --it->second;
                } else {
                    to_free.push_back(block);
                }
            }
            if (!to_free.empty()) {
                kv_cache_groups_.at(entry.tag)->free(to_free);
            }
        }
    }

    std::vector<KVCacheResource> old_resources;
    batch_kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);
    batch_kv_cache_resource->initGroups(config_.topologyPtr());
    std::unordered_map<std::string, size_t> next_replacement;

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            batch_kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            for (const auto& entry : old_resources[old_batch_idx].groupResources()) {
                const auto& tag = entry.tag;
                batch_kv_cache_resource->setBatchCacheKeys(
                    new_batch_idx, tag, old_resources[old_batch_idx].cacheKeys(tag));
                auto& cloned_resource = batch_kv_cache_resource->cacheResource(new_batch_idx);
                cloned_resource.setBlockDependencies(tag, old_resources[old_batch_idx].blockDependencies(tag));
                cloned_resource.setCacheKeysAreCpCanonical(tag,
                                                           old_resources[old_batch_idx].cacheKeysAreCpCanonical(tag));
                cloned_resource.setLastBlockAligned(tag, old_resources[old_batch_idx].lastBlockAligned(tag));
                auto& block_ids = batch_kv_cache_resource->mutableBlockIds(new_batch_idx, tag);
                kv_cache_groups_.at(tag)->reference(block_ids, entry.block_ids->blocks());

                if (copy_last_block && !block_ids.blocks().empty()) {
                    const int  old_block       = block_ids.popBack();
                    const bool old_block_valid = !isNullBlockIdx(old_block) && old_block > 0;
                    if (old_block_valid) {
                        kv_cache_groups_.at(tag)->free({old_block});
                    }

                    auto&      reserved     = replacement_blocks[tag];
                    const auto reserved_idx = next_replacement[tag]++;
                    RTP_LLM_CHECK_WITH_INFO(reserved_idx < reserved.size(),
                                            "missing reserved replacement block for hybrid kv cache update, tag=%s",
                                            tag.c_str());
                    const int new_block = reserved[reserved_idx];
                    block_ids.add({new_block});
                    if (old_block_valid && !isNullBlockIdx(new_block) && new_block > 0) {
                        block_update_mapping.push_back({tag, old_block, new_block});
                    }
                }
            }
            batch_kv_cache_resource->cacheResource(new_batch_idx).requestPrefix() =
                old_resources[old_batch_idx].requestPrefix();
        }
        --fork_count;
    }
    for (const auto& [tag, blocks] : replacement_blocks) {
        RTP_LLM_CHECK_WITH_INFO(next_replacement[tag] == blocks.size(),
                                "unused replacement blocks after hybrid kv cache update, tag=%s used=%zu reserved=%zu",
                                tag.c_str(),
                                next_replacement[tag],
                                blocks.size());
    }
    return true;
}

int KVCacheAllocator::seqSizePerBlock() const {
    RTP_LLM_CHECK_WITH_INFO(reuse_span_tokens_ > 0, "hybrid cache reuse span is not initialized");
    return reuse_span_tokens_;
}

void KVCacheAllocator::initializeReuseSpan() {
    uint64_t span               = 1;
    bool     saw_reusable_group = false;
    for (const auto& group : config_.topology().groups()) {
        if (!group.policy.enable_prefix_reuse) {
            continue;
        }
        saw_reusable_group        = true;
        const uint64_t group_span = logicalSeqSizePerBlockForCapacity(group.tag);
        const uint64_t gcd        = std::gcd(span, group_span);
        RTP_LLM_CHECK_WITH_INFO(span / gcd <= std::numeric_limits<uint64_t>::max() / group_span,
                                "reusable cache span LCM overflow");
        span = span / gcd * group_span;
    }
    if (!saw_reusable_group) {
        for (const auto& group : config_.topology().groups()) {
            const uint64_t group_span = logicalSeqSizePerBlockForCapacity(group.tag);
            const uint64_t gcd        = std::gcd(span, group_span);
            RTP_LLM_CHECK_WITH_INFO(span / gcd <= std::numeric_limits<uint64_t>::max() / group_span,
                                    "cache span LCM overflow");
            span = span / gcd * group_span;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(span > 0 && span <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                            "invalid cache reuse span %lu",
                            span);
    reuse_span_tokens_ = static_cast<int>(span);
}

std::pair<size_t, size_t> KVCacheAllocator::layerCopyStrides(std::string_view tag, int layer_id) const {
    const auto& group = config_.physicalGroupForLayer(layer_id, tag);
    return {group.kv_block_stride_bytes, group.kv_scale_stride_bytes};
}

void KVCacheAllocator::rollbackBlockIdsToSize(std::string_view tag, BlockIds& block_ids, size_t original_size) {
    if (block_ids.blocksNum() <= original_size) {
        return;
    }
    const auto blocks_to_free = validBlocksAfter(block_ids.blocks(), original_size);
    block_ids.resize(original_size);
    if (!blocks_to_free.empty()) {
        freeBlocksInGroup(tag, blocks_to_free);
    }
}

void KVCacheAllocator::rollbackInitMalloc(BatchKVCacheResource&                                    kv_resource,
                                          const std::unordered_map<std::string, BlockIndicesType>& referenced_blocks,
                                          const std::unordered_map<std::string, size_t>&           original_sizes) {
    for (const auto& entry : kv_resource.groupResources()) {
        const auto& tag       = entry.tag;
        auto&       block_ids = kv_resource.mutableBlockIds(0, tag);
        const auto  original  = original_sizes.find(tag);
        if (original != original_sizes.end() && block_ids.blocksNum() > original->second) {
            rollbackBlockIdsToSize(tag, block_ids, original->second);
        }
        const auto referenced = referenced_blocks.find(tag);
        if (referenced != referenced_blocks.end() && !referenced->second.empty()) {
            freeBlocksInGroup(tag, referenced->second);
        }
        block_ids.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseTokenNum(0);
}

void KVCacheAllocator::rollbackIncrMalloc(BatchKVCacheResource&                                       kv_resource,
                                          const std::vector<std::unordered_map<std::string, size_t>>& original_sizes,
                                          int                                                         failed_batch) {
    const int last_touched_batch = std::min(failed_batch, kv_resource.batchSize() - 1);
    for (int b = 0; b <= last_touched_batch; ++b) {
        for (const auto& entry : kv_resource.groupResources(b)) {
            auto&        block_ids    = kv_resource.mutableBlockIds(b, entry.tag);
            const size_t original_num = original_sizes[static_cast<size_t>(b)].at(entry.tag);
            rollbackBlockIdsToSize(entry.tag, block_ids, original_num);
        }
    }
}

MemoryType KVCacheAllocator::memoryTypeForGroup(std::string_view tag) const {
    (void)tag;
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

void KVCacheAllocator::copyBlockMappingForGroup(std::string_view                tag,
                                                const std::vector<BlockIdPair>& block_update_mapping) const {
    if (block_update_mapping.empty()) {
        return;
    }

    const auto      memory_type = memoryTypeForGroup(tag);
    const auto      copy_type   = BatchCopyParams::get_copy_type(memory_type, memory_type);
    BatchCopyParams copy_params;
    size_t          buffers_per_mapping = 0;
    for (int layer_id : config_.layerIdsForGroup(tag)) {
        const auto [kv_stride, scale_stride] = layerCopyStrides(tag, layer_id);
        (void)kv_stride;
        buffers_per_mapping += scale_stride > 0 ? 2 : 1;
    }
    copy_params.reserve(copy_type, buffers_per_mapping * block_update_mapping.size());

    for (const auto& [src_block_index, dest_block_index] : block_update_mapping) {
        for (int layer_id : config_.layerIdsForGroup(tag)) {
            const auto [kv_block_size_bytes, scale_block_bytes] = layerCopyStrides(tag, layer_id);
            auto src_addr_info = kv_cache_groups_.at(std::string(tag))->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = kv_cache_groups_.at(std::string(tag))->convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "failed to get block address for tag %s layer %d src_block %d dst_block %d",
                                    std::string(tag).c_str(),
                                    layer_id,
                                    src_block_index,
                                    dest_block_index);

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
            }
        }
    }

    execBatchCopy(copy_params);
}

int KVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const int  batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int  total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int  raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int  raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool reuse_enabled      = malloc_info.reuse_cache;
    const int  reuse_blocks_len   = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto cp_mapper        = cpSlotMapper(tag);
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
        const auto need =
            group->getNeedBlocks(group_common_seq, group_seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

int KVCacheAllocator::estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                             int                    seq_len,
                                             int                    remaining_tokens,
                                             int                    reserve_step,
                                             bool                   enable_reuse_cache) const {
    int need_blocks = 0;
    for (const auto& entry : kv_cache_resource.groupResources()) {
        need_blocks += kv_cache_groups_.at(entry.tag)->estimatePeakNeedBlocks(
            seq_len, entry.block_ids->blocks(), remaining_tokens, reserve_step, enable_reuse_cache);
    }
    return need_blocks;
}

int KVCacheAllocator::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                         int  common_seq_len,
                                                         int  remaining_tokens,
                                                         int  reserve_step,
                                                         bool enable_reuse_cache,
                                                         int  target_batch_size) const {
    int peak_blocks = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        peak_blocks += group->estimateInitialBatchPeakNeedBlocks(
            seq_len, common_seq_len, remaining_tokens, reserve_step, enable_reuse_cache, target_batch_size);
    }
    return peak_blocks;
}

void KVCacheAllocator::checkCPShardedMallocResult(const MallocInfo& malloc_info) const {
    const auto& kv_resource  = malloc_info.batch_kv_cache_resource;
    const int   seq_len      = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    for (int batch_id = 0; batch_id < kv_resource->batchSize(); ++batch_id) {
        for (const auto& entry : kv_resource->groupResources(batch_id)) {
            const auto& tag       = entry.tag;
            const auto  cp_mapper = cpSlotMapper(tag);
            if (!cpBlockRoundRobinGroup(cp_mapper, config_, tag)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, seq_len);
            const int expected_blocks   = kv_cache_groups_.at(tag)->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks     = kv_resource->blocksNum(batch_id, tag);
            RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                    "CP invariant violated: batch=%d tag=%s blocks=%d != expected_local_blocks=%d "
                                    "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                    "block_size=%d, cacheKeys=%zu)",
                                    batch_id,
                                    tag.c_str(),
                                    actual_blocks,
                                    expected_blocks,
                                    seq_len,
                                    effective_seq_len,
                                    reserve_step,
                                    cp_mapper->cpSize(),
                                    cp_mapper->blockSize(),
                                    kv_resource->cacheKeys(batch_id, tag).size());
        }
    }
}

int KVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                            int                            seq_len,
                                            int                            reserve_step) const {
    int need_blocks = 0;
    for (const auto& entry : batch_kv_cache_resource->groupResources()) {
        const int effective_seq_len = cpEffectiveSeqLenForGroup(cpSlotMapper(entry.tag), config_, entry.tag, seq_len);
        const int cur_blocks        = batch_kv_cache_resource->blocksNum(0, entry.tag);
        need_blocks += kv_cache_groups_.at(entry.tag)->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm

namespace rtp_llm {
namespace {

inline int cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper,
                                       const CacheConfig&                   config,
                                       std::string_view                     tag,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) : seq_len;
}

void appendPoolSummary(std::ostringstream&    os,
                       bool&                  has_any,
                       const std::string&     tag,
                       CacheGroupType         group_type,
                       const BlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", tag=" << tag << ", type=" << cacheGroupTypeName(group_type)
       << ", size=" << pool_config.total_size_bytes << " bytes(" << std::fixed << std::setprecision(2)
       << static_cast<double>(pool_config.total_size_bytes) / kBytesPerMB << " MB)"
       << ", blocks=" << pool_config.block_num;
}

}  // namespace

KVCacheAllocator::KVCacheAllocator(const CacheConfig&                 config,
                                   AllocationType                     allocation_type,
                                   const kmonitor::MetricsReporterPtr metrics_reporter,
                                   int64_t                            reserve_block_ratio,
                                   RoleType                           role_type):
    config_(config),
    allocation_type_(allocation_type),
    metrics_reporter_(metrics_reporter),
    reserve_block_ratio_(reserve_block_ratio),
    role_type_(role_type) {}

bool KVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");
    initializeReuseSpan();

    SharedBlockCache*       shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    static constexpr double kBytesPerMB      = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    for (const auto& cache_group : config_.topology().groups()) {
        auto pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, cache_group.tag);
        appendPoolSummary(pool_summary, has_pool, cache_group.tag, cache_group.policy.group_type, pool_config);
        pool_total_bytes += pool_config.total_size_bytes;
        pool_total_blocks += pool_config.block_num;
        const auto group_type = cache_group.policy.group_type;
        auto       group_pool =
            std::make_shared<BlockPool>(pool_config, allocation_type_, false, use_cuda_malloc_block_pool_);
        RTP_LLM_CHECK_WITH_INFO(
            group_pool->init(), "Failed to initialize block pool %s", pool_config.pool_name.c_str());

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(
                cache_group, group_pool, config_.linear_step, shared_cache_raw, metrics_reporter_);
            linear_group_tags_.push_back(cache_group.tag);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                cache_group, group_pool, config_.linear_step, shared_cache_raw, metrics_reporter_);
            swa_group_tags_.push_back(cache_group.tag);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, group_pool, shared_cache_raw, metrics_reporter_);
            full_group_tags_.push_back(cache_group.tag);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup %s", pool_config.pool_name.c_str());
        RTP_LLM_CHECK(kv_cache_groups_.emplace(cache_group.tag, std::move(group)).second);
    }

    if (has_pool) {
        const auto summary = pool_summary.str();
        RTP_LLM_LOG_INFO("HybridPool pool summary: pools=[%s], total_size=%zu bytes total_size_mb=%.2f "
                         "total_blocks=%zu",
                         summary.c_str(),
                         pool_total_bytes,
                         static_cast<double>(pool_total_bytes) / kBytesPerMB,
                         pool_total_blocks);
    }

    if (shared_block_cache_) {
        std::vector<SharedBlockCache::GroupPool> pools;
        pools.reserve(kv_cache_groups_.size());
        for (const auto& [tag, group] : kv_cache_groups_) {
            const auto& pool = group->blockPool();
            pools.push_back({tag, pool});
        }
        shared_block_cache_->init(pools);
    }

    RTP_LLM_LOG_INFO("KVCacheAllocator init success, group pools=%zu", kv_cache_groups_.size());
    return true;
}

namespace {
const GroupBase& validatePoolGroupForLayer(const CacheConfig& config, int layer_id, std::string_view tag) {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_all_num,
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config.layer_all_num);
    return config.topology().groupForLayer(layer_id, tag);
}
}  // namespace

void KVCacheAllocator::referenceBlocksInGroup(std::string_view        tag,
                                              const BlockIndicesType& blocks,
                                              bool                    is_connector) const {
    if (is_connector) {
        kv_cache_groups_.at(std::string(tag))->blockPool()->connectorReference(blocks);
    } else {
        kv_cache_groups_.at(std::string(tag))->blockPool()->requestReference(blocks);
    }
}

void KVCacheAllocator::freeBlocksInGroup(std::string_view tag, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        kv_cache_groups_.at(std::string(tag))->blockPool()->connectorFree(blocks);
    } else {
        kv_cache_groups_.at(std::string(tag))->blockPool()->requestFree(blocks);
    }
}

GroupedCacheLayerLayout KVCacheAllocator::allLayerCacheBase() const {
    const auto topology = config_.topologyPtr();
    RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.size() == topology->groups().size(),
                            "cache group count=%zu topology count=%zu",
                            kv_cache_groups_.size(),
                            topology->groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& [tag, cache_group] : kv_cache_groups_) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        const auto                      layer_tensors = cache_group->allLayerCacheBase();
        const auto                      scale_tensors = cache_group->allLayerScaleCacheBase();
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "layer_id %d out of group kv layout range %zu",
                                    layer_id,
                                    layers.size());
            layers[static_cast<size_t>(layer_id)].kv_addr = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "layer_id %d out of group scale layout range %zu",
                                    layer_id,
                                    layers.size());
            layers[static_cast<size_t>(layer_id)].kv_scale_addr = tensor;
        }
        groups.emplace(tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo KVCacheAllocator::convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const {
    validatePoolGroupForLayer(config_, layer_id, tag);
    return kv_cache_groups_.at(tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
KVCacheAllocator::convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const {
    validatePoolGroupForLayer(config_, layer_id, tag);
    return kv_cache_groups_.at(tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> KVCacheAllocator::convertIndexToBuffer(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    validatePoolGroupForLayer(config_, layer_id, tag);
    return kv_cache_groups_.at(tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void KVCacheAllocator::blockBatchCopy(const std::vector<GroupBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (const auto& mapping : copy_mapping) {
        const auto& pool      = kv_cache_groups_.at(mapping.tag)->blockPool();
        const auto  copy_type = BatchCopyParams::get_copy_type(pool->where(), pool->where());
        for (int layer_id : config_.layerIdsForGroup(mapping.tag)) {
            const auto [kv_stride, scale_stride] = layerCopyStrides(mapping.tag, layer_id);
            (void)kv_stride;
            copy_nums[copy_type] += scale_stride > 0 ? 2 : 1;
        }
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const auto& pool      = kv_cache_groups_.at(mapping.tag)->blockPool();
        const auto  copy_type = BatchCopyParams::get_copy_type(pool->where(), pool->where());

        for (int layer_id : config_.layerIdsForGroup(mapping.tag)) {
            const auto [kv_block_size_bytes, scale_block_bytes] = layerCopyStrides(mapping.tag, layer_id);
            auto src_addr_info = kv_cache_groups_.at(mapping.tag)->convertIndexToAddr(layer_id, mapping.src);
            auto dst_addr_info = kv_cache_groups_.at(mapping.tag)->convertIndexToAddr(layer_id, mapping.dst);

            if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for pool %s(tag %s) layer %d, src_block %d, "
                                  "dst_block %d",
                                  pool->poolName().c_str(),
                                  mapping.tag.c_str(),
                                  layer_id,
                                  mapping.src,
                                  mapping.dst);
                continue;
            }

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
            }
        }
    }

    execBatchCopy(copy_params);
}

size_t KVCacheAllocator::freeBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->freeBlocksNum();
    }
    return total;
}

size_t KVCacheAllocator::availableBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->availableBlocksNum();
    }
    return total;
}

BatchKVCacheResourcePtr KVCacheAllocator::popBlocksFromCache(size_t min_blocks_to_free) {
    if (min_blocks_to_free == 0 || !shared_block_cache_) {
        return nullptr;
    }

    auto evict_result = shared_block_cache_->selectAndEvict(min_blocks_to_free);
    if (evict_result.evicted_keys.empty()) {
        return nullptr;
    }
    if (metrics_reporter_) {
        for (const auto& [cache_key, lifetime_ms] : evict_result.evicted_lifetime_ms) {
            RtpLLMCacheEvictionMetricsCollector collector;
            collector.lifetime_ms = lifetime_ms;
            kmonitor::MetricsTags tags("scope", "gpu");
            tags.AddTag("evict_policy",
                        evict_result.evicted_independent_tag.count(cache_key) ? "independent" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.topologyPtr());
    for (const auto& group : config_.topology().groups()) {
        batch_resource->setLastBlockAligned(group.tag, true);
    }

    for (const auto& group : config_.topology().groups()) {
        CacheKeysType         group_keys;
        BlockDependenciesType group_dependencies;
        auto&                 group_blocks = batch_resource->mutableBlockIds(0, group.tag);
        for (const auto cache_key : evict_result.evicted_keys) {
            const auto& evicted_group_blocks = evict_result.evicted_blocks_by_group.at(cache_key);
            const auto  block_it =
                std::find_if(evicted_group_blocks.begin(), evicted_group_blocks.end(), [&](const auto& block) {
                    return block.tag == group.tag && !isNullBlockIdx(block.block_id);
                });
            if (block_it == evicted_group_blocks.end()) {
                continue;
            }
            BlockDependency dependency;
            dependency.ordinal = block_it->physical_ordinal;
            if (!group_keys.empty()) {
                dependency.has_parent = true;
                dependency.parent_key = group_keys.back();
            }
            group_keys.push_back(cache_key);
            group_dependencies.push_back(dependency);
            const auto block_pos = group_blocks.blocksNum();
            group_blocks.resize(block_pos + 1, NULL_BLOCK_IDX);
            group_blocks.setAt(block_pos, block_it->block_id);
        }
        batch_resource->cacheResource(0).setCacheKeys(group.tag, std::move(group_keys));
        batch_resource->cacheResource(0).setBlockDependencies(group.tag, std::move(group_dependencies));
        batch_resource->cacheResource(0).setCacheKeysAreCpCanonical(group.tag, true);
    }
    return batch_resource;
}

void KVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        for (const auto& entry : batch_kv_cache_resource->groupResources(batch_id)) {
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : entry.block_ids->blocks()) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                kv_cache_groups_.at(entry.tag)->blockPool()->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t KVCacheAllocator::requestRefBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->requestRefBlocksNum();
    }
    return total;
}

size_t KVCacheAllocator::connectorRefBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->connectorRefBlocksNum();
    }
    return total;
}

size_t KVCacheAllocator::blockCacheRefBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->blockCacheRefBlocksNum();
    }
    return total;
}

size_t KVCacheAllocator::notInUseBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->notInUseBlocksNum();
    }
    return total;
}

size_t KVCacheAllocator::minTokenCapacity(bool use_available_blocks) const {
    if (kv_cache_groups_.empty()) {
        return 0;
    }

    size_t min_tokens = std::numeric_limits<size_t>::max();
    bool   saw_group  = false;
    for (const auto& [tag, group] : kv_cache_groups_) {
        if (config_.group(tag).policy.fixed_block_num > 0) {
            continue;
        }
        const auto& pool = group->blockPool();
        if (!pool) {
            continue;
        }
        saw_group        = true;
        const auto block = use_available_blocks ? pool->availableBlocksNum() : pool->totalBlocksNum();
        min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(tag));
    }
    return saw_group ? min_tokens : 0;
}

size_t KVCacheAllocator::availableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/true);
}

size_t KVCacheAllocator::totalTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false);
}

size_t KVCacheAllocator::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t KVCacheAllocator::maxAvailableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false);
}

KVCacheTokenCapacity KVCacheAllocator::tokenCapacity() const {
    return {totalTokensNum(), availableTokensNum()};
}

std::vector<KVCachePoolMetricsSnapshot> KVCacheAllocator::poolMetricsSnapshots() const {
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(kv_cache_groups_.size());
    const size_t reserve_units = reserveBlocksNum();
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = pool->availableBlocksNum();
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = pool->requestRefBlocksNum();
        snapshot.connector_ref_blocks = pool->connectorRefBlocksNum();
        snapshot.reserve_blocks       = reserveBlocksForPool(tag, reserve_units);
        snapshot.used_ratio           = (snapshot.total_blocks == 0) ?
                                            0.0f :
                                            static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.available_blocks)
                                                     / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(snapshot);
    }
    return snapshots;
}

void KVCacheAllocator::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        pool->regUserMr(model_id, cache_store);
    }
}

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    int64_t total = 0;
    for (const auto& [tag, group] : kv_cache_groups_) {
        const auto& pool = group->blockPool();
        total += pool->getMrCostTimeMs();
    }
    return total;
}

size_t KVCacheAllocator::reservableLogicalUnits() const {
    size_t reservable_tokens = std::numeric_limits<size_t>::max();
    bool   saw_reservable    = false;
    for (const auto& [tag, group] : kv_cache_groups_) {
        if (!group || !group->blockPool() || !group->isReservable() || config_.usesFixedBlocks(tag)) {
            continue;
        }
        saw_reservable    = true;
        reservable_tokens = std::min(reservable_tokens,
                                     group->blockPool()->availableBlocksNum() * logicalSeqSizePerBlockForCapacity(tag));
    }
    return saw_reservable ? reservable_tokens / logicalCoverageUnitTokens() : 0;
}

size_t KVCacheAllocator::logicalCoverageUnitTokens() const {
    size_t unit = 1;
    for (const auto& [tag, group] : kv_cache_groups_) {
        if (!group || !group->isReservable() || config_.usesFixedBlocks(tag)) {
            continue;
        }
        const size_t span = logicalSeqSizePerBlockForCapacity(tag);
        const size_t gcd  = std::gcd(unit, span);
        RTP_LLM_CHECK_WITH_INFO(unit / gcd <= std::numeric_limits<size_t>::max() / span,
                                "reservable logical coverage unit overflow");
        unit = unit / gcd * span;
    }
    return unit;
}

size_t KVCacheAllocator::reservableAvailableBlocksNum() const {
    // The base allocator stores one scalar reserve count. For independent pools
    // that scalar is a count of joint logical coverage units, not a sum of
    // unrelated pool-local block IDs.
    return reservableLogicalUnits();
}

size_t KVCacheAllocator::reserveBlocksForPool(std::string_view tag, size_t reserve_units) const {
    const auto group = kv_cache_groups_.find(std::string(tag));
    if (group == kv_cache_groups_.end() || !group->second || !group->second->blockPool()
        || !group->second->isReservable() || config_.usesFixedBlocks(tag) || reserve_units == 0) {
        return 0;
    }
    const size_t unit = logicalCoverageUnitTokens();
    RTP_LLM_CHECK_WITH_INFO(unit <= std::numeric_limits<size_t>::max() / reserve_units,
                            "reserved logical token coverage overflow");
    const size_t reserve_tokens = reserve_units * unit;
    const size_t span           = logicalSeqSizePerBlockForCapacity(tag);
    return reserve_tokens / span + (reserve_tokens % span != 0 ? 1 : 0);
}

bool KVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return true;
    }
    const int  batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int  total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int  raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int  raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool reuse_enabled      = malloc_info.reuse_cache;

    for (const auto& [tag, group] : kv_cache_groups_) {
        if (!group->isReservable() || config_.usesFixedBlocks(tag)) {
            continue;
        }
        const auto cp_mapper              = cpSlotMapper(tag);
        const int  group_common_seq       = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_common_seq_len);
        const int  group_seq_len          = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_seq_len);
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, tag) : 0;
        const auto need =
            group->getNeedBlocks(group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool                 = kv_cache_groups_.at(tag)->blockPool();
        const size_t available_blocks     = pool->availableBlocksNum();
        const size_t total_blocks         = pool->totalBlocksNum();
        const size_t group_reserve_blocks = reserveBlocksForPool(tag, reserve_blocks);
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "tag=%s need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 tag.c_str(),
                                 need_blocks,
                                 total_blocks,
                                 available_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm
