#include "rtp_llm/cpp/cache/allocator/HybridKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

// CP shard helpers: when mapper is null/passthrough, all helpers no-op.
inline int cpEffectiveSeqLen(const std::shared_ptr<CPSlotMapper>& mapper, int seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(seq_len) : seq_len;
}

inline CacheKeysType cpEffectiveCacheKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full) {
    if (!mapper || !mapper->isSharded()) {
        return full;
    }
    CacheKeysType local;
    const int     cp_size = mapper->cpSize();
    const int     start   = cp_size - 1;
    for (int i = start; i < static_cast<int>(full.size()); i += cp_size) {
        local.push_back(full[i]);
    }
    return local;
}

inline int cpVirtualBlockSize(const std::shared_ptr<CPSlotMapper>& mapper, int block_size) {
    return (mapper && mapper->isSharded()) ? mapper->virtualBlockSize() : block_size;
}

inline bool containsGroupId(const std::vector<int>& group_ids, int gid) {
    return std::find(group_ids.begin(), group_ids.end(), gid) != group_ids.end();
}

inline bool cpShardThisGroup(const std::shared_ptr<CPSlotMapper>& mapper, const DeviceKVCacheGroupPtr& group) {
    return mapper && mapper->isSharded() && group && group->isCpShardable();
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const DeviceKVCacheGroupPtr&         group,
                                     int                                  seq_len) {
    return cpShardThisGroup(mapper, group) ? mapper->effectiveSeqLenForAlloc(seq_len) : seq_len;
}

inline int cpVirtualBlockSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                      const DeviceKVCacheGroupPtr&         group,
                                      int                                  block_size) {
    return cpShardThisGroup(mapper, group) ? mapper->virtualBlockSize() : block_size;
}

inline size_t groupSeqSize(const CacheConfig& config, int gid, size_t fallback) {
    return (gid >= 0 && static_cast<size_t>(gid) < config.group_seq_size_per_block.size()
            && config.group_seq_size_per_block[static_cast<size_t>(gid)] > 0) ?
               config.group_seq_size_per_block[static_cast<size_t>(gid)] :
               fallback;
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

bool HybridKVCacheAllocator::skipReuseCacheGroup(int gid) const {
    auto g = group(gid);
    return g && g->reusePolicy() == CacheReusePolicy::NON_REUSABLE;
}

std::vector<int> HybridKVCacheAllocator::independentEvictionGroupIds() const {
    std::vector<int> group_ids;
    if (!block_tree_cache_) {
        return group_ids;
    }
    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        auto g = group(gid);
        if (g && g->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            group_ids.push_back(gid);
        }
    }
    return group_ids;
}

bool HybridKVCacheAllocator::cpCompactSwaGroup(int gid, const std::shared_ptr<CPSlotMapper>& mapper) const {
    auto g = group(gid);
    if (!mapper || !mapper->isSharded() || !g || !g->cpCompactTailBlocks()) {
        return false;
    }
    const auto row_tokens = groupSeqSize(config_, gid, seqSizePerBlock());
    return row_tokens == static_cast<size_t>(mapper->virtualBlockSize());
}

HybridKVCacheAllocator::HybridKVCacheAllocator(const CacheConfig&                 config,
                                               AllocationType                     allocation_type,
                                               const kmonitor::MetricsReporterPtr metrics_reporter,
                                               int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

DeviceKVCacheGroupPtr HybridKVCacheAllocator::group(int gid) const {
    // Groups are owned by BlockTreeCache (created in the factory) and injected via
    // setBlockTreeCache(); returns null before injection or when gid is out of range.
    return block_tree_cache_ ? block_tree_cache_->deviceKVGroup(gid) : nullptr;
}

int HybridKVCacheAllocator::reuseCache(const CacheKeysType&                 cache_keys,
                                       BatchKVCacheResource&                kv_resource,
                                       const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                       std::shared_ptr<LoadBackTicket>&     ticket) {
    ticket.reset();
    if (!block_tree_cache_ || cache_keys.empty()) {
        return 0;
    }
    // Under cp shard, FULL groups index block_ids by cp-virtual-block units
    // (one entry covers cp_size physical blocks). LINEAR/SWA groups index by
    // raw block_size logical blocks. So when populating tail blocks for
    // LINEAR/SWA we need to scale the array length and matched-block position
    // back to the logical-block coordinate system.
    const int cp_scale = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;

    // Whole-sequence match: one joint-validated prefix match over all groups.
    // BlockTreeCache::match() references the matched device blocks internally; we take
    // the per-group indices we need and immediately release those match-protection
    // refs, leaving referenceBlocksInGroup() (in initMallocForCommonLen) as the sole
    // owner so the existing rollback/free machinery stays balanced.
    auto match_result          = block_tree_cache_->match(cache_keys);
    ticket                     = match_result.load_back_ticket;
    const int reuse_blocks_len = static_cast<int>(match_result.matched_blocks);
    if (reuse_blocks_len <= 0) {
        block_tree_cache_->releaseMatchedBlocks(match_result.matched_block_sets);
        return 0;
    }

    auto groupBlocks = [&](int gid) -> const BlockIndicesType& {
        static const BlockIndicesType kEmpty;
        auto                          it = match_result.group_block_indices.find(gid);
        return it == match_result.group_block_indices.end() ? kEmpty : it->second;
    };

    // FULL groups: assign the jointly-matched prefix (clamped to reuse_blocks_len).
    for (int gid : full_group_ids_) {
        BlockIndicesType full_blocks = groupBlocks(gid);
        if (static_cast<int>(full_blocks.size()) > reuse_blocks_len) {
            full_blocks.resize(static_cast<size_t>(reuse_blocks_len));
        }
        kv_resource.mutableBlockIds(0, gid).assign(std::move(full_blocks));
    }

    // LINEAR/SWA arrays are sized in logical-block units (cp_size× larger
    // than the FULL groups' cp-virtual-block units). The reusable tail block
    // is the LAST jointly-matched block, landing at logical index
    // `(reuse_blocks_len * cp_size) - 1`, NOT `reuse_blocks_len - 1`.
    const int logical_reuse_len = reuse_blocks_len * cp_scale;
    for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
        const int gid = linear_group_ids_[i];
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        const auto& blocks = groupBlocks(gid);
        if (!blocks.empty()) {
            kv_resource.mutableBlockIds(0, gid).setAt(static_cast<size_t>(logical_reuse_len - 1), blocks.back());
        }
    }
    for (size_t i = 0; i < swa_group_ids_.size(); ++i) {
        const int gid             = swa_group_ids_[i];
        const int group_reuse_len = cpCompactSwaGroup(gid, cp_mapper) ? reuse_blocks_len : logical_reuse_len;
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(group_reuse_len), NULL_BLOCK_IDX));
        if (skipReuseCacheGroup(gid)) {
            continue;
        }
        const auto& blocks = groupBlocks(gid);
        if (!blocks.empty()) {
            kv_resource.mutableBlockIds(0, gid).setAt(static_cast<size_t>(group_reuse_len - 1), blocks.back());
        }
    }

    block_tree_cache_->releaseMatchedBlocks(match_result.matched_block_sets);
    return reuse_blocks_len;
}

MallocResult HybridKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();
    RTP_LLM_CHECK_WITH_INFO(batch_size == 1, "currently batch size should be 1 in hybrid attention but %d", batch_size);

    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = cp_slot_mapper_;
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ != nullptr,
                            "HybridKVCacheAllocator: BlockTreeCache not injected before allocation");
    // reuse_unit_tokens is computed against the canonical (paged FULL) group's
    // block_size: cache_keys reuse only happens for paged groups so virtual block
    // size = canonical block_size * cp_size; non-paged groups don't enter reuse.
    const DeviceKVCacheGroupPtr reuse_group =
        full_group_ids_.empty() ? DeviceKVCacheGroupPtr{} : group(full_group_ids_.front());
    const int reuse_unit_tokens = cpVirtualBlockSizeForGroup(cp_mapper, reuse_group, seqSizePerBlock());

    const auto&                     cache_keys         = kv_resource->cacheKeys(0);
    int64_t                         match_cost_time_us = 0;
    const size_t                    reserve_blocks     = reserveBlockNum();
    int                             reuse_blocks       = 0;
    std::vector<BlockIndicesType>   referenced_blocks(static_cast<size_t>(kv_resource->groupNums()));
    std::shared_ptr<AsyncContext>   async_ctx;
    std::shared_ptr<LoadBackTicket> load_back_ticket;

    if (malloc_info.enable_device_cache) {
        // CP-sharded: subsample to last-rank canonical key namespace before matching.
        CacheKeysType cp_keys = cpEffectiveCacheKeys(cp_mapper, cache_keys);
        // Off mode drops the last key to skip the partial trailing block. Under
        // CP sharding cpEffectiveCacheKeys already excludes the partial block
        // (last-rank stride lands inside completed full blocks only), so the
        // extra drop would discard a valid full-block key — costing the SWA
        // tail-loop its only matchable key (full_keys[cp_size-1 + (n-1)*cp_size]
        // is exactly what the non-sharded SWA group caches).
        const bool    cp_active = cp_mapper && cp_mapper->isSharded();
        CacheKeysType match_keys(cp_keys.begin(),
                                 cp_active ? cp_keys.end() : (cp_keys.empty() ? cp_keys.end() : cp_keys.end() - 1));
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource, cp_mapper, load_back_ticket);
        match_cost_time_us     = currentTimeUs() - begin_us;

        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            const auto&      blocks = kv_resource->blocks(0, gid);
            BlockIndicesType valid;
            valid.reserve(blocks.size());
            for (auto b : blocks) {
                if (!isNullBlockIdx(b)) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                referenceBlocksInGroup(gid, valid);
                referenced_blocks[static_cast<size_t>(gid)] = std::move(valid);
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        rollbackInitMalloc(*kv_resource, referenced_blocks, {});
        return {false, 0};
    }

    std::vector<size_t> original_sizes(static_cast<size_t>(kv_resource->groupNums()));
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        original_sizes[static_cast<size_t>(gid)] = kv_resource->blocksNum(0, gid);
    }
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        auto&     block_ids_0   = kv_resource->mutableBlockIds(0, gid);
        const int group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, group(gid), common_seq_len);
        if (!group(gid)->malloc(block_ids_0, group_seq_len, malloc_info.reuse_cache, 0)) {
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            group(gid)->reference(kv_resource->mutableBlockIds(b, gid), kv_resource->blocks(0, gid));
        }
    }

    // All allocations and the reserve post-check passed: commit the deferred load_back
    // (allocate device targets and submit async copies). Any earlier return above leaves
    // the ticket uncommitted, so its destructor aborts the planned load_back.
    if (load_back_ticket != nullptr && !load_back_ticket->empty()) {
        async_ctx = load_back_ticket->commit();
        if (async_ctx == nullptr) {
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes);
            RTP_LLM_LOG_WARNING("Hybrid initMalloc failed because load_back target allocation was not atomic");
            return {false, 0};
        }
    }

    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us, async_ctx};
}

MallocResult HybridKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_resource = malloc_info.batch_kv_cache_resource;
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ != nullptr,
                            "HybridKVCacheAllocator: BlockTreeCache not injected before allocation");
    const auto& cp_mapper    = cp_slot_mapper_;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::vector<BlockIndicesType>> original_blocks(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        original_blocks[static_cast<size_t>(b)].resize(static_cast<size_t>(kv_resource->groupNums()));
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            original_blocks[static_cast<size_t>(b)][static_cast<size_t>(gid)] = kv_resource->blocks(b, gid);
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&     block_ids     = kv_resource->mutableBlockIds(b, gid);
            const int group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, group(gid), raw_seq_len);
            if (!group(gid)->malloc(block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step)) {
                all_success  = false;
                failed_batch = b;
                failed_group = gid;
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
            for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
                group(gid)->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, gid), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&       block_ids = kv_resource->mutableBlockIds(b, gid);
            const auto& original  = original_blocks[static_cast<size_t>(b)][static_cast<size_t>(gid)];

            std::unordered_set<BlockIdxType> original_valid_blocks;
            original_valid_blocks.reserve(original.size());
            for (auto block : original) {
                if (!isNullBlockIdx(block)) {
                    original_valid_blocks.insert(block);
                }
            }

            BlockIndicesType blocks_to_free;
            for (auto block : block_ids.blocks()) {
                if (!isNullBlockIdx(block) && original_valid_blocks.find(block) == original_valid_blocks.end()) {
                    blocks_to_free.push_back(block);
                }
            }
            if (!blocks_to_free.empty()) {
                freeBlocksInGroup(gid, blocks_to_free);
            }
            block_ids.assign(original);
        }
    }
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d group=%d", failed_batch, failed_group);
    return {false, 0};
}

void HybridKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            group(gid)->free(kv_cache_resource->blocks(batch_id, gid));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!block_tree_cache_) {
        return;
    }

    const auto& cp_mapper  = cp_slot_mapper_;
    const bool  cp_active  = cp_mapper && cp_mapper->isSharded();
    const int   group_nums = kv_cache_resource->groupNums();
    const int   batch_size = kv_cache_resource->batchSize();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        kv_cache_resource->cacheResource(batch_id).ensureLinearBlockDependencies();
        const auto& full_keys = kv_cache_resource->cacheKeys(batch_id);
        if (full_keys.empty()) {
            continue;
        }

        // Whole-sequence insert on the single canonical chain (design §7.6): under
        // CP all groups share the last-rank canonical key sequence; non-CP aggregates
        // every group under the full key sequence. Parent-child dependencies are
        // implicit in the ordered keys, so no explicit BlockDependency is needed.
        const CacheKeysType insert_keys = cp_active ? cpEffectiveCacheKeys(cp_mapper, full_keys) : full_keys;
        if (insert_keys.empty()) {
            continue;
        }

        std::vector<std::vector<GroupSlot>> slots(insert_keys.size(),
                                                  std::vector<GroupSlot>(static_cast<size_t>(group_nums)));
        bool                                any_block = false;
        for (int gid = 0; gid < group_nums; ++gid) {
            if (skipReuseCacheGroup(gid)) {
                continue;
            }
            const auto&  blocks   = kv_cache_resource->blocks(batch_id, gid);
            const size_t loop_end = std::min(insert_keys.size(), blocks.size());
            for (size_t i = 0; i < loop_end; ++i) {
                if (isNullBlockIdx(blocks[i])) {
                    continue;
                }
                slots[i][static_cast<size_t>(gid)].device_blocks = {blocks[i]};
                any_block                                        = true;
            }
        }
        if (any_block) {
            block_tree_cache_->insert(/*parent=*/nullptr, insert_keys, slots);
        }
    }
}

std::shared_ptr<KVCacheResource> HybridKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                        const CacheKeysType&   cache_keys,
                                                                        bool                   is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(kvcache_resource.groupNums(),
                                  static_cast<int>(config_.layer_all_num),
                                  config_.layerGroupIdsSnapshot(),
                                  config_.kernelBlocksPerKvBlock(),
                                  config_.groupTypesSnapshot());

    CacheKeysType                 selected_keys;
    BlockDependenciesType         selected_dependencies;
    std::vector<BlockIndicesType> selected_blocks(static_cast<size_t>(kvcache_resource.groupNums()));
    const auto&                   source_dependencies = kvcache_resource.blockDependencies();

    selected_dependencies.reserve(cache_keys.size());
    selected_keys.reserve(cache_keys.size());
    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t              pos             = it->second;
        bool                      any_valid_block = false;
        std::vector<BlockIdxType> blocks_for_key(static_cast<size_t>(kvcache_resource.groupNums()), NULL_BLOCK_IDX);
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            const auto& src_blocks                   = kvcache_resource.blocks(gid);
            const auto  block                        = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key[static_cast<size_t>(gid)] = block;
            any_valid_block                          = any_valid_block || (!isNullBlockIdx(block) && block > 0);
        }
        const bool preserve_connector_tail = is_connector && !kvcache_resource.lastBlockAligned()
                                             && pos + 1 == resource_keys.size() && !selected_keys.empty();
        if (!any_valid_block && !preserve_connector_tail) {
            continue;
        }
        selected_keys.push_back(key);
        selected_dependencies.push_back(
            pos < source_dependencies.size() ?
                source_dependencies[pos] :
                BlockDependency{false, 0, static_cast<uint32_t>(selected_dependencies.size())});
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            selected_blocks[static_cast<size_t>(gid)].push_back(blocks_for_key[static_cast<size_t>(gid)]);
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->cacheKeys() = std::move(selected_keys);
    selected_resource->setBlockDependencies(std::move(selected_dependencies));
    for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
        BlockIndicesType valid;
        for (auto b : selected_blocks[static_cast<size_t>(gid)]) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocksInGroup(gid, valid, is_connector);
        }
        selected_resource->mutableBlockIds(gid).assign(std::move(selected_blocks[static_cast<size_t>(gid)]));
    }
    return selected_resource;
}

void HybridKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
        BlockIndicesType valid;
        for (auto b : kvcache_resource.blocks(gid)) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            freeBlocksInGroup(gid, valid, is_connector);
        }
    }
}

bool HybridKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           const std::vector<int>&        block_src_batch,
                                           bool                           copy_last_block,
                                           std::vector<BlockIdPair>&      block_update_mapping) {
    (void)batch_kv_cache_resource;
    (void)block_src_batch;
    (void)copy_last_block;
    (void)block_update_mapping;
    RTP_LLM_FAIL("HybridKVCacheAllocator::updateKVBlock is not supported");
}

int HybridKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

bool HybridKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const {
    const int need_blocks = getNeedBlocks(malloc_info);
    if (need_blocks <= 0) {
        return true;
    }
    const size_t required_blocks = static_cast<size_t>(need_blocks) + reserve_blocks;
    const size_t free_blocks = freeBlocksNum();
    const bool accepted = free_blocks >= required_blocks;
    if (!accepted && malloc_info.verbose) {
        RTP_LLM_LOG_INFO("Hybrid initMalloc rejected by reserve blocks: request_id=%ld "
                         "need_blocks=%d free_blocks=%zu reserve_blocks=%zu",
                         malloc_info.request_id,
                         need_blocks,
                         free_blocks,
                         reserve_blocks);
    }
    return accepted;
}

void HybridKVCacheAllocator::rollbackBlockIdsToSize(int gid, BlockIds& block_ids, size_t original_size) {
    if (block_ids.blocksNum() <= original_size) {
        return;
    }
    const auto blocks_to_free = validBlocksAfter(block_ids.blocks(), original_size);
    block_ids.resize(original_size);
    if (!blocks_to_free.empty()) {
        freeBlocksInGroup(gid, blocks_to_free);
    }
}

void HybridKVCacheAllocator::rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                                const std::vector<BlockIndicesType>& referenced_blocks,
                                                const std::vector<size_t>&           original_sizes) {
    for (int gid = 0; gid < kv_resource.groupNums(); ++gid) {
        auto& block_ids = kv_resource.mutableBlockIds(0, gid);
        if (!original_sizes.empty() && static_cast<size_t>(gid) < original_sizes.size()
            && block_ids.blocksNum() > original_sizes[static_cast<size_t>(gid)]) {
            rollbackBlockIdsToSize(gid, block_ids, original_sizes[static_cast<size_t>(gid)]);
        }
        if (static_cast<size_t>(gid) < referenced_blocks.size()
            && !referenced_blocks[static_cast<size_t>(gid)].empty()) {
            freeBlocksInGroup(gid, referenced_blocks[static_cast<size_t>(gid)]);
        }
        block_ids.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseBlockNum(0);
}

void HybridKVCacheAllocator::rollbackIncrMalloc(BatchKVCacheResource&                   kv_resource,
                                                const std::vector<std::vector<size_t>>& original_sizes,
                                                int                                     failed_batch) {
    const int last_touched_batch = std::min(failed_batch, kv_resource.batchSize() - 1);
    for (int b = 0; b <= last_touched_batch; ++b) {
        for (int gid = 0; gid < kv_resource.groupNums(); ++gid) {
            auto&        block_ids    = kv_resource.mutableBlockIds(b, gid);
            const size_t original_num = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)];
            rollbackBlockIdsToSize(gid, block_ids, original_num);
        }
    }
}

int HybridKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;
    const int   reuse_blocks_len   = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        const auto grp              = group(gid);
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, grp, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, grp, raw_seq_len);
        const auto need =
            grp->getNeedBlocks(group_common_seq, group_seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

void HybridKVCacheAllocator::checkCPShardedMallocResult(const MallocInfo& malloc_info) const {
    if (!cp_slot_mapper_ || !cp_slot_mapper_->isSharded()) {
        return;
    }

    const auto& kv_resource  = malloc_info.batch_kv_cache_resource;
    const int   seq_len      = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    for (int batch_id = 0; batch_id < kv_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            const auto grp = group(gid);
            if (!cpShardThisGroup(cp_slot_mapper_, grp)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, grp, seq_len);
            const int expected_blocks   = grp->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks     = kv_resource->blocksNum(batch_id, gid);
            RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                    "CP invariant violated: batch=%d group=%d blocks=%d != expected_local_blocks=%d "
                                    "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                    "block_size=%d, cacheKeys=%zu)",
                                    batch_id,
                                    gid,
                                    actual_blocks,
                                    expected_blocks,
                                    seq_len,
                                    effective_seq_len,
                                    reserve_step,
                                    cp_slot_mapper_->cpSize(),
                                    cp_slot_mapper_->blockSize(),
                                    kv_resource->cacheKeys(batch_id).size());
        }
    }
}

int HybridKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                  int                            seq_len,
                                                  int                            reserve_step) const {
    int need_blocks = 0;
    for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
        const auto grp               = group(gid);
        const int  effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, grp, seq_len);
        const int  cur_blocks        = batch_kv_cache_resource->blocksNum(0, gid);
        need_blocks += grp->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
