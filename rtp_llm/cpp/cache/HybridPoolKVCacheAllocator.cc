#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// ---------- M01-PR2: SuperBlockFreeList ----------
SuperBlockFreeList::SuperBlockFreeList(uint32_t num_super_blocks): num_super_blocks_(num_super_blocks) {
    // §1.1 invariant 1: super_block_id 0 is RESERVED. Free list starts at 1.
    RTP_LLM_CHECK_WITH_INFO(num_super_blocks_ > 0,
                            "SuperBlockFreeList requires num_super_blocks > 0 (got %u)",
                            num_super_blocks_);
    for (uint32_t s = 1; s < num_super_blocks_; ++s) {
        free_list_.push_back(static_cast<int>(s));
    }
}

int SuperBlockFreeList::allocSuperBlock() {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_list_.empty()) {
        return -1;
    }
    const int s = free_list_.front();
    free_list_.pop_front();
    // §1.1 invariant 1 (defence in depth): never hand out 0.
    RTP_LLM_CHECK_WITH_INFO(s > 0, "SuperBlockFreeList produced reserved id 0");
    return s;
}

void SuperBlockFreeList::freeSuperBlock(int S) {
    RTP_LLM_CHECK_WITH_INFO(S > 0 && static_cast<uint32_t>(S) < num_super_blocks_,
                            "SuperBlockFreeList::freeSuperBlock invalid id %d (budget=%u)",
                            S,
                            num_super_blocks_);
    std::lock_guard<std::mutex> lk(mu_);
    free_list_.push_back(S);
}

size_t SuperBlockFreeList::totalCount() const {
    return static_cast<size_t>(num_super_blocks_);
}

size_t SuperBlockFreeList::freeCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    return free_list_.size();
}


HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio,
                                                       RoleType                           role_type):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

HybridPoolKVCacheAllocator::~HybridPoolKVCacheAllocator() {
    // M03-PR3: detach the shared cache's wiring BEFORE our owned counter and
    // super-block free list are destroyed. The base ~KVCacheAllocator runs
    // AFTER this dtor and tears down shared_block_cache_; any callback
    // captured by that cache would otherwise see a dangling raw pointer.
    if (shared_block_cache_) {
        shared_block_cache_->setUnifiedRefCounter(nullptr);
        shared_block_cache_->setSuperBlockReclaimCallback({});
    }
}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "no cache_specs found in CacheConfig");
    RTP_LLM_CHECK_WITH_INFO(config_.cache_specs.size() == config_.global_layer_ids.size(),
                            "cache_specs size %zu != global_layer_ids size %zu",
                            config_.cache_specs.size(),
                            config_.global_layer_ids.size());

    const int group_nums = static_cast<int>(config_.cache_specs.size());
    group_block_pools_.reserve(static_cast<size_t>(group_nums));
    kv_cache_groups_.reserve(static_cast<size_t>(group_nums));

    SharedBlockCache* shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;

    for (int gid = 0; gid < group_nums; ++gid) {
        RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(config_.group_types.size()),
                                "missing group type for group %d in HybridPoolKVCacheAllocator",
                                gid);
        const auto group_type = config_.group_types[static_cast<size_t>(gid)];

        auto pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, static_cast<size_t>(gid));
        // State pools live on pinned CPU iff the single source of truth flag
        // is set. CacheConfigCreator sets this when either:
        //   - state_pool_memory_mb > 0 (explicit env-driven budget), or
        //   - role_type == PREFILL (DSV4 PD-sep prefill rank stages state to
        //     pinned host memory before handing the request to decode).
        // Both gated by state_block_size_bytes > 0 (skip non-DSV4 paths).
        // The HBM divisor + fixed_pool_reserve_bytes already exclude STATE
        // bytes when this flag is true, so allocator and budget agree.
        const bool state_on_pinned_cpu =
            allocation_type_ == AllocationType::DEVICE && config_.state_pool_uses_pinned_cpu
            && static_cast<size_t>(gid) < config_.group_region_names.size()
            && isStateRegion(config_.group_region_names[static_cast<size_t>(gid)]);
        auto group_pool = std::make_shared<BlockPool>(pool_config, allocation_type_, state_on_pinned_cpu);
        RTP_LLM_CHECK_WITH_INFO(group_pool->init(), "Failed to initialize block pool for group %d", gid);

        const auto& ids  = config_.global_layer_ids[static_cast<size_t>(gid)];
        auto        spec = config_.cache_specs[static_cast<size_t>(gid)];

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group =
                std::make_shared<LinearKVCacheGroup>(ids, spec, group_pool, gid, config_.linear_step, shared_cache_raw);
            linear_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::SWA) {
            group =
                std::make_shared<SWAKVCacheGroup>(ids, spec, group_pool, gid, config_.linear_step, shared_cache_raw);
            swa_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, group_pool, gid, shared_cache_raw);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    // HybridPool owns one BlockPool per group; do not read pool stats from block_pool_ in HybridPool mode.
    block_pool_ = group_block_pools_.empty() ? nullptr : group_block_pools_[0];

    if (shared_block_cache_) {
        shared_block_cache_->init(group_nums, group_block_pools_);
    }

    // M01-PR2: instantiate the super-block free list only when the unified
    // layout is enabled. When disabled (default), super_block_allocator_ stays
    // nullptr and the legacy per-pool path is bit-equal to today's behaviour.
    if (config_.super_block_layout.isUnified()) {
        const uint32_t budget = config_.super_block_layout.num_super_blocks;
        RTP_LLM_CHECK_WITH_INFO(budget > 0,
                                "super_block_layout.enabled=true but num_super_blocks=0 — config bug");
        RTP_LLM_CHECK_WITH_INFO(config_.super_block_layout.bps.size() == config_.cache_specs.size(),
                                "super_block_layout.bps size %zu != cache_specs size %zu",
                                config_.super_block_layout.bps.size(),
                                config_.cache_specs.size());
        for (size_t p = 0; p < config_.super_block_layout.bps.size(); ++p) {
            RTP_LLM_CHECK_WITH_INFO(config_.super_block_layout.bps[p] >= 1,
                                    "super_block_layout.bps[%zu]=%u (must be >= 1)",
                                    p,
                                    config_.super_block_layout.bps[p]);
        }
        super_block_allocator_ = std::make_unique<SuperBlockFreeList>(budget);
        // M03-PR3: construct the unified ref counter (5-counter family per
        // super-block) and wire it into the shared cache so put / match /
        // evict paths honour the dual-write contract. Lifetime is tied to
        // *this; the raw pointer handed to SharedBlockCache stays valid as
        // long as the allocator outlives the cache (verified at teardown by
        // KVCacheAllocator destruction order — cache resets first).
        unified_ref_counter_ = std::make_unique<UnifiedRefCounter>();
        unified_ref_counter_->init(static_cast<int>(budget));
        if (shared_block_cache_) {
            shared_block_cache_->setUnifiedRefCounter(unified_ref_counter_.get());
            // Reclaim callback: SharedBlockCache invokes this OUTSIDE its mu_
            // when a cache eviction drops the last UnifiedRefCounter primary
            // for S. Bypass the public freeSuperBlock to avoid a RTP_LLM_CHECK
            // re-validation hop — we already know the super_block_allocator_
            // is non-null inside this branch.
            SuperBlockFreeList* sbfl = super_block_allocator_.get();
            shared_block_cache_->setSuperBlockReclaimCallback([sbfl](int S) {
                if (sbfl) {
                    sbfl->freeSuperBlock(S);
                }
            });
        }
        RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator unified path ENABLED: num_super_blocks=%u, "
                         "free=%zu (id 0 reserved), UnifiedRefCounter wired",
                         budget,
                         super_block_allocator_->freeCount());
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu, unified=%s",
                     group_block_pools_.size(),
                     config_.super_block_layout.isUnified() ? "true" : "false");
    return true;
}

// ---------- M01-PR2: unified-path public primitives ----------

int HybridPoolKVCacheAllocator::allocSuperBlock() {
    RTP_LLM_CHECK_WITH_INFO(super_block_allocator_ != nullptr,
                            "allocSuperBlock called but super_block_allocator_ is null "
                            "(unified path disabled)");
    return super_block_allocator_->allocSuperBlock();
}

void HybridPoolKVCacheAllocator::freeSuperBlock(int S) {
    RTP_LLM_CHECK_WITH_INFO(super_block_allocator_ != nullptr,
                            "freeSuperBlock called but super_block_allocator_ is null "
                            "(unified path disabled)");
    super_block_allocator_->freeSuperBlock(S);
}

size_t HybridPoolKVCacheAllocator::freeSuperBlocksNum() const {
    return super_block_allocator_ ? super_block_allocator_->freeCount() : 0;
}

int HybridPoolKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_to_group_id.size()) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const int gid = config_.layer_to_group_id[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()),
                            "invalid default group id %d for layer %d",
                            gid,
                            layer_id);
    return gid;
}

int HybridPoolKVCacheAllocator::groupIdForLayerRegion(int layer_id, KVCacheRegionName region_name) const {
    const size_t attn_id = static_cast<size_t>(region_name);
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_region_to_group_id.size()) {
        const auto& dense = config_.layer_region_to_group_id[static_cast<size_t>(layer_id)];
        if (attn_id < dense.size() && dense[attn_id] >= 0) {
            const int gid = dense[attn_id];
            RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(kv_cache_groups_.size()),
                                    "invalid group id %d for layer %d region %zu",
                                    gid,
                                    layer_id,
                                    attn_id);
            return gid;
        }
    }
    if (region_name == KVCacheRegionName::DEFAULT) {
        return defaultGroupIdForLayer(layer_id);
    }
    RTP_LLM_FAIL("missing group mapping for layer_id=%d region=%zu", layer_id, attn_id);
}

void HybridPoolKVCacheAllocator::referenceBlocksInGroup(int                     gid,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorReference(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestReference(blocks);
    }
}

void HybridPoolKVCacheAllocator::freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorFree(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestFree(blocks);
    }
}

CacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    layout.layer_to_groups          = config_.layer_to_group_id;
    layout.layer_to_group_ids       = config_.layer_to_group_ids;
    layout.layer_region_to_group_id = config_.layer_region_to_group_id;
    layout.group_types              = config_.group_types;
    layout.group_region_names       = config_.group_region_names;
    layout.layer_group_types        = config_.layer_group_types;

    const bool has_typed_mapping = !config_.layer_region_to_group_id.empty();
    if (has_typed_mapping) {
        RTP_LLM_CHECK_WITH_INFO(config_.group_region_names.size() == kv_cache_groups_.size(),
                                "group_region_names size %zu != group num %zu for typed layer-region mapping",
                                config_.group_region_names.size(),
                                kv_cache_groups_.size());
    }

    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);
    const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    layout.layers_to_kv_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        layout.layers_to_kv_buffer_ptrs_by_attn[layer_id].resize(region_name_count);
        layout.layers_to_scale_buffer_ptrs_by_attn[layer_id].resize(region_name_count);
    }

    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        const int  gid           = defaultGroupIdForLayer(static_cast<int>(layer_id));
        const auto layer_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerCacheBase();
        const auto scale_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerScaleCacheBase();
        auto       it            = layer_tensors.find(static_cast<int>(layer_id));
        if (it != layer_tensors.end()) {
            layout.layers_to_kv_buffer_ptrs[layer_id] = it->second;
        }
        auto scale_it = scale_tensors.find(static_cast<int>(layer_id));
        if (scale_it != scale_tensors.end()) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = scale_it->second;
        }
    }

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto layer_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerCacheBase();
        const auto scale_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerScaleCacheBase();
        const auto region_name   = static_cast<size_t>(gid < static_cast<int>(config_.group_region_names.size()) ?
                                                         config_.group_region_names[static_cast<size_t>(gid)] :
                                                         KVCacheRegionName::DEFAULT);
        RTP_LLM_CHECK_WITH_INFO(
            region_name < region_name_count, "group %d has invalid region id %zu", gid, region_name);
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_kv_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed kv layout range %zu",
                layer_id,
                layout.layers_to_kv_buffer_ptrs_by_attn.size());
            layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][region_name] = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_scale_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed scale layout range %zu",
                layer_id,
                layout.layers_to_scale_buffer_ptrs_by_attn.size());
            layout.layers_to_scale_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][region_name] = tensor;
        }
    }
    return layout;
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    // M01-PR2 / §6.1 Fix 11: under multi-spec layouts (DSV4 = 7 regions), the
    // legacy 2-arg accessor cannot disambiguate the region and would silently
    // mis-map. Force callers to use the region-aware overload when unified is
    // enabled. Legacy (env=0) path keeps its existing semantics.
    RTP_LLM_CHECK_WITH_INFO(!config_.super_block_layout.isUnified(),
                            "Legacy 2-arg convertIndexToAddr(layer_id, block_id) is forbidden when "
                            "super_block_layout.enabled=true. Call convertIndexToAddr(layer_id, "
                            "region_name, block_id) instead.");
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(!config_.super_block_layout.isUnified(),
                            "Legacy 2-arg convertIndexToBuffer(layer_id, block_id) is forbidden when "
                            "super_block_layout.enabled=true. Call convertIndexToBuffer(layer_id, "
                            "region_name, block_id) instead.");
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(!config_.super_block_layout.isUnified(),
                            "Legacy 2-arg convertIndexToBuffer(layer_id, block_id, partition_count, "
                            "partition_id) is forbidden when super_block_layout.enabled=true. Call "
                            "convertIndexToBuffer(layer_id, region_name, block_id, ...) instead.");
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, KVCacheRegionName region_name, int block_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, KVCacheRegionName region_name, int block_id, int partition_count, int partition_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < group_block_pools_.size(), "missing block pool for group %d", gid);
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < config_.cache_specs.size(), "missing cache spec for group %d", gid);
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < config_.global_layer_ids.size(), "missing layer ids for group %d", gid);
        const auto   copy_type = BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                                              group_block_pools_[static_cast<size_t>(gid)]->where());
        const auto&  spec      = config_.cache_specs[static_cast<size_t>(gid)];
        const size_t buffers_per_layer = spec->scale_block_size_bytes() > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.global_layer_ids[static_cast<size_t>(gid)].size()
                                * static_cast<size_t>(end_ptr - begin_ptr) * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(gid) < config_.cache_specs.size(), "missing cache spec for group %d", gid);
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(gid) < config_.global_layer_ids.size(), "missing layer ids for group %d", gid);

            const size_t kv_block_size_bytes = config_.cache_specs[static_cast<size_t>(gid)]->block_size_bytes();
            const size_t scale_block_bytes   = config_.cache_specs[static_cast<size_t>(gid)]->scale_block_size_bytes();
            const auto   copy_type =
                BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                               group_block_pools_[static_cast<size_t>(gid)]->where());

            for (int layer_id : config_.global_layer_ids[static_cast<size_t>(gid)]) {
                auto src_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, src_block_index);
                auto dst_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, dest_block_index);

                if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                    RTP_LLM_LOG_ERROR("Failed to get block address for group %d layer %d, src_block %d, dst_block %d",
                                      gid,
                                      layer_id,
                                      src_block_index,
                                      dest_block_index);
                    continue;
                }

                copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

                if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                    copy_params.add(
                        dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
                }
            }
        }
    }

    execBatchCopy(copy_params);
}

size_t HybridPoolKVCacheAllocator::freeBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->freeBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::availableBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->availableBlocksNum();
    }
    return total;
}

BatchKVCacheResourcePtr HybridPoolKVCacheAllocator::popBlocksFromCache(size_t min_blocks_to_free) {
    if (min_blocks_to_free == 0 || !shared_block_cache_) {
        return nullptr;
    }

    auto evict_result = shared_block_cache_->selectAndEvict(min_blocks_to_free);
    if (evict_result.evicted_keys.empty()) {
        return nullptr;
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.groupNums(),
                               static_cast<int>(config_.layer_all_num),
                               config_.layer_to_group_id,
                               config_.kernelBlocksPerKvBlock(),
                               config_.group_types,
                               config_.layer_region_to_group_id);
    batch_resource->setLastBlockAligned(true);

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        batch_resource->mutableBlockIds(0, gid).resize(evict_result.evicted_keys.size(), NULL_BLOCK_IDX);
    }

    for (size_t evicted_idx = 0; evicted_idx < evict_result.evicted_keys.size(); ++evicted_idx) {
        const auto  cache_key = evict_result.evicted_keys[evicted_idx];
        const auto& slots     = evict_result.evicted_slots.at(cache_key);
        batch_resource->pushBackCacheKey(0, cache_key);
        for (int gid = 0; gid < static_cast<int>(slots.size()) && gid < config_.groupNums(); ++gid) {
            if (!isNullBlockIdx(slots[gid])) {
                batch_resource->mutableBlockIds(0, gid).setAt(evicted_idx, slots[gid]);
            }
        }
    }
    return batch_resource;
}

void HybridPoolKVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : batch_kv_cache_resource->blocks(batch_id, gid)) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                group_block_pools_[static_cast<size_t>(gid)]->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t HybridPoolKVCacheAllocator::requestRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->requestRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::connectorRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->connectorRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::blockCacheRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->blockCacheRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::notInUseBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->notInUseBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::availableTokensNum() const {
    if (group_block_pools_.empty()) {
        return 0;
    }
    size_t min_tokens = std::numeric_limits<size_t>::max();
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const size_t seq_size =
            (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                config_.group_seq_size_per_block[gid] :
                config_.seq_size_per_block;
        min_tokens = std::min(min_tokens, group_block_pools_[gid]->availableBlocksNum() * seq_size);
    }
    return min_tokens;
}

size_t HybridPoolKVCacheAllocator::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::maxAvailableTokensNum() const {
    if (group_block_pools_.empty()) {
        return 0;
    }
    size_t min_tokens     = std::numeric_limits<size_t>::max();
    bool   saw_full_group = false;
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        if (gid < config_.group_types.size() && config_.group_types[gid] != CacheGroupType::FULL) {
            continue;
        }
        saw_full_group = true;
        const size_t seq_size =
            (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                config_.group_seq_size_per_block[gid] :
                config_.seq_size_per_block;
        min_tokens = std::min(min_tokens, group_block_pools_[gid]->totalBlocksNum() * seq_size);
    }
    if (!saw_full_group) {
        for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
            const size_t seq_size =
                (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                    config_.group_seq_size_per_block[gid] :
                    config_.seq_size_per_block;
            min_tokens = std::min(min_tokens, group_block_pools_[gid]->totalBlocksNum() * seq_size);
        }
    }
    return min_tokens;
}

void HybridPoolKVCacheAllocator::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& pool : group_block_pools_) {
        pool->regUserMr(model_id, cache_store);
    }
}

int64_t HybridPoolKVCacheAllocator::getMrCostTimeMs() const {
    int64_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->getMrCostTimeMs();
    }
    return total;
}

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return true;
    }
    const int  batch_size     = malloc_info.batch_kv_cache_resource->batchSize();
    const int  total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int  common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int  seq_len        = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step   = malloc_info.complete_token_ids->getReserveStep();
    const bool reuse_enabled  = malloc_info.reuse_cache;

    size_t total_available_blocks = 0;
    for (const auto& pool : group_block_pools_) {
        total_available_blocks += pool->availableBlocksNum();
    }

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, gid) : 0;
        const auto need                   = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            common_seq_len, seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const size_t available_blocks = group_block_pools_[static_cast<size_t>(gid)]->availableBlocksNum();
        const size_t group_reserve_blocks =
            total_available_blocks > 0 ? reserve_blocks * available_blocks / total_available_blocks : 0;
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld group=%d "
                                 "need_blocks=%d available_blocks=%zu reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 gid,
                                 need_blocks,
                                 available_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            return false;
        }
    }
    return true;
}

// ---------- M01-PR2: unified malloc/free ----------
//
// PR-2 scope:
//   * super_block_id N maps to per-pool block ids via poolBlockId(p, N, k) =
//     N * bps[p] + k. For DSV4 (bps[p]==1 for all p) this is identity.
//   * Per-pool BlockPool::free_block_ids_ and ref-counter remain dormant under
//     unified mode (M03-PR3 migrates them away). PR-2 still bumps the per-pool
//     request_ref_counter via requestReference so today's accounting queries
//     (freeBlocksNum / blockCacheRefBlocksNum / etc.) stay coherent.
//   * KVCacheResource::group_block_ids continues to carry the per-pool block
//     ids (M01-PR3 collapses this to a single super_block_ids vector).
//
// Lock order (M01 §3.7 canonical 3-tier):
//   [L1] SharedBlockCache::mu_     — NOT taken in PR-2 (no prefix reuse path)
//   [L2] super_block_allocator_->mu_ — held only inside allocSuperBlock /
//                                     freeSuperBlock primitives (their internal
//                                     std::mutex). PR-2 acquires and releases
//                                     it pointwise per super-block call.
//   [L3] BlockPool::{ref_mu_, free_mu_} — taken inside requestReference via
//                                         the std::scoped_lock pair.
// PR-2 never holds L2 across L3 boundaries because allocSuperBlock /
// freeSuperBlock return before any BlockPool call.

MallocResult HybridPoolKVCacheAllocator::unifiedMalloc(const MallocInfo& malloc_info) {
    RTP_LLM_CHECK_WITH_INFO(super_block_allocator_ != nullptr,
                            "unifiedMalloc called without super_block_allocator_ (config bug)");
    RTP_LLM_CHECK_WITH_INFO(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids,
                            "unifiedMalloc: null batch_kv_cache_resource or complete_token_ids");

    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();
    RTP_LLM_CHECK_WITH_INFO(
        batch_size == 1, "unifiedMalloc currently requires batch_size==1 (got %d)", batch_size);

    const int group_nums = kv_resource->groupNums();
    RTP_LLM_CHECK_WITH_INFO(group_nums > 0, "unifiedMalloc: groupNums()==0");
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(group_nums) == config_.super_block_layout.bps.size(),
                            "unifiedMalloc: groupNums=%d != bps.size=%zu",
                            group_nums,
                            config_.super_block_layout.bps.size());

    // Determine how many super-blocks this call needs. The unified path treats
    // bps[p]==1 as the structural invariant (M01 §6 risk 1); under that
    // invariant every pool needs the same per-super-block count, so the
    // canonical group 0's getNeedBlocks gives the super-block count directly.
    const int need_blocks = getNeedBlocks(malloc_info);
    if (need_blocks <= 0) {
        // Nothing to do — preserves today's "no-op malloc" semantics for paths
        // that arrive with all blocks already cached.
        return {true, 0};
    }

    // §1.1 invariant 5: bps[p]>=1 per pool. PR-2 only supports bps[p]==1 to
    // keep poolBlockId(p, S, 0) identical across pools (fail loudly if a future
    // helper tries to flip bps>1 without an updated allocator).
    for (size_t p = 0; p < config_.super_block_layout.bps.size(); ++p) {
        RTP_LLM_CHECK_WITH_INFO(config_.super_block_layout.bps[p] == 1,
                                "unifiedMalloc (PR-2): bps[%zu]=%u, only bps==1 is supported in PR-2",
                                p,
                                config_.super_block_layout.bps[p]);
    }

    // Allocate super-block ids one-by-one; rollback on first failure.
    std::vector<int> allocated;
    allocated.reserve(static_cast<size_t>(need_blocks));
    for (int i = 0; i < need_blocks; ++i) {
        const int s = super_block_allocator_->allocSuperBlock();
        if (s < 0) {
            for (int prev : allocated) {
                super_block_allocator_->freeSuperBlock(prev);
            }
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("unifiedMalloc rejected: super-block free list exhausted "
                                 "(request_id=%ld, need=%d, granted=%zu, free=%zu, total=%zu)",
                                 malloc_info.request_id,
                                 need_blocks,
                                 allocated.size(),
                                 super_block_allocator_->freeCount(),
                                 super_block_allocator_->totalCount());
            }
            return {false, 0};
        }
        // §1.1 invariant 1: never assign super_block_id==0 to a write path.
        RTP_LLM_CHECK_WITH_INFO(s > 0, "unifiedMalloc received reserved super_block_id 0");
        allocated.push_back(s);
    }

    // Materialise per-pool block ids and append to resource. Under bps[p]==1
    // poolBlockId(p, S, 0) == S, so each pool's view is identical.
    for (int p = 0; p < group_nums; ++p) {
        auto&            block_ids = kv_resource->mutableBlockIds(0, p);
        BlockIndicesType new_ids;
        new_ids.reserve(allocated.size());
        for (int s : allocated) {
            const int k_count = static_cast<int>(config_.super_block_layout.bps[static_cast<size_t>(p)]);
            for (int k = 0; k < k_count; ++k) {
                new_ids.push_back(config_.poolBlockId(p, s, static_cast<uint32_t>(k)));
            }
        }
        block_ids.add(new_ids);
        // Bump per-pool request_ref_counter to keep today's accounting queries
        // coherent. requestReference itself takes BlockPool::ref_mu_ via the
        // std::scoped_lock pair (L3) — we are no longer inside L2 here.
        if (!new_ids.empty()) {
            referenceBlocksInGroup(p, new_ids, /*is_connector=*/false);
        }
    }

    // M03-PR3 dual-write: bump REQUEST on UnifiedRefCounter for each freshly
    // allocated super-block. Pairs with the per-pool ``requestReference``
    // fan-out above; together they keep the unified primary view and the
    // legacy per-pool view aligned for PD / connector observers.
    if (unified_ref_counter_) {
        unified_ref_counter_->bumpRange(allocated, UnifiedRefCounter::Kind::REQUEST);
    }

    // M01-PR3 dual-storage: append the freshly allocated super_block_ids to the
    // canonical KVCacheResource::super_block_ids_ vector. The per-pool
    // ``group_block_ids`` is retained above (byte-equal under bps[p]==1) so
    // existing readers see no behaviour change; Phase-6 cleanup will drop the
    // per-pool population once all consumers migrate to ``superBlockIds()``.
    {
        BlockIndicesType to_append;
        to_append.reserve(allocated.size());
        for (int s : allocated) {
            to_append.push_back(static_cast<BlockIdxType>(s));
        }
        auto& super_ids = kv_resource->superBlockIds(0);
        super_ids.insert(super_ids.end(), to_append.begin(), to_append.end());
    }

    return {true, 0};
}

void HybridPoolKVCacheAllocator::unifiedFree(const FreeInfo& free_info) {
    RTP_LLM_CHECK_WITH_INFO(super_block_allocator_ != nullptr,
                            "unifiedFree called without super_block_allocator_ (config bug)");
    auto& kv_resource = free_info.batch_kv_cache_resource;
    if (!kv_resource || kv_resource->curBlocksNum() == 0) {
        return;
    }

    const int group_nums = kv_resource->groupNums();
    const int batch_size = kv_resource->batchSize();

    // First drop per-pool request_ref_counter on every distinct block id so the
    // legacy stats (freeBlocksNum / requestRefBlocksNum) stay coherent. We do
    // NOT pop from BlockPool::free_block_ids_ here — that's by design (M01 §1):
    // the per-pool free list is dormant under unified mode and the super-block
    // free list is the single source of truth.
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        for (int gid = 0; gid < group_nums; ++gid) {
            BlockIndicesType                 valid;
            std::unordered_set<BlockIdxType> seen;
            for (auto b : kv_resource->blocks(batch_id, gid)) {
                if (isNullBlockIdx(b) || b <= 0) {
                    continue;
                }
                if (seen.insert(b).second) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                freeBlocksInGroup(gid, valid, /*is_connector=*/false);
            }
        }
    }

    // M01-PR3: collect unique super-block ids from the canonical
    // ``super_block_ids_`` vector. Under bps[p]==1 this is byte-identical to
    // the previous ``blocks(batch_id, gid=0)`` source, but reading the
    // unified field directly makes the migration explicit and unblocks the
    // Phase-6 ``group_block_ids`` removal. Fall back to the per-pool gid=0
    // view if the resource was constructed by a legacy path that bypassed
    // ``unifiedMalloc`` (e.g. older tests) — this preserves PR-2 behaviour.
    std::vector<int>        unique_S;
    std::unordered_set<int> seen_S;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const bool unified_view = kv_resource->isUnified(batch_id);
        if (unified_view) {
            for (auto b : kv_resource->superBlockIds(batch_id)) {
                if (isNullBlockIdx(b) || b <= 0) {
                    continue;
                }
                if (seen_S.insert(b).second) {
                    unique_S.push_back(b);
                }
            }
        } else {
            for (auto b : kv_resource->blocks(batch_id, /*gid=*/0)) {
                if (isNullBlockIdx(b) || b <= 0) {
                    continue;
                }
                if (seen_S.insert(b).second) {
                    unique_S.push_back(b);
                }
            }
        }
    }

    // M03-PR3 dual-write: dec REQUEST on UnifiedRefCounter (pairs with
    // ``bumpRange(REQUEST)`` issued in ``unifiedMalloc`` / connector path).
    // Then ONLY push S back to the super-block free list when isZero(S)
    // (request==0 && connector==0 && cache==0 && !useRefPinned). This is the
    // semantic shift vs PR-2: cached blocks survive the request finish; only
    // the cache eviction path (SharedBlockCache::evictAndFreeUnified) and the
    // counterpart connector-finish path will drop CACHE / CONNECTOR and
    // ultimately trigger the reclaim through the SharedBlockCache callback
    // OR through this branch on the request-only path.
    if (unified_ref_counter_) {
        unified_ref_counter_->decRange(unique_S, UnifiedRefCounter::Kind::REQUEST);
        for (int S : unique_S) {
            if (unified_ref_counter_->isZero(S)) {
                super_block_allocator_->freeSuperBlock(S);
            }
        }
    } else {
        // Fallback (counter not wired — e.g. tests bypassing doInit). Behave
        // as PR-2 did: unconditionally release every distinct S.
        for (int S : unique_S) {
            super_block_allocator_->freeSuperBlock(S);
        }
    }

    // M01-PR3 dual-storage drain: reset super_block_ids_ on every batch slot.
    // ``BatchKVCacheResource::clearBlocks`` only resizes the per-pool
    // ``group_block_ids`` to zero — the canonical super-block list is owned by
    // the underlying ``KVCacheResource`` and must be cleared explicitly so the
    // next ``unifiedMalloc`` starts from an empty list.
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        kv_resource->superBlockIds(batch_id).clear();
    }
    kv_resource->clearBlocks();
}

}  // namespace rtp_llm
