#include "rtp_llm/cpp/cache/KVCacheManager.h"

#include <algorithm>
#include <chrono>
#include <unordered_set>

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const KVCacheConfig&               kv_cache_config,
                               const ParallelismConfig&           parallelism_config,
                               const RuntimeConfig&               runtime_config,
                               const SpeculativeExecutionConfig&  sp_config,
                               const PDSepConfig&                 pd_sep_config,
                               const CacheStoreConfig&            cache_store_config):
    config_(config),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config),
    sp_config_(sp_config),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config) {
    if (warmup) {
        // Warmup mode: allocate minimal memory (1 block per model).
        for (auto& ac : config_.allocator_configs) {
            ac.block_num = 1;
        }
    } else {
        allocateAndSync();
    }

    const uint32_t log_layer_num = config_.allocator_configs.empty() ? 0 : config_.getAllocatorConfig(0).layer_num;
    const uint32_t log_block_num = config_.allocator_configs.empty() ? 0 : config_.getAllocatorConfig(0).block_num;
    const size_t   log_block_size =
        config_.allocator_configs.empty() ? 0 : config_.getAllocatorConfig(0).block_size_bytes;
    RTP_LLM_LOG_INFO("cache config: layer_num=%u, block_num=%u, block_size=%zuB, seq_size_per_block=%zu",
                     log_layer_num,
                     log_block_num,
                     log_block_size,
                     config_.seq_size_per_block);
}

KVCacheManager::~KVCacheManager() {
    stop_.store(true, std::memory_order_relaxed);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    allocators_.clear();
    shared_block_cache_.reset();
    coordinator_.reset();
}

// 初始化和配置相关

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!config_.allocator_configs.empty() && !config_.allocator_configs[0].cache_specs.empty(),
                            "allocator_configs must be non-empty and cache specs must not be empty");

    // Create the shared BlockCache that all allocators (models) will use for joint eviction.
    shared_block_cache_ = std::make_shared<BlockCache>();

    // Determine the number of models.
    const size_t model_num = config_.allocator_configs.empty() ? 1 : config_.allocator_configs.size();
    allocators_.reserve(model_num);
    per_model_configs_.reserve(model_num);

    for (size_t model_id = 0; model_id < model_num; ++model_id) {
        // Build a minimal single-model CacheConfig for this allocator.
        // Each allocator receives a CacheConfig that contains only its own KVCacheAllocatorConfig
        // at index 0, along with the shared fields (seq_size_per_block, kernel_seq_size_per_block).
        CacheConfig per_model_config;
        if (!config_.allocator_configs.empty()) {
            const auto& ac                               = config_.allocator_configs[model_id];
            per_model_config.seq_size_per_block          = ac.seq_size_per_block;
            per_model_config.kernel_seq_size_per_block   = config_.kernel_seq_size_per_block;
            per_model_config.layer_all_num               = ac.layer_num;
            per_model_config.layer_to_group_id           = ac.layer_to_group_id;
            per_model_config.layer_to_block_stride_bytes = ac.layer_to_block_stride_bytes;
            per_model_config.allocator_configs           = {ac};
        } else {
            per_model_config = config_;
        }

        const bool is_hybrid = per_model_config.groupNums() > 1;
        const int  group_num = per_model_config.groupNums();

        KVCacheAllocatorPtr allocator;
        if (is_hybrid) {
            allocator = std::make_shared<rtp_llm::HybridTypeKVCacheAllocator>(
                per_model_config, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        } else {
            allocator = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(
                per_model_config, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        }

        // Inject shared BlockCache so this model's BlockPool participates in joint eviction.
        allocator->setExternalBlockCache(shared_block_cache_);

        RTP_LLM_CHECK_WITH_INFO(allocator->init(), "allocator[%zu] init failed", model_id);

        // Register this model's BlockPool in the shared BlockCache registry.
        shared_block_cache_->registerModel(model_id, static_cast<size_t>(group_num), allocator->getBlockPool());

        const auto& per_ac = per_model_config.getAllocatorConfig(0);
        RTP_LLM_LOG_INFO("allocator[%zu] initialized: layer_num=%u, block_num=%u, groups=%d",
                         model_id,
                         per_ac.layer_num,
                         per_ac.block_num,
                         group_num);

        per_model_configs_.push_back(per_model_config);
        allocators_.push_back(std::move(allocator));
    }

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    initConnectorCoordinator();
    return true;
}

const CacheConfig& KVCacheManager::cacheConfig() const {
    return config_;
}

const CacheConfig& KVCacheManager::getMTPModuleCacheConfig(int mtp_module_id) const {
    const size_t idx = static_cast<size_t>(mtp_module_id) + 1;
    RTP_LLM_CHECK_WITH_INFO(idx < per_model_configs_.size(),
                            "mtp_module_id %d out of range (per_model_configs_.size()=%zu)",
                            mtp_module_id,
                            per_model_configs_.size());
    return per_model_configs_[idx];
}

// 显存管理和缓存分配

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids);

    // Cache keys are based on model 0's block state (shared across models).
    const int seq_size_per_block = config_.seq_size_per_block;
    if (!malloc_info.batch_kv_cache_resource->curBlocksNum(0)) {
        initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    }

    // Allocate blocks for all models synchronously. If any model fails, roll back all.
    // Cache reuse metadata (reuse_len) is taken from the main model (model 0).
    MallocResult main_result{false, 0};
    for (size_t i = 0; i < allocators_.size(); ++i) {
        MallocResult result = allocators_[i]->malloc(malloc_info);
        if (!result.success) {
            FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
            for (size_t j = 0; j < i; ++j) {
                allocators_[j]->free(free_info);
            }
            return result;
        }
        if (i == 0) {
            main_result = result;
        }
    }
    return main_result;
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    for (auto& allocator : allocators_) {
        allocator->free(free_info);
    }
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    RTP_LLM_PROFILE_FUNCTION();
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);

    auto& batch_kv = insert_info.batch_kv_cache_resource;
    if (!batch_kv || batch_kv->batchSize() == 0) {
        return;
    }

    // Use batch_id = 0 (single-stream insert path; beam-search multi-batch not supported).
    const auto& cache_keys = batch_kv->cacheKeys(0);
    if (cache_keys.empty()) {
        return;
    }

    const size_t model_num = allocators_.size();

    // Assemble full_slots[model_id][group_id] for each cache_key, then call
    // shared_block_cache_->upsertCacheItem once per key so all models are written
    // atomically and ref-counts are managed internally.
    // Iterate from the last key downward (consistent with per-allocator legacy order).
    for (int i = static_cast<int>(cache_keys.size()) - 1; i >= 0; --i) {
        const CacheKeyType cache_key = cache_keys[static_cast<size_t>(i)];

        std::vector<std::vector<BlockCache::CacheSlot>> full_slots(model_num);
        bool has_valid_slot = false;

        for (size_t mid = 0; mid < model_num; ++mid) {
            if (mid >= batch_kv->modelNum()) {
                // Model not yet populated in this BatchKVCacheResource — skip.
                continue;
            }
            const int group_num = per_model_configs_[mid].groupNums();
            full_slots[mid].resize(static_cast<size_t>(group_num));
            for (int gid = 0; gid < group_num; ++gid) {
                const auto& blks = batch_kv->blocks(0, gid, mid);
                if (static_cast<size_t>(i) < blks.size()) {
                    const BlockIdxType block_id = blks[static_cast<size_t>(i)];
                    if (!isNullBlockIdx(block_id)) {
                        full_slots[mid][static_cast<size_t>(gid)].block_id = block_id;
                        has_valid_slot = true;
                    }
                }
            }
        }

        if (has_valid_slot) {
            shared_block_cache_->upsertCacheItem(cache_key, full_slots, insert_info.is_resident);
        }
    }
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    return allocators_[0]->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

// 块操作相关

void KVCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    for (auto& allocator : allocators_) {
        allocator->blockCopy(src_block_index, dest_block_index);
    }
}

void KVCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    for (auto& allocator : allocators_) {
        allocator->blockBatchCopy(copy_mapping);
    }
}

void KVCacheManager::blockBatchCopy(const torch::Tensor& copy_mapping) {
    for (auto& allocator : allocators_) {
        allocator->blockBatchCopy(copy_mapping);
    }
}

void KVCacheManager::blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) {
    for (auto& allocator : allocators_) {
        allocator->blockBatchCopy(copy_mapping_begin, copy_mapping_end);
    }
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<BlockIdPair>&      block_update_mapping) {
    RTP_LLM_PROFILE_FUNCTION();
    // Delegate to the main model (model 0).  It updates the shared BatchKVCacheResource
    // structure (resetAndReturnOldResources / initGroups) and returns the block copy mapping
    // for the main model's block ID space.
    bool ok =
        allocators_[0]->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
    if (!ok || allocators_.size() == 1) {
        return ok;
    }

    // MTP sub-models (allocators_[1..N]) have their own independent block ID spaces.
    // The block_update_mapping produced by allocators_[0] references main-model block IDs
    // and MUST NOT be re-used to copy data in sub-model pools.
    //
    // Full multi-model beam-search block shuffling is not yet implemented. Silently skipping
    // the sub-model copy would leave sub-model KV data in a stale/incorrect state which would
    // cause decode correctness failures. Therefore we return false to signal to the caller that
    // the beam-search fork cannot be completed, preventing data corruption.
    if (!block_update_mapping.empty()) {
        RTP_LLM_LOG_ERROR(
            "updateKVBlock: beam-search block fork is not supported when %zu MTP sub-model allocator(s) "
            "are active. Sub-model block copy is not yet implemented. Returning failure to prevent "
            "data corruption.",
            allocators_.size() - 1);
        return false;
    }
    return true;
}

// Write one KV block (optionally per-layer) from host/device tensors for test
bool KVCacheManager::setKVBlockValue(int                  block_index,
                                     int                  layer_id,
                                     const torch::Tensor& k_buffer,
                                     const torch::Tensor& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    auto&  spec             = config_.getAllocatorConfig(0).cache_specs[0];
    size_t expected_k_bytes = spec->k_block_size_bytes();
    size_t expected_v_bytes = spec->v_block_size_bytes();
    size_t src_k_bytes      = k_buffer.nbytes();
    size_t src_v_bytes      = v_buffer.nbytes();
    if (src_k_bytes < expected_k_bytes || src_v_bytes < expected_v_bytes) {
        RTP_LLM_LOG_ERROR("setKVBlockValue src bytes too small: k[%zu]<[%zu] or v[%zu]<[%zu]",
                          src_k_bytes,
                          expected_k_bytes,
                          src_v_bytes,
                          expected_v_bytes);
        return false;
    }

    auto dst = allocators_[0]->convertIndexToBuffer(layer_id, block_index);
    RTP_LLM_CHECK_WITH_INFO(
        !dst.empty(), "convertIndexToBuffer returned empty for layer %d, block %d", layer_id, block_index);
    if (!dst[0].addr) {
        RTP_LLM_LOG_ERROR("convertIndexToBuffer returned null for layer %d, block %d", layer_id, block_index);
        return false;
    }

    auto copyFunc = [&](const torch::Tensor& src_tensor, const BlockInfo& dst_block, size_t dst_byte_offset) -> bool {
        const size_t dst_bytes = dst_block.size_bytes;
        const size_t src_bytes = src_tensor.nbytes();
        if (dst_bytes < dst_byte_offset + src_bytes) {
            RTP_LLM_LOG_ERROR("dst block bytes[%zu] < dst_offset[%zu] + src bytes[%zu] in setKVBlockValue(layer=%d)",
                              dst_bytes,
                              dst_byte_offset,
                              src_bytes,
                              layer_id);
            return false;
        }

        auto* dst_ptr    = static_cast<char*>(dst_block.addr) + dst_byte_offset;
        auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
        auto  src_device = src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
        auto  dst_t      = torch::from_blob(
            dst_ptr, {(int64_t)src_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(dst_device));
        auto src_t = torch::from_blob(src_tensor.data_ptr(),
                                      {(int64_t)src_bytes},
                                      torch::TensorOptions().dtype(torch::kUInt8).device(src_device));
        dst_t.copy_(src_t);
        return true;
    };

    if (!copyFunc(k_buffer, dst[0], 0)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst[0], expected_k_bytes)) {
        return false;
    }

    cudaSyncAndCheck();
    return true;
}

bool KVCacheManager::setKVBlockValue(int block_index, const torch::Tensor& k_buffer, const torch::Tensor& v_buffer) {
    const int main_block_num = static_cast<int>(config_.getAllocatorConfig(0).block_num);
    if (block_index < 0 || block_index >= main_block_num) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, main_block_num);
        return false;
    }

    bool all_success = true;
    for (uint32_t layer_id = 0; layer_id < config_.getAllocatorConfig(0).layer_num; ++layer_id) {
        all_success = setKVBlockValue(block_index, static_cast<int>(layer_id), k_buffer, v_buffer) && all_success;
    }
    return all_success;
}

// 地址转换和缓冲区访问

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id, size_t model_id) const {
    const size_t idx = (model_id < allocators_.size()) ? model_id : 0;
    return allocators_[idx]->convertIndexToAddr(layer_id, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, size_t model_id) const {
    const size_t idx = (model_id < allocators_.size()) ? model_id : 0;
    return allocators_[idx]->convertIndexToBuffer(layer_id, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(
    int block_index, int layer_id, int partition_count, int partition_id, size_t model_id) const {
    const size_t idx = (model_id < allocators_.size()) ? model_id : 0;
    return allocators_[idx]->convertIndexToBuffer(layer_id, block_index, partition_count, partition_id);
}

CacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return allocators_[0]->allLayerCacheBase();
}

CacheLayerLayout KVCacheManager::getCacheLayerLayout(size_t model_id) const {
    RTP_LLM_CHECK_WITH_INFO(model_id < allocators_.size(),
                            "getCacheLayerLayout: model_id %zu out of range (allocators_.size()=%zu)",
                            model_id,
                            allocators_.size());

    CacheLayerLayout layout = allocators_[model_id]->allLayerCacheBase();

    const std::vector<int>*            layer_to_group_src = nullptr;
    const std::vector<CacheGroupType>* group_types_src    = nullptr;
    const std::vector<CacheGroupType>* attn_types_src     = nullptr;
    uint32_t                           layer_num          = 0;

    if (model_id < config_.allocator_configs.size()) {
        const auto& ac     = config_.allocator_configs[model_id];
        layer_num          = ac.layer_num;
        layer_to_group_src = &ac.layer_to_group_id;
        group_types_src    = &ac.group_types;
        attn_types_src     = &ac.layer_attn_types;
    }

    // layer_to_groups
    if (layer_to_group_src && !layer_to_group_src->empty()) {
        layout.layer_to_groups.resize(layer_num);
        for (uint32_t i = 0; i < layer_num && i < layer_to_group_src->size(); ++i) {
            layout.layer_to_groups[i] = (*layer_to_group_src)[i];
        }
    }

    // group_types
    if (group_types_src && !group_types_src->empty()) {
        layout.group_types = *group_types_src;
    }

    // layer_attn_types
    layout.layer_attn_types.resize(layer_num, CacheGroupType::FULL);
    if (attn_types_src) {
        for (uint32_t i = 0; i < layer_num && i < attn_types_src->size(); ++i) {
            layout.layer_attn_types[i] = (*attn_types_src)[i];
        }
    }

    return layout;
}

CacheLayerLayout KVCacheManager::getMainModelCacheLayerLayout() const {
    return getCacheLayerLayout(0);
}

CacheLayerLayout KVCacheManager::getMTPModuleCacheLayerLayout(int mtp_module_id) const {
    // Each MTP module has its own allocator at index (mtp_module_id + 1).
    const size_t model_id = static_cast<size_t>(mtp_module_id) + 1;
    RTP_LLM_CHECK_WITH_INFO(model_id < allocators_.size(),
                            "getMTPModuleCacheLayerLayout: mtp_module_id=%d -> model_id=%zu out of range "
                            "(allocators_.size()=%zu). MTP modules must have independent allocators.",
                            mtp_module_id,
                            model_id,
                            allocators_.size());
    return getCacheLayerLayout(model_id);
}

// 资源统计和信息查询
// NOTE(multi-model): In MTP scenarios, each allocator has its own independent BlockPool.
// The methods below return the minimum across all allocators so the scheduler sees the
// most constrained model's headroom (one request consumes one block from every model).
// For pure metrics/monitoring use getKVCacheInfo() or sum across allocators directly.

size_t KVCacheManager::freeBlocksNum() const {
    if (allocators_.empty()) {
        return 0;
    }
    size_t min_free = allocators_[0]->freeBlocksNum();
    for (size_t i = 1; i < allocators_.size(); ++i) {
        min_free = std::min(min_free, allocators_[i]->freeBlocksNum());
    }
    return min_free;
}

size_t KVCacheManager::availableBlocksNum() const {
    if (allocators_.empty()) {
        return 0;
    }
    size_t min_avail = allocators_[0]->availableBlocksNum();
    for (size_t i = 1; i < allocators_.size(); ++i) {
        min_avail = std::min(min_avail, allocators_[i]->availableBlocksNum());
    }
    return min_avail;
}

size_t KVCacheManager::notInUseBlocksNum() const {
    if (allocators_.empty()) {
        return 0;
    }
    size_t min_not_in_use = allocators_[0]->notInUseBlocksNum();
    for (size_t i = 1; i < allocators_.size(); ++i) {
        min_not_in_use = std::min(min_not_in_use, allocators_[i]->notInUseBlocksNum());
    }
    return min_not_in_use;
}

BatchKVCacheResourcePtr KVCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    return allocators_[0]->popBlocksFromCache(min_blocks_to_free);
}

void KVCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    allocators_[0]->blockCacheFree(batch_kv_cache_resource);
}

size_t KVCacheManager::availableTokensNum() const {
    // Return the min across all models (bottleneck model governs scheduling capacity).
    if (allocators_.empty()) {
        return 0;
    }
    size_t min_tokens = allocators_[0]->availableTokensNum();
    for (size_t i = 1; i < allocators_.size(); ++i) {
        min_tokens = std::min(min_tokens, allocators_[i]->availableTokensNum());
    }
    return min_tokens;
}

size_t KVCacheManager::totalBlocksNum() const {
    // Return the main model's total block count to preserve backward-compatible metrics semantics.
    // The main model is always the scheduling bottleneck in a single-request view; summing across
    // all models would inflate the reported capacity in MTP scenarios and break existing monitors.
    return allocators_.empty() ? 0 : allocators_[0]->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    // Return the main model's max available tokens for scheduling capacity estimates.
    // (Consistent with totalBlocksNum semantics above.)
    return allocators_.empty() ? 0 : allocators_[0]->maxAvailableTokensNum();
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;

    if (allocators_.empty()) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        info.version = latest_version;
        return info;
    }

    if (need_cache_keys) {
        std::unordered_set<CacheKeyType> all_keys;
        // device cache keys
        auto block_cache = allocators_[0]->blockCache();
        auto snapshot    = block_cache->cacheSnapshot(latest_version);
        for (const auto& cacheItem : snapshot.values) {
            all_keys.insert(cacheItem.cache_key);
        }
        // memory cache keys
        const auto mem_cache_keys = coordinator_->memoryCacheKeys();
        all_keys.insert(mem_cache_keys.begin(), mem_cache_keys.end());

        info.cached_keys.assign(all_keys.begin(), all_keys.end());
        info.version = snapshot.version;
    }

    const size_t block_size_tokens = config_.seq_size_per_block;
    const size_t total_blocks      = allocators_[0]->totalBlocksNum();
    const size_t available_blocks  = allocators_[0]->availableBlocksNum();

    info.block_size         = block_size_tokens;
    info.total_kv_cache     = total_blocks * block_size_tokens;
    info.available_kv_cache = available_blocks * block_size_tokens;
    // cached_keys left empty for now; can be populated when distributed cache is wired up.

    return info;
}

// 系统资源管理

void KVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& allocator : allocators_) {
        allocator->regUserMr(model_id, cache_store);
    }
}

void KVCacheManager::setCacheStore(std::shared_ptr<CacheStore> cache_store) {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    cache_store_ = std::move(cache_store);
}

std::shared_ptr<CacheStore> KVCacheManager::getCacheStore() const {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    return cache_store_;
}

bool KVCacheManager::hasActiveConnectors() const {
    return coordinator_ && coordinator_->hasActiveConnectors();
}

// PD separation: increment KV cache reference count.
// Iterates all allocators so that every model's blocks are reference-counted.
// Returns the main model (allocators_[0]) KVCacheResource for backward-compatible callers.
std::shared_ptr<KVCacheResource> KVCacheManager::incrKVCacheRef(const ModelKVResources& model_resources,
                                                                const CacheKeysType&    cache_keys,
                                                                bool                    is_connector) {
    std::shared_ptr<KVCacheResource> main_resource;
    for (size_t mid = 0; mid < allocators_.size(); ++mid) {
        auto res = allocators_[mid]->incrKVCacheRef(model_resources, cache_keys, is_connector);
        if (mid == 0) {
            main_resource = std::move(res);
        }
    }
    return main_resource;
}

bool KVCacheManager::hasP2PConnector() const {
    return coordinator_ && coordinator_->hasP2PConnector();
}

// 异步连接器操作

std::shared_ptr<AsyncContext>
KVCacheManager::asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    return coordinator_->asyncRead(connector_context);
}

std::shared_ptr<AsyncContext>
KVCacheManager::asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    return coordinator_->asyncWrite(connector_context);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    return coordinator_->executeFunction(request, response);
}

void KVCacheManager::initConnectorCoordinator() {
    RTP_LLM_LOG_INFO(
        "init connector coordinator, cache config: [%s], kv cache config: [%s], runtime config: [%s], parallelism config: [%s], sp config: [%s]",
        config_.debugString().c_str(),
        kv_cache_config_.to_string().c_str(),
        runtime_config_.to_string().c_str(),
        parallelism_config_.to_string().c_str(),
        sp_config_.to_string().c_str());
    coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(config_,
                                                                 kv_cache_config_,
                                                                 runtime_config_,
                                                                 parallelism_config_,
                                                                 sp_config_,
                                                                 allocators_[0],
                                                                 metrics_reporter_,
                                                                 pd_sep_config_,
                                                                 cache_store_config_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_->init(), "connector coordinator init failed");
}

void KVCacheManager::allocateAndSync() {
    size_t world_size = parallelism_config_.tp_size * parallelism_config_.dp_size;
    if (world_size > 1 && !config_.allocator_configs.empty()) {
        size_t local_rank    = parallelism_config_.tp_size * parallelism_config_.dp_rank + parallelism_config_.tp_rank;
        auto   block_num_t   = torch::empty({(int64_t)world_size}, torch::kInt32).pin_memory();
        auto   block_num_ptr = block_num_t.data_ptr<int>();
        block_num_ptr[local_rank] = static_cast<int>(config_.allocator_configs[0].block_num);
        execAllGather({{block_num_t}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        uint32_t synced_main_block_num;
        if (parallelism_config_.ffn_disaggregate_config.is_ffn_service()) {
            synced_main_block_num = 1;
        } else {
            synced_main_block_num = static_cast<uint32_t>(*std::min_element(block_num_ptr, block_num_ptr + world_size));
        }
        // Propagate the synced block_num into all allocator configs.
        // NOTE(P1-4): This sets every model's block_num to the same value, which overrides
        // any ratio-based allocation computed by CacheConfigCreator::createSpConfig when
        // MAIN_MODEL_KVCACHE_RATIO is non-zero.  A proper fix would sync each model's
        // block_num independently (one allgather per model), but this is deferred until
        // ratio-based multi-model allocation is validated end-to-end.
        for (auto& alloc_config : config_.allocator_configs) {
            alloc_config.block_num = synced_main_block_num;
        }
    }
    RTP_LLM_LOG_INFO("block_num is %d after tp sync", config_.getAllocatorConfig(0).block_num);
}

void KVCacheManager::reportMetricsLoop() {
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    while (!stop_.load(std::memory_order_relaxed)) {
        if (!metrics_reporter_ || allocators_.empty()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector collector;

        auto block_cache = allocators_[0]->blockCache();

        const auto total_blocks         = allocators_[0]->totalBlocksNum();
        const auto available_blocks     = allocators_[0]->availableBlocksNum();
        const auto request_ref_blocks   = allocators_[0]->requestRefBlocksNum();
        const auto connector_ref_blocks = allocators_[0]->connectorRefBlocksNum();

        collector.kv_cache_item_num             = block_cache ? static_cast<int64_t>(block_cache->size()) : 0;
        collector.kv_cache_left_seq             = static_cast<int64_t>(available_blocks * config_.seq_size_per_block);
        collector.kv_cache_available_blocks     = static_cast<int64_t>(available_blocks);
        collector.kv_cache_request_ref_blocks   = static_cast<int64_t>(request_ref_blocks);
        collector.kv_cache_connector_ref_blocks = static_cast<int64_t>(connector_ref_blocks);
        collector.kv_cache_free_blocks          = static_cast<int64_t>(allocators_[0]->freeBlocksNum());
        collector.kv_cache_used_ratio =
            (total_blocks == 0) ?
                0.0f :
                static_cast<float>(100.0 * (total_blocks - available_blocks) / static_cast<double>(total_blocks));
        collector.mr_cost_time_ms = allocators_[0]->getMrCostTimeMs();

        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &collector);
        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

void KVCacheManager::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                P2PConnectorStartLoadResponsePB&      response,
                                std::function<bool()>                 is_cancelled) {
    if (coordinator_) {
        coordinator_->handleRead(request, response, is_cancelled);
    }
}

}  // namespace rtp_llm
