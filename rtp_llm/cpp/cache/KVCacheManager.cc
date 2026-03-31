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
        config_.block_num = 1;
    } else {
        allocateAndSync();
    }

    RTP_LLM_LOG_INFO("cache config: layer_num=%d, block_num=%d, block_size=%dB, seq_size_per_block=%zu",
                     config_.layer_num,
                     config_.block_num,
                     config_.block_size_bytes,
                     config_.seq_size_per_block);
}

KVCacheManager::~KVCacheManager() {
    stop_.store(true, std::memory_order_relaxed);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    allocator_.reset();
    coordinator_.reset();
}

// 初始化和配置相关

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "cache specs must not be empty");

    const bool is_hybrid = config_.groupNums() > 1;
    if (is_hybrid) {
        allocator_ = std::make_shared<rtp_llm::HybridTypeKVCacheAllocator>(
            config_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "HybridTypeKVCacheAllocator init failed");
    } else {
        allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(
            config_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "SingleTypeKVCacheAllocator init failed");
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
    return *config_.mtp_sub_configs[mtp_module_id];
}

// 显存管理和缓存分配

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids);

    const int seq_size_per_block = config_.seq_size_per_block;
    if (!malloc_info.batch_kv_cache_resource->curBlocksNum()) {
        initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    }

    return allocator_->malloc(malloc_info);
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    allocator_->free(free_info);
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    RTP_LLM_PROFILE_FUNCTION();
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    allocator_->insertIntoCache(insert_info);
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    return allocator_->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

// 块操作相关

void KVCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    return allocator_->blockCopy(src_block_index, dest_block_index);
}

void KVCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const torch::Tensor& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) {
    return allocator_->blockBatchCopy(copy_mapping_begin, copy_mapping_end);
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<BlockIdPair>&      block_update_mapping) {
    RTP_LLM_PROFILE_FUNCTION();
    return allocator_->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
}

// Write one KV block (optionally per-layer) from host/device tensors for test
bool KVCacheManager::setKVBlockValue(int                  block_index,
                                     int                  layer_id,
                                     const torch::Tensor& k_buffer,
                                     const torch::Tensor& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    auto&  spec             = config_.cache_specs[0];
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

    auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);
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
    if (block_index < 0 || block_index >= config_.block_num) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_num);
        return false;
    }

    bool all_success = true;
    for (int layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        all_success = setKVBlockValue(block_index, layer_id, k_buffer, v_buffer) && all_success;
    }
    return all_success;
}

// 地址转换和缓冲区访问

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return allocator_->convertIndexToAddr(layer_id, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(int block_index, int layer_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index, partition_count, partition_id);
}

CacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return allocator_->allLayerCacheBase();
}

CacheLayerLayout KVCacheManager::getMainModelCacheLayerLayout() const {
    CacheLayerLayout layout;

    auto  all_layout        = allocator_->allLayerCacheBase();
    auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
    auto& all_scale_tensors = all_layout.layers_to_scale_buffer_ptrs;

    layout.layer_to_groups.resize(config_.layer_num);
    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_num);
    if (!all_scale_tensors.empty()) {
        layout.layers_to_scale_buffer_ptrs.resize(config_.layer_num);
    }

    layout.layer_to_groups = config_.layer_to_group_id;
    layout.group_types     = config_.group_types;
    layout.layer_attn_types.resize(config_.layer_num, CacheGroupType::FULL);

    RTP_LLM_CHECK_WITH_INFO(config_.layer_num <= all_layer_tensors.size(),
                            "config_.layer_num[%d] > all_layer_tensors.size()[%ld]",
                            config_.layer_num,
                            all_layer_tensors.size());

    for (int layer_id = 0; layer_id < static_cast<int>(config_.layer_num); ++layer_id) {
        if (static_cast<size_t>(layer_id) < all_layer_tensors.size()) {
            layout.layer_to_groups[layer_id]          = all_layout.layer_to_groups[layer_id];
            layout.layers_to_kv_buffer_ptrs[layer_id] = all_layer_tensors[layer_id];
        } else {
            RTP_LLM_CHECK(false);
        }

        if (!all_scale_tensors.empty()) {
            if (static_cast<size_t>(layer_id) < all_scale_tensors.size()) {
                layout.layers_to_scale_buffer_ptrs[layer_id] = all_scale_tensors[layer_id];
            } else {
                RTP_LLM_CHECK(false);
            }
        }
        if (static_cast<size_t>(layer_id) < config_.layer_attn_types.size()) {
            layout.layer_attn_types[layer_id] = config_.layer_attn_types[static_cast<size_t>(layer_id)];
        }
    }

    return layout;
}

CacheLayerLayout KVCacheManager::getMTPModuleCacheLayerLayout(int mtp_module_id) const {
    CacheLayerLayout layout;

    RTP_LLM_CHECK_WITH_INFO(mtp_module_id >= 0 && static_cast<size_t>(mtp_module_id) < config_.mtp_sub_configs.size(),
                            "Invalid mtp_module_id: %d, must be in range [0, %zu)",
                            mtp_module_id,
                            config_.mtp_sub_configs.size());

    const auto& mtp_sub_config = config_.mtp_sub_configs[mtp_module_id];
    RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%d] is null", mtp_module_id);
    RTP_LLM_CHECK_WITH_INFO(
        !mtp_sub_config->global_layer_ids.empty(), "mtp_sub_configs[%d]->global_layer_ids is empty", mtp_module_id);
    RTP_LLM_CHECK_WITH_INFO(!mtp_sub_config->global_layer_ids[0].empty(),
                            "mtp_sub_configs[%d]->global_layer_ids[0] is empty",
                            mtp_module_id);

    const auto&    mtp_global_layer_ids = mtp_sub_config->global_layer_ids[0];
    const uint32_t mtp_layer_num        = mtp_sub_config->layer_num;

    auto  all_layout        = allocator_->allLayerCacheBase();
    auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
    auto& all_scale_tensors = all_layout.layers_to_scale_buffer_ptrs;

    layout.layer_to_groups.resize(mtp_layer_num);
    layout.layers_to_kv_buffer_ptrs.resize(mtp_layer_num);
    if (!all_scale_tensors.empty()) {
        layout.layers_to_scale_buffer_ptrs.resize(mtp_layer_num);
    }
    layout.layer_attn_types.resize(mtp_layer_num, CacheGroupType::FULL);

    for (uint32_t local_layer_id = 0; local_layer_id < mtp_layer_num; ++local_layer_id) {
        if (local_layer_id < mtp_global_layer_ids.size()) {
            const int global_layer_id = mtp_global_layer_ids[local_layer_id];

            if (global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < all_layer_tensors.size()) {
                layout.layer_to_groups[local_layer_id]          = all_layout.layer_to_groups[global_layer_id];
                layout.layers_to_kv_buffer_ptrs[local_layer_id] = all_layer_tensors[global_layer_id];
            } else {
                RTP_LLM_CHECK(false);
            }

            if (!all_scale_tensors.empty()) {
                if (global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < all_scale_tensors.size()) {
                    layout.layers_to_scale_buffer_ptrs[local_layer_id] = all_scale_tensors[global_layer_id];
                } else {
                    RTP_LLM_CHECK(false);
                }
            }
            if (local_layer_id < mtp_sub_config->layer_attn_types.size()) {
                layout.layer_attn_types[local_layer_id] = mtp_sub_config->layer_attn_types[local_layer_id];
            }
        } else {
            RTP_LLM_CHECK(false);
        }
    }

    return layout;
}

// 资源统计和信息查询

size_t KVCacheManager::freeBlocksNum() const {
    return allocator_->freeBlocksNum();
}

size_t KVCacheManager::availableBlocksNum() const {
    return allocator_->availableBlocksNum();
}

size_t KVCacheManager::notInUseBlocksNum() const {
    return allocator_->notInUseBlocksNum();
}

BatchKVCacheResourcePtr KVCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    return allocator_->popBlocksFromCache(min_blocks_to_free);
}

void KVCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    allocator_->blockCacheFree(batch_kv_cache_resource);
}

size_t KVCacheManager::availableTokensNum() const {
    return allocator_->availableTokensNum();
}

size_t KVCacheManager::totalBlocksNum() const {
    return allocator_->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    return allocator_->maxAvailableTokensNum();
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;

    if (!allocator_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        info.version = latest_version;
        return info;
    }

    if (need_cache_keys) {
        std::unordered_set<CacheKeyType> all_keys;
        // device cache keys
        auto block_cache = allocator_->getBlockPool()->blockCache();
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
    const size_t total_blocks      = allocator_->totalBlocksNum();
    const size_t available_blocks  = allocator_->availableBlocksNum();

    info.block_size         = block_size_tokens;
    info.total_kv_cache     = total_blocks * block_size_tokens;
    info.available_kv_cache = available_blocks * block_size_tokens;
    // cached_keys left empty for now; can be populated when distributed cache is wired up.

    return info;
}

// 系统资源管理

void KVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    allocator_->regUserMr(model_id, std::move(cache_store));
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
                                                                 allocator_,
                                                                 metrics_reporter_,
                                                                 pd_sep_config_,
                                                                 cache_store_config_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_->init(), "connector coordinator init failed");
}

void KVCacheManager::allocateAndSync() {
    size_t world_size = parallelism_config_.tp_size * parallelism_config_.dp_size;
    if (world_size > 1) {
        size_t local_rank    = parallelism_config_.tp_size * parallelism_config_.dp_rank + parallelism_config_.tp_rank;
        auto   block_num_t   = torch::empty({(int64_t)world_size}, torch::kInt32).pin_memory();
        auto   block_num_ptr = block_num_t.data_ptr<int>();
        block_num_ptr[local_rank] = config_.block_num;
        execAllGather({{block_num_t}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        if (parallelism_config_.ffn_disaggregate_config.is_ffn_service()) {
            config_.block_num = 1;
        } else {
            config_.block_num = *std::min_element(block_num_ptr, block_num_ptr + world_size);
        }
    }
    RTP_LLM_LOG_INFO("block_num is %d after tp sync", config_.block_num);
}

void KVCacheManager::reportMetricsLoop() {
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    while (!stop_.load(std::memory_order_relaxed)) {
        if (!metrics_reporter_ || !allocator_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector collector;

        auto block_pool  = allocator_->getBlockPool();
        auto block_cache = block_pool ? block_pool->blockCache() : nullptr;

        const auto total_blocks         = allocator_->totalBlocksNum();
        const auto available_blocks     = allocator_->availableBlocksNum();
        const auto request_ref_blocks   = allocator_->requestRefBlocksNum();
        const auto connector_ref_blocks = allocator_->connectorRefBlocksNum();

        collector.kv_cache_item_num             = block_cache ? static_cast<int64_t>(block_cache->size()) : 0;
        collector.kv_cache_left_seq             = static_cast<int64_t>(available_blocks * config_.seq_size_per_block);
        collector.kv_cache_available_blocks     = static_cast<int64_t>(available_blocks);
        collector.kv_cache_request_ref_blocks   = static_cast<int64_t>(request_ref_blocks);
        collector.kv_cache_connector_ref_blocks = static_cast<int64_t>(connector_ref_blocks);
        collector.kv_cache_free_blocks          = static_cast<int64_t>(allocator_->freeBlocksNum());
        collector.kv_cache_used_ratio =
            (total_blocks == 0) ?
                0.0f :
                static_cast<float>(100.0 * (total_blocks - available_blocks) / static_cast<double>(total_blocks));
        collector.mr_cost_time_ms = allocator_->getMrCostTimeMs();

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
