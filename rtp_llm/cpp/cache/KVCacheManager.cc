#include "rtp_llm/cpp/cache/KVCacheManager.h"

#include <algorithm>
#include <chrono>

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Buffer.h"

#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               rtp_llm::DeviceBase*               device,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const KVCacheConfig&               kv_cache_config,
                               const ParallelismConfig&           parallelism_config,
                               const RuntimeConfig&               runtime_config):
    config_(config),
    device_(device),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config) {
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

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "cache specs must not be empty");

    const bool is_hybrid = config_.groupNums() > 1;
    if (is_hybrid) {
        allocator_ = std::make_shared<rtp_llm::HybridLayerKVCacheAllocator>(
            config_, device_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "HybridLayerKVCacheAllocator init failed");
    } else {
        allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(
            config_, device_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "SingleTypeKVCacheAllocator init failed");
    }

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    initConnectorCoordinator();
    return true;
}

size_t KVCacheManager::availableTokensNum() const {
    return allocator_->availableTokensNum();
}

const CacheConfig& KVCacheManager::cacheConfig() const {
    return config_;
}

CacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return allocator_->allLayerCacheBase();
}

KVCacheBuffer KVCacheManager::kvCacheBuffer() const {
    return allocator_->kvCacheBuffer();
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    return allocator_->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

void KVCacheManager::regUserMr(size_t model_id) {
    allocator_->regUserMr(model_id);
}

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return allocator_->convertIndexToAddr(layer_id, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index, partition_count, partition_id);
}

bool KVCacheManager::setKVBlockValue(int              block_index,
                                     int              layer_id,
                                     rtp_llm::Buffer& k_buffer,
                                     rtp_llm::Buffer& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    auto&  spec             = config_.cache_specs[0];
    size_t expected_k_bytes = spec->k_block_size_bytes();
    size_t expected_v_bytes = spec->v_block_size_bytes();
    size_t src_k_bytes      = k_buffer.size() * rtp_llm::getTypeSize(k_buffer.type());
    size_t src_v_bytes      = v_buffer.size() * rtp_llm::getTypeSize(v_buffer.type());
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

    auto copyFunc = [&](rtp_llm::Buffer& src_buffer, const BlockInfo& dst_block, size_t dst_byte_offset) -> bool {
        const size_t dst_bytes = dst_block.size_bytes;
        const size_t src_bytes = src_buffer.sizeBytes();
        if (dst_bytes < dst_byte_offset + src_bytes) {
            RTP_LLM_LOG_ERROR("dst block bytes[%zu] < dst_offset[%zu] + src bytes[%zu] in setKVBlockValue(layer=%d)",
                              dst_bytes,
                              dst_byte_offset,
                              src_bytes,
                              layer_id);
            return false;
        }

        auto*           dst_ptr   = static_cast<char*>(dst_block.addr) + dst_byte_offset;
        auto            dst_where = dst_block.is_cuda ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
        rtp_llm::Buffer dst_view(dst_where, src_buffer.type(), {src_buffer.size()}, dst_ptr);
        rtp_llm::Buffer src_view(src_buffer.where(), src_buffer.type(), {src_buffer.size()}, src_buffer.data());
        device_->copy({dst_view, src_view});
        return true;
    };

    if (!copyFunc(k_buffer, dst[0], 0)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst[0], expected_k_bytes)) {
        return false;
    }

    device_->syncAndCheck();
    return true;
}

bool KVCacheManager::setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer) {
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

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
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
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    allocator_->free(free_info);
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    allocator_->insertIntoCache(insert_info);
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;

    if (!allocator_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        info.version = latest_version;
        return info;
    }

    if (need_cache_keys) {
        auto block_cache = allocator_->getBlockPool()->blockCache();
        auto snapshot    = block_cache->cacheSnapshot(latest_version);
        info.cached_keys.clear();
        info.cached_keys.reserve(snapshot.values.size());
        for (const auto& cacheItem : snapshot.values) {
            info.cached_keys.push_back(cacheItem.cache_key);
        }
    }

    const size_t block_size_tokens = config_.seq_size_per_block;
    const size_t total_blocks      = allocator_->totalBlocksNum();
    const size_t available_blocks  = allocator_->availableBlocksNum();

    info.block_size         = block_size_tokens;
    info.total_kv_cache     = total_blocks * block_size_tokens;
    info.available_kv_cache = available_blocks * block_size_tokens;
    info.version            = latest_version;
    // cached_keys left empty for now; can be populated when distributed cache is wired up.

    return info;
}

size_t KVCacheManager::freeBlocksNum() const {
    return allocator_->freeBlocksNum();
}

size_t KVCacheManager::availableBlocksNum() const {
    return allocator_->availableBlocksNum();
}

size_t KVCacheManager::totalBlocksNum() const {
    return allocator_->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    return allocator_->maxAvailableTokensNum();
}

void KVCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    return allocator_->blockCopy(src_block_index, dest_block_index);
}

void KVCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const rtp_llm::Buffer& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) {
    return allocator_->blockBatchCopy(copy_mapping_begin, copy_mapping_end);
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<BlockIdPair>&      block_update_mapping) {
    return allocator_->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
}

void KVCacheManager::allocateAndSync() {
    const auto properties = device_->getDeviceProperties();
    size_t     world_size = properties.tp_size * properties.dp_size;
    if (world_size > 1) {
        size_t    local_rank = properties.tp_size * properties.dp_rank + properties.tp_rank;
        BufferPtr block_num_infos =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {world_size}, rtp_llm::AllocationType::HOST});
        auto block_num_ptr        = block_num_infos->data<int>();
        block_num_ptr[local_rank] = config_.block_num;
        device_->allGather({{block_num_infos}, ParallelMode::DP_AND_TP});
        device_->syncCommunication(false);
        device_->syncAndCheck();

        if (properties.ffn_as_service) {
            config_.block_num = 1;
        } else {
            config_.block_num = *std::min_element(block_num_ptr, block_num_ptr + world_size);
        }
    }
    RTP_LLM_LOG_INFO("block_num is %d after tp sync", config_.block_num);
}

void KVCacheManager::reportMetricsLoop() {
    kmonitor::MetricsTags tags;
    while (!stop_.load(std::memory_order_relaxed)) {
        if (!metrics_reporter_ || !allocator_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector collector;

        auto block_pool  = allocator_->getBlockPool();
        auto block_cache = block_pool ? block_pool->blockCache() : nullptr;

        const auto total_blocks     = allocator_->totalBlocksNum();
        const auto available_blocks = allocator_->availableBlocksNum();

        collector.kv_cache_item_num         = block_cache ? static_cast<int64_t>(block_cache->size()) : 0;
        collector.kv_cache_left_seq         = static_cast<int64_t>(available_blocks * config_.seq_size_per_block);
        collector.kv_cache_available_blocks = static_cast<int64_t>(available_blocks);
        collector.kv_cache_free_blocks      = static_cast<int64_t>(allocator_->freeBlocksNum());
        collector.kv_cache_used_ratio =
            (total_blocks == 0) ?
                0.0f :
                static_cast<float>(100.0 * (total_blocks - available_blocks) / static_cast<double>(total_blocks));
        collector.mr_cost_time_ms = allocator_->getMrCostTimeMs();

        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &collector);
        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

KVCacheBuffer KVCacheManager::getMTPModuleKVCacheBuffer(int mtp_module_id) const {
    return allocator_->getMTPModuleKVCacheBuffer(mtp_module_id);
}

const CacheConfig& KVCacheManager::getMTPModuleCacheConfig(int mtp_module_id) const {
    return *config_.mtp_sub_configs[mtp_module_id];
}

void KVCacheManager::initConnectorCoordinator() {
    coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(
        config_, kv_cache_config_, runtime_config_, allocator_, device_, metrics_reporter_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_->init(), "connector coordinator init failed");
}

std::shared_ptr<AsyncContext>
KVCacheManager::asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    return coordinator_->asyncRead(connector_context);
}

std::shared_ptr<AsyncContext>
KVCacheManager::asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    return coordinator_->asyncWrite(connector_context);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    return coordinator_->executeFunction(request, response);
}

}  // namespace rtp_llm
