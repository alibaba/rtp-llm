#include "rtp_llm/cpp/cache_new/KVCacheManager.h"

#include <algorithm>

#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache_new/KVCacheHashUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Buffer.h"

#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               rtp_llm::DeviceBase*               device,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const GptInitParameter&            params):
    // TODO, warmup metrics_reporter params 都没有用起来
    config_(config), device_(device), metrics_reporter_(metrics_reporter), params_(params) {}

KVCacheManager::~KVCacheManager() {}

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(config_.cache_specs.size() == 1, "cache specs size should be 1");

    auto& spec = config_.cache_specs[0];
    if (spec->type == rtp_llm::KVCacheType::MultiHeadAttention
        || spec->type == rtp_llm::KVCacheType::MultiHeadLatentAttention) {
        allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(config_, device_, AllocationType::DEVICE);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "SingleTypeKVCacheAllocator init failed");
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "SingleTypeKVCacheAllocator only support Full Attention");
        return false;
    }

    if (params_.kv_cache_config.memory_block_cache_size_mb > 0) {
        if (!initMemoryConnector()) {
            RTP_LLM_LOG_ERROR("init memory connector failed");
            return false;
        }
    }
    return true;
}

size_t KVCacheManager::availableTokensNum() const {
    return allocator_->availableTokensNum();
}

const CacheConfig& KVCacheManager::cacheConfig() const {
    return config_;
}

CacheLayerLayout KVCacheManager::layerCacheBase() const {
    return allocator_->layerCacheBase();
}

KVCacheBuffer KVCacheManager::kvCacheBuffer() const {
    return allocator_->kvCacheBuffer();
}

void KVCacheManager::regUserMr(size_t model_id) {
    allocator_->regUserMr(model_id);
}

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return allocator_->convertIndexToAddr(layer_id, block_index);
}

bool KVCacheManager::setKVBlockValue(int              block_index,
                                     int              layer_id,
                                     rtp_llm::Buffer& k_buffer,
                                     rtp_llm::Buffer& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    auto&  spec             = config_.cache_specs[0];
    size_t expected_k_bytes = spec->k_block_size();
    size_t expected_v_bytes = spec->v_block_size();
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
    if (!dst.k_addr || !dst.v_addr) {
        RTP_LLM_LOG_ERROR("convertIndexToBuffer returned null for layer %d, block %d", layer_id, block_index);
        return false;
    }

    auto copyFunc = [&](rtp_llm::Buffer& src_buffer, rtp_llm::BufferPtr& dst_buffer, size_t dst_byte_offset) -> bool {
        const size_t dst_bytes = dst_buffer->sizeBytes();
        const size_t src_bytes = src_buffer.sizeBytes();
        if (dst_bytes < dst_byte_offset + src_bytes) {
            RTP_LLM_LOG_ERROR("dst block bytes[%zu] < dst_offset[%zu] + src bytes[%zu] in setKVBlockValue(layer=%d)",
                              dst_bytes,
                              dst_byte_offset,
                              src_bytes,
                              layer_id);
            return false;
        }

        auto*           dst_ptr = static_cast<char*>(dst_buffer->data()) + dst_byte_offset;
        rtp_llm::Buffer dst_view(dst_buffer->where(), src_buffer.type(), {src_buffer.size()}, dst_ptr);
        rtp_llm::Buffer src_view(src_buffer.where(), src_buffer.type(), {src_buffer.size()}, src_buffer.data());
        device_->copy({dst_view, src_view});
        return true;
    };

    if (!copyFunc(k_buffer, dst.k_addr, 0)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst.v_addr, expected_k_bytes)) {
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
    if (!malloc_info.batch_kv_cache_resource->first_fill_finished) {
        initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
        malloc_info.batch_kv_cache_resource->first_fill_finished = true;
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    }

    return allocator_->malloc(malloc_info);
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    if (free_info.reuse_cache || free_info.enable_memory_cache) {
        InsertInfo insert_info{free_info.batch_kv_cache_resource,
                               free_info.complete_token_ids,
                               /*is_resident*/ false,
                               free_info.reuse_cache,
                               free_info.enable_memory_cache};
        // free blocks inside
        insertIntoCache(insert_info);
        return;
    }
    allocator_->free(free_info);
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);

    // insert to gpu
    if (insert_info.reuse_cache) {
        allocator_->insertIntoCache(insert_info);
    }

    // insert to cpu
    if (insert_info.enable_memory_cache) {
        // 拷贝一下batch resource, 外部可能会对batch resource中的blocks进行修改, 导致deleter中free时blocks未被释放
        auto     copy_batch_resource = std::make_shared<BatchKVCacheResource>(*(insert_info.batch_kv_cache_resource));
        FreeInfo free_info{copy_batch_resource, insert_info.complete_token_ids};
        auto deleter = [free_info, allocator = allocator_](KVCacheResourceV1* resource) { allocator->free(free_info); };
        std::shared_ptr<KVCacheResourceV1> resource(&(copy_batch_resource->batch_resource.at(0)), deleter);
        auto                               context = memory_connector_->asyncWrite(resource, nullptr);
        if (context) {
            wait_cache_thread_pool_->pushTask([context]() { context->waitDone(); });
        }
    } else {
        FreeInfo free_info{insert_info.batch_kv_cache_resource, insert_info.complete_token_ids};
        allocator_->free(free_info);
    }
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    KVCacheInfo info;

    if (!allocator_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        info.version = latest_version;
        return info;
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

bool KVCacheManager::initMemoryConnector() {
    const auto memory_block_cache_size_mb         = params_.kv_cache_config.memory_block_cache_size_mb;
    const auto memory_block_cache_sync_timeout_ms = params_.kv_cache_config.memory_block_cache_sync_timeout_ms;
    if (memory_block_cache_size_mb <= 0 || memory_block_cache_sync_timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING(
            "init memory connector failed, memory size or sync timeout is invalid, memory size: %ld MB, sync timeout: %ld ms",
            memory_block_cache_size_mb,
            memory_block_cache_sync_timeout_ms);
        return false;
    }

    config_.memory_block_cache_size_mb         = memory_block_cache_size_mb;
    config_.memory_block_cache_sync_timeout_ms = memory_block_cache_sync_timeout_ms;
    RTP_LLM_LOG_INFO("init memory connector, size: %ld MB, sync timeout: %ld ms",
                     config_.memory_block_cache_size_mb,
                     config_.memory_block_cache_sync_timeout_ms);

    memory_connector_ =
        std::make_shared<KVCacheMemoryConnector>(config_, allocator_, device_, params_.worker_grpc_addrs_);
    if (!memory_connector_->init()) {
        RTP_LLM_LOG_ERROR("memory connector init failed");
        memory_connector_.reset();
        return false;
    }

    wait_cache_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitCacheThreadPool");
    if (!wait_cache_thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("wait cache thread pool start failed");
        wait_cache_thread_pool_.reset();
        return false;
    }

    return true;
}

std::shared_ptr<AsyncContext> KVCacheManager::asyncLoadCache(const BatchKVCacheResourcePtr& batch_resource) {
    if (!memory_connector_ || !batch_resource) {
        RTP_LLM_LOG_WARNING(
            "async load cache failed, memory connector or resource is null, memory connector: %p, resource: %p",
            memory_connector_.get(),
            batch_resource.get());
        return nullptr;
    }

    // TODO(LXQ): only support batch0 now, need to support all batch?
    std::shared_ptr<KVCacheResourceV1> resource(batch_resource, &(batch_resource->batch_resource.at(0)));
    auto                               context = memory_connector_->asyncRead(resource, nullptr);
    if (context) {
        wait_cache_thread_pool_->pushTask([context]() { context->waitDone(); });
    }
    return context;
}

bool KVCacheManager::copyCache(const CopyCacheRequestPB& request, CopyCacheResponsePB& response) {
    if (request.has_mem_request()) {
        if (!memory_connector_) {
            RTP_LLM_LOG_WARNING("copy cache failed, memory connector is null, request: [%s]",
                                request.DebugString().c_str());
            response.mutable_mem_response()->set_success(false);
            return false;
        }
        auto memory_connector = std::dynamic_pointer_cast<KVCacheMemoryConnector>(memory_connector_);
        if (!memory_connector) {
            RTP_LLM_LOG_WARNING("copy cache failed, memory connector is not a KVCacheMemoryConnector");
            response.mutable_mem_response()->set_success(false);
            return false;
        }
        return memory_connector->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else {
        RTP_LLM_LOG_WARNING("copy cache failed, request is invalid, request: [%s]", request.DebugString().c_str());
        return false;
    }
}

void KVCacheManager::clearLocalCache() {
    // clear gpu cache
    if (allocator_) {
        allocator_->clearCache();
    }
    // clear cpu cache
    if (memory_connector_) {
        auto memory_connector = std::dynamic_pointer_cast<KVCacheMemoryConnector>(memory_connector_);
        if (memory_connector) {
            memory_connector->clearCache();
        }
    }
}

}  // namespace rtp_llm
