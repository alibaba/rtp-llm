#include "rtp_llm/cpp/cache_new/KVCacheManager.h"

#include <algorithm>

#include "rtp_llm/cpp/cache_new/HybridReadAsyncContext.h"
#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache_new/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache_new/KVCacheHashUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Buffer.h"

#include "rtp_llm/cpp/core/Types.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               rtp_llm::DeviceBase*               device,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const GptInitParameter&            params):
    config_(config), device_(device), metrics_reporter_(metrics_reporter), params_(params) {}

KVCacheManager::~KVCacheManager() {}

bool KVCacheManager::init() {
    bool multiple_types = config_.cache_specs.size() > 1;
    if (multiple_types) {
        RTP_LLM_LOG_ERROR("multiple types not supported");
        return false;
    }

    if (config_.cache_specs.empty()) {
        RTP_LLM_LOG_ERROR("no cache_specs");
        return false;
    }

    auto& spec = config_.cache_specs[0];
    if (spec->type != rtp_llm::KVCacheType::MultiHeadAttention
        && spec->type != rtp_llm::KVCacheType::MultiHeadLatentAttention) {
        RTP_LLM_LOG_ERROR("SingleTypeKVCacheAllocator only support Full Attention");
        return false;
    }

    allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(config_, device_, AllocationType::DEVICE);
    if (!allocator_->init()) {
        RTP_LLM_LOG_ERROR("SingleTypeKVCacheAllocator init failed");
        allocator_.reset();
        return false;
    }

    if (params_.kv_cache_config.memory_block_cache_size_mb > 0) {
        if (!initMemoryConnector()) {
            RTP_LLM_LOG_ERROR("init memory connector failed");
            return false;
        }
    }

    if (params_.kv_cache_config.enable_remote_cache) {
        if (!initRemoteConnector()) {
            RTP_LLM_LOG_ERROR("init remote connector failed");
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
    // Delegate to allocator implementation
    if (!allocator_) {
        RTP_LLM_LOG_ERROR("kvCacheBuffer called before KVCacheManager initialized");
        return {};
    }
    return allocator_->kvCacheBuffer();
}

void KVCacheManager::regUserMr(size_t model_id) {
    if (!allocator_) {
        return;
    }
    allocator_->regUserMr(model_id);
}

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    if (!allocator_) {
        return {};
    }
    return allocator_->convertIndexToAddr(layer_id, block_index);
}

bool KVCacheManager::setKVBlockValue(int              block_index,
                                     int              layer_id,
                                     rtp_llm::Buffer& k_buffer,
                                     rtp_llm::Buffer& v_buffer) {
    if (!allocator_ || !device_) {
        RTP_LLM_LOG_ERROR("setKVBlockValue called before KVCacheManager initialized");
        return false;
    }

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
    if (!allocator_ || !device_) {
        RTP_LLM_LOG_ERROR("setKVBlockValue called before KVCacheManager initialized");
        return false;
    }

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
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("malloc_info is invalid: batch_kv_cache_resource or complete_token_ids is null");
        return {false, 0};
    }
    const int seq_size_per_block = config_.seq_size_per_block;

    // Build or update cache_keys for each batch based on current complete_token_ids.
    if (!malloc_info.batch_kv_cache_resource->first_fill_finished) {
        initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
        malloc_info.batch_kv_cache_resource->first_fill_finished = true;
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    }

    return allocator_->malloc(malloc_info);
}

FreeResult KVCacheManager::free(const FreeInfo& free_info) {
    return allocator_->free(free_info);
}

InsertResult KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    InsertResult result{true};
    // TODO : reuse_cache 从名字来看应该是个全局开关, 最好是再加另一个标记, 这里先用了 enable_device_cache
    if (insert_info.reuse_cache && params_.kv_cache_config.enable_device_cache) {
        result = allocator_->insertIntoCache(insert_info);
    }

    if (insert_info.enable_memory_cache || params_.kv_cache_config.enable_remote_cache) {
        auto resource_batch0 = insert_info.batch_kv_cache_resource->batch_resource.at(0);
        auto deleter         = [insert_info, allocator = allocator_](KVCacheResourceV1* resource) {
            FreeInfo free_info(insert_info.batch_kv_cache_resource, insert_info.complete_token_ids);
            allocator->free(free_info);
            delete resource;
        };
        std::shared_ptr<KVCacheResourceV1> resource(new KVCacheResourceV1(resource_batch0), deleter);

        if (insert_info.enable_memory_cache) {
            auto context = memory_connector_->asyncWrite(resource, nullptr);
            if (context) {
                if (sync_wait_write_) {
                    context->waitDone();
                } else {
                    wait_cache_thread_pool_->pushTask([context]() { context->waitDone(); });
                }
            }
        }

        if (params_.kv_cache_config.enable_remote_cache) {
            std::string                             unique_id    = "";  // TODO : support lora
            auto                                    trace_id_str = std::to_string(insert_info.request_id);
            std::vector<int64_t>                    tokens;  // TODO : get tokens
            std::shared_ptr<KVCacheConnector::Meta> remote_connector_meta =
                std::make_shared<RemoteConnectorMeta>(unique_id, trace_id_str, tokens);
            auto async_context = remote_connector_->asyncWrite(resource, remote_connector_meta);
            if (sync_wait_write_) {
                async_context->waitDone();
            }
        }
    } else {
        FreeInfo free_info(insert_info.batch_kv_cache_resource, insert_info.complete_token_ids);
        free(free_info);
    }

    return result;
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    // return allocator_->getKVCacheInfo(latest_version, need_cache_keys);
    return {0, 0, 0, {}, latest_version};
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

size_t KVCacheManager::maxSeqLen() const {
    return allocator_->maxSeqLen();
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

bool KVCacheManager::getCacheForRank(const CacheKeysType&                      cache_keys,
                                     const BlockIndicesType&                   block_indices,
                                     size_t                                    ignore_block_num,
                                     int64_t                                   request_id,
                                     const std::map<std::string, std::string>& extra_metas) const {
    RTP_LLM_LOG_WARNING("getCacheForRank is not implemented in new KVCacheManager yet");
    return false;
}

bool KVCacheManager::putCacheForRank(const CacheKeysType&                      cache_keys,
                                     const BlockIndicesType&                   block_indices,
                                     size_t                                    ignore_block_num,
                                     int64_t                                   request_id,
                                     const std::map<std::string, std::string>& extra_metas) const {
    RTP_LLM_LOG_WARNING("putCacheForRank is not implemented in new KVCacheManager yet");
    return false;
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<BlockIdPair>&      block_update_mapping) {
    return allocator_->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
}

bool KVCacheManager::tryInitThreadPool() {
    if (wait_cache_thread_pool_ != nullptr) {
        return true;
    }

    wait_cache_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitCacheThreadPool");
    if (!wait_cache_thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("wait cache thread pool start failed");
        wait_cache_thread_pool_.reset();
        return false;
    }

    sync_wait_write_ = autil::EnvUtil::getEnv("KVCACHE_CONNECTOR_WRITE_SYNC", false);
    if (sync_wait_write_) {
        RTP_LLM_LOG_INFO("connector write kvcache in sync mode");
    }

    return true;
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

    if (!tryInitThreadPool()) {
        return false;
    }

    return true;
}

bool KVCacheManager::initRemoteConnector() {
    // TODO : get register buffer base + size
    // TODO : get lora info map
    // TODO : support different group mode
    remote_connector_ = std::make_shared<RemoteConnector>(config_,
                                                          params_,
                                                          device_,
                                                          nullptr,
                                                          0,
                                                          allocator_,
                                                          RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                                                          std::vector<int32_t>({0}),
                                                          std::vector<int32_t>({}),
                                                          metrics_reporter_);
    if (!remote_connector_->init()) {
        RTP_LLM_LOG_ERROR("kvcache remote connector init failed");
        remote_connector_.reset();
        return false;
    }

    if (!tryInitThreadPool()) {
        return false;
    }

    return true;
}

std::shared_ptr<AsyncContext> KVCacheManager::asyncLoadCache(int64_t                        request_id,
                                                             const BatchKVCacheResourcePtr& batch_resource) {
    if (!batch_resource) {
        RTP_LLM_LOG_WARNING("empry resource");
        return nullptr;
    }
    if (memory_connector_ == nullptr && remote_connector_ == nullptr) {
        RTP_LLM_LOG_WARNING("no invalid connector");
        return nullptr;
    }
    // TODO(LXQ): only support batch0 now, need to support all batch?
    std::shared_ptr<KVCacheResourceV1> resource(batch_resource, &(batch_resource->batch_resource.at(0)));
    auto context = std::make_shared<HybridReadAsyncContext>(request_id, resource, memory_connector_, remote_connector_);
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
    } else if (request.has_remote_request()) {
        if (!remote_connector_) {
            RTP_LLM_LOG_WARNING("copy cache failed, remote connector is null, request: [%s]",
                                request.DebugString().c_str());
            return false;
        }
        auto remote_connector = std::static_pointer_cast<RemoteConnector>(remote_connector_);
        return remote_connector->copyCache(request.remote_request(), *(response.mutable_remote_response()));
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
