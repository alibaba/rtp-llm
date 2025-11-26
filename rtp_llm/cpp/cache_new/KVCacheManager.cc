#include "rtp_llm/cpp/cache_new/KVCacheManager.h"

#include <algorithm>

#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
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
    if (spec->type == rtp_llm::KVCacheType::MultiHeadAttention
        || spec->type == rtp_llm::KVCacheType::MultiHeadLatentAttention) {
        allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(config_, device_, AllocationType::DEVICE);
        if (!allocator_->init()) {
            RTP_LLM_LOG_ERROR("SingleTypeKVCacheAllocator init failed");
            allocator_.reset();
            return false;
        }
        return true;
    } else {
        RTP_LLM_LOG_ERROR("SingleTypeKVCacheAllocator only support Full Attention");
        return false;
    }
    return false;
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
        initCacheKeys(*malloc_info.batch_kv_cache_resource, *malloc_info.complete_token_ids, seq_size_per_block);
        malloc_info.batch_kv_cache_resource->first_fill_finished = true;
    } else {
        updateCacheKeys(*malloc_info.batch_kv_cache_resource, *malloc_info.complete_token_ids, seq_size_per_block);
    }

    return allocator_->malloc(malloc_info);
}

FreeResult KVCacheManager::free(const FreeInfo& free_info) {
    return allocator_->free(free_info);
}

InsertResult KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    return allocator_->insertIntoCache(insert_info);
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

std::shared_ptr<MemoryBlockCache> KVCacheManager::memoryBlockCache() const {
    RTP_LLM_LOG_WARNING("memoryBlockCache is not implemented in new KVCacheManager yet");
    return nullptr;
}

}  // namespace rtp_llm
