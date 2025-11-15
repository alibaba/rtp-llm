#include "rtp_llm/cpp/cache_new/KVCacheManager.h"

#include <algorithm>

#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               rtp_llm::DeviceBase*               device,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const GptInitParameter&            params):
    config_(config), device_(device), metrics_reporter_(metrics_reporter), params_(params) {}

KVCacheManager::~KVCacheManager() {}

bool KVCacheManager::init() {
    bool multiple_types = (config_.layer_type_num > 1) || (config_.layer_type_params.size() > 1);
    if (multiple_types) {
        // if (config_.enable_independent_pool) {
        //     RTP_LLM_LOG_ERROR("HybridPoolKVCacheAllocator not implemented");
        //     return false;
        // }
        RTP_LLM_LOG_ERROR("multiple types not supported");
        return false;
    }

    if (config_.layer_type_params.empty()) {
        RTP_LLM_LOG_ERROR("no layer_type_params");
        return false;
    }

    auto& spec = config_.layer_type_params[0];
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

size_t KVCacheManager::availableTokenNums() const {
    // TODO(chanyin): implement this
    return 0;
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

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("malloc_info is invalid: batch_kv_cache_resource or complete_token_ids is null");
        return {false, 0};
    }

    int batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    int seq_size_per_block = config_.seq_size_per_block;
    int seq_len            = malloc_info.complete_token_ids->seqLength();
    int desired_blocks     = seq_len / (int)seq_size_per_block;

    // append cache_keys for eache batch
    for (int i = 0; i < batch_size; ++i) {
        auto& keys = malloc_info.batch_kv_cache_resource->cache_keys[i];

        if ((int)keys.size() > desired_blocks) {
            keys.resize(desired_blocks);
        }

        int64_t rolling_hash = keys.empty() ? 0 : keys.back();
        int     start_index  = (int)keys.size();
        if (start_index < desired_blocks) {
            auto* token_ids = malloc_info.complete_token_ids->data(i);
            for (int index = start_index; index < desired_blocks; ++index) {
                int pos = index * seq_size_per_block;
                rolling_hash =
                    rtp_llm::hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + (int)seq_size_per_block);
                keys.push_back(rolling_hash);
            }
        }

        for (auto& hash : keys) {
            RTP_LLM_LOG_INFO("DECODE: malloc cache_key %s", std::to_string(hash).c_str());
        }
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

size_t KVCacheManager::freeBlocksNums() const {
    return allocator_->freeBlocksNums();
}

size_t KVCacheManager::availableBlocksNums() const {
    return allocator_->availableBlocksNums();
}

size_t KVCacheManager::totalBlocksNums() const {
    return allocator_->totalBlocksNums();
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

std::shared_ptr<MemoryBlockCache> KVCacheManager::memoryBlockCache() const {
    RTP_LLM_LOG_WARNING("memoryBlockCache is not implemented in new KVCacheManager yet");
    return nullptr;
}

}  // namespace rtp_llm
