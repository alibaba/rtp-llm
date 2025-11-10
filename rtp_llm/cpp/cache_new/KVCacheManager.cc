#include "rtp_llm/cpp/cache_new/KVCacheManager.h"

#include <algorithm>

#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

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
        auto& keys = malloc_info.batch_kv_cache_resource->batch_resource[i].cache_keys;
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
    }

    return allocator_->malloc(malloc_info);
}

FreeResult KVCacheManager::free(const FreeInfo& free_info) {
    return allocator_->free(free_info);
}

InsertResult KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    return allocator_->insertIntoCache(insert_info);
}

}  // namespace rtp_llm
