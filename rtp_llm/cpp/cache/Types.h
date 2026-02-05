#pragma once

#include <vector>
#include <cstdint>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

typedef int32_t          GroupIdType;
typedef std::vector<int> LayerIdsType;

constexpr int32_t NULL_BLOCK_IDX = -1;

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

struct BlockAddrInfo {
    void* kv_addr       = nullptr;
    void* kv_scale_addr = nullptr;
};

struct BlockBufferPtrInfo {
    BufferPtr kv_addr       = nullptr;
    BufferPtr kv_scale_addr = nullptr;
};

struct CacheLayerLayout {
    std::vector<int>       layer_to_groups;
    std::vector<BufferPtr> layers_to_buffer_ptrs;
    std::vector<BufferPtr> layers_to_scale_buffer_ptrs;
};

struct KVCacheInfo {
    size_t                    available_kv_cache = 0;
    size_t                    total_kv_cache     = 0;
    size_t                    block_size         = 0;
    std::vector<CacheKeyType> cached_keys;
    int64_t                   version = -1;
};

// For backward compatibility with old cache system (same as GptModel.h definition)
struct KVCacheBuffer {
    // Layout convention: [layer_num, block_num, local_head_num_kv, seq_size_per_block, hidden_size_per_head], INT8
    rtp_llm::BufferPtr kv_blocks = nullptr;
    // Layout convention: [layer_num, block_num * 2, local_head_num_kv, seq_size_per_block], FP32.
    rtp_llm::BufferPtr kv_scale_blocks = nullptr;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

struct MatchResult {
    size_t           reuse_length = 0;
    size_t           reuse_blocks = 0;
    BlockIndicesType block_indices;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MatchResult reuse_length: " << reuse_length << ", reuse_blocks: " << reuse_blocks
                     << ", block_indices: ";
        for (const auto& v : block_indices) {
            debug_string << v << ", ";
        }
        return debug_string.str();
    }
};

struct MallocInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id = 0;
    int64_t                 epoch      = 0;     // Batch Epoch ID: 0 = not assigned, >0 = batch-specific
    bool                    verbose    = true;  // for failed log
};

struct MallocResult {
    bool success;
    int  reuse_len;

    int64_t match_cost_time_us = 0;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MallocResult success: " << (success ? "true" : "false") << ", reuse_len: " << reuse_len
                     << ", match_cost_time_us: " << match_cost_time_us;
        return debug_string.str();
    }
};

struct FreeInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;

    int64_t request_id = 0;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "FreeInfo request_id: " << request_id;
        if (batch_kv_cache_resource) {
            debug_string << ", batch_size: " << batch_kv_cache_resource->batchSize();
            if (batch_kv_cache_resource->batchSize() > 0) {
                debug_string << ", cache_keys: ";
                const auto& cache_keys = batch_kv_cache_resource->cacheKeys(0);
                for (size_t i = 0; i < cache_keys.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << cache_keys[i] << ", ";
                }
                if (cache_keys.size() > 10) {
                    debug_string << "...";
                }
                debug_string << ", blocks: ";
                const auto& blocks = batch_kv_cache_resource->blocks(0);
                for (size_t i = 0; i < blocks.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << blocks[i] << ", ";
                }
                if (blocks.size() > 10) {
                    debug_string << "...";
                }
            }
        }
        return debug_string.str();
    }
};

struct InsertInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    bool                    is_resident;
    int64_t                 epoch = 0;  // Epoch ID: 0 = global visible, >0 = batch-specific

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "InsertInfo is_resident: " << (is_resident ? "true" : "false");
        debug_string << ", epoch: " << epoch;
        if (batch_kv_cache_resource) {
            debug_string << ", batch_size: " << batch_kv_cache_resource->batchSize();
            if (batch_kv_cache_resource->batchSize() > 0) {
                debug_string << ", cache_keys: ";
                const auto& cache_keys = batch_kv_cache_resource->cacheKeys(0);
                for (size_t i = 0; i < cache_keys.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << cache_keys[i] << ", ";
                }
                if (cache_keys.size() > 10) {
                    debug_string << "...";
                }
                debug_string << ", blocks: ";
                const auto& blocks = batch_kv_cache_resource->blocks(0);
                for (size_t i = 0; i < blocks.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << blocks[i] << ", ";
                }
                if (blocks.size() > 10) {
                    debug_string << "...";
                }
            }
        }
        return debug_string.str();
    }
};

}  // namespace rtp_llm