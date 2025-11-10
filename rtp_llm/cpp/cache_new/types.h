#pragma once

#include <vector>
#include <cstdint>
#include <map>

#include <torch/torch.h>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"

namespace rtp_llm {

class CompleteTokenIds;
using CompleteTokenIdsPtr = std::shared_ptr<CompleteTokenIds>;

class BatchKVCacheResource;
using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

typedef int32_t               GroupIdType;
typedef std::vector<float>    LossType;
typedef std::vector<LossType> LossesType;
typedef std::vector<int>      LayerIdsType;

constexpr int32_t NULL_BLOCK_IDX = -1;

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

// TODO(chanyin): remove k_scale_addr and v_scale_addr in future
struct BlockAddrInfo {
    void* k_addr       = nullptr;
    void* v_addr       = nullptr;
    void* k_scale_addr = nullptr;
    void* v_scale_addr = nullptr;
};

struct BlockBufferInfo {
    BufferPtr k_addr;
    BufferPtr v_addr;
    // BufferPtr k_scale_addr;
    // BufferPtr v_scale_addr;
};

struct CacheLayerLayout {
    std::vector<int>           layer_to_groups;
    std::vector<torch::Tensor> layers_to_buffer_ptrs;
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
    rtp_llm::BufferPtr k_blocks;
    rtp_llm::BufferPtr v_blocks;
    rtp_llm::BufferPtr k_scale;
    rtp_llm::BufferPtr v_scale;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

struct MatchResult {
    size_t           reuse_length = 0;
    size_t           reuse_blocks = 0;
    BlockIndicesType block_indices;
};

struct MallocInfo {
    MallocInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}

    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    BufferPtr               loss       = nullptr;
    int64_t                 request_id = 0;  // for logging and debugging
    bool                    verbose    = true;

    // For common/extra blocks allocation strategy
    int common_seq_len = -1;  // -1 means no distinction between common and extra
    int total_seq_len  = -1;  // -1 means use complete_token_ids->seqLength()
};

struct MallocResult {
    bool success;
    int  reuse_len;
};

struct FreeInfo {
    FreeInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}

    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;

    // Metadata
    int64_t request_id = 0;  // for logging and debugging
};

struct FreeResult {
    bool success;
};

struct InsertInfo {
    InsertInfo(BatchKVCacheResourcePtr batch_kv_cache_resource,
               CompleteTokenIdsPtr     complete_token_ids,
               bool                    is_resident):
        batch_kv_cache_resource(batch_kv_cache_resource),
        complete_token_ids(complete_token_ids),
        is_resident(is_resident) {}

    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    bool                    is_resident;
};

struct InsertResult {
    bool success;
};

}  // namespace rtp_llm