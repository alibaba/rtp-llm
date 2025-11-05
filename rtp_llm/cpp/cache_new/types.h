#pragma once

#include <vector>
#include <cstdint>
#include <map>

#include <torch/torch.h>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

typedef size_t  CacheKeyType;
typedef int32_t BlockIdxType;

typedef std::vector<CacheKeyType> CacheKeysType;
typedef std::vector<BlockIdxType> BlockIndicesType;
typedef std::vector<float>        LossType;
typedef std::vector<LossType>     LossesType;
typedef std::vector<int>          LayerIdsType;

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
    size_t               available_kv_cache = 0;
    size_t               total_kv_cache     = 0;
    size_t               block_size         = 0;
    std::vector<int64_t> cached_keys;
    int64_t              version = -1;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

struct MatchResult {
    size_t           reuse_length;
    BlockIndicesType block_indices;
};

struct MallocInfo {
    MallocInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}

    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    BufferPtr               loss = nullptr;
};

struct MallocResult {
    bool               success;
    int                reuse_len;
    std::vector<float> loss;  // TODO(chanyin): remove this in future
};

// fallback
struct FreeInfo {
    FreeInfo(BatchKVCacheResourcePtr batch_kv_cache_resource, CompleteTokenIdsPtr complete_token_ids):
        batch_kv_cache_resource(batch_kv_cache_resource), complete_token_ids(complete_token_ids) {}
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
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