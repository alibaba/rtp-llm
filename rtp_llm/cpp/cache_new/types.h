#pragma once

#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

typedef int32_t          GroupIdType;
typedef std::vector<int> LayerIdsType;

constexpr int32_t NULL_BLOCK_IDX = -1;

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

struct BlockAddrInfo {
    void* k_addr = nullptr;
    void* v_addr = nullptr;
};

struct BlockBufferPtrInfo {
    BufferPtr k_addr;
    BufferPtr v_addr;
};

struct CacheLayerLayout {
    std::vector<int>       layer_to_groups;
    std::vector<BufferPtr> layers_to_buffer_ptrs;
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
    rtp_llm::BufferPtr k_blocks = nullptr;
    rtp_llm::BufferPtr v_blocks = nullptr;
};

// TODO, 和KVCacheBuffer类似的有好几个结构体，需要简化。

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
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id = 0;
    bool                    verbose    = true;  // for failed log

    // For common/extra blocks allocation strategy
    int common_seq_len = -1;  // -1 means no distinction between common and extra, // TODO, move to complete_token_ids ?
    int total_seq_len  = -1;  // -1 means use complete_token_ids->seqLength(), // TODO, fix this
};

struct MallocResult {
    bool success;
    int  reuse_len;  // TODO, move reuse len to batch_kv_cache_resource ？
};

struct FreeInfo {
    FreeInfo(BatchKVCacheResourcePtr batch_kv_cache_resource,
             CompleteTokenIdsPtr     complete_token_ids,
             bool                    reuse_cache         = false,
             bool                    enable_memory_cache = false,
             bool                    enable_remote_cache = false,
             bool                    enable_device_cache = true,
             bool                    sync_wait_write     = false):
        batch_kv_cache_resource(batch_kv_cache_resource),
        complete_token_ids(complete_token_ids),
        reuse_cache(reuse_cache),
        enable_memory_cache(enable_memory_cache),
        enable_remote_cache(enable_remote_cache),
        enable_device_cache(enable_device_cache),
        sync_wait_write(sync_wait_write) {}
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id{0};
    bool                    reuse_cache{false};
};

struct InsertInfo {
    InsertInfo(int64_t                 request_id,
               BatchKVCacheResourcePtr batch_kv_cache_resource,
               CompleteTokenIdsPtr     complete_token_ids,
               bool                    is_resident,
               bool                    enable_device_cache = true,
               bool                    enable_memory_cache = false,
               bool                    enable_remote_cache = false,
               bool                    sync_wait_write     = false):
        request_id(request_id),
        batch_kv_cache_resource(batch_kv_cache_resource),
        complete_token_ids(complete_token_ids),
        is_resident(is_resident),
        enable_device_cache(enable_device_cache),
        enable_memory_cache(enable_memory_cache),
        enable_remote_cache(enable_remote_cache),
        sync_wait_write(sync_wait_write) {}

    int64_t                 request_id;
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    bool                    is_resident{false};
};

}  // namespace rtp_llm