#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    // 方式1:
    struct GroupBuffer {
        int32_t group_id;
        // layer 
        std::vector<BufferPtr>            buffers;  // all layer, k+v for single layer
        std::vector<string> layer_ids;       // cache key
        // string layout_info;
    };
    struct CacheKeyBuffer { 
        int64_t              key;
        string match_info;
        std::vector<GroupBuffer> group_buffers;
    };

    // 方式2:
    // 远端cache是一个kkv的结构
    struct Buffer {
        int32_t              key1;       // cache key
        int64_t              key2;       // group id
        // int32_t              layer_idx;  // for get buffer from memory pool
        std::vector<BufferPtr>            buffers;  // all layer kvs, k+v for single layer
        std::vector<string> layer_ids;
        // std::optional<float> loss;
        string match_info;
    };
    using Buffers = std::vector<Buffer>;

    struct Meta {
        int64_t request_id;
        int32_t rank;
        // tokens
        // lora
    };

    struct MatchResult {
        int64_t cache_key;
        bool matched;
        string match_info; // for remote 
    }

    using CallBack = std::function<void(bool success)>;

public:
    virtual bool init() = 0;

    virtual void asyncPut(const Buffers& buffers, const Meta& meta, const CallBack& callback)       = 0;
    virtual void asyncPrefixPut(const Buffers& buffers, const Meta& meta, const CallBack& callback) = 0;

    virtual void asyncGet(const Buffers& buffers, const Meta& meta, const CallBack& callback)       = 0;
    virtual void asyncPrefixGet(const Buffers& buffers, const Meta& meta, const CallBack& callback) = 0;

    virtual MatchResult match(const std::vector<int64_t>& keys, const Meta& meta)       = 0;
    virtual MatchResult prefixMatch(const std::vector<int64_t>& keys, const Meta& meta) = 0;
};

}  // namespace rtp_llm