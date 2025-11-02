#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    // 远端cache是一个kkv的结构
    struct Buffer {
        int32_t              key1;       // group id
        int64_t              key2;       // cache key
        int32_t              layer_idx;  // for get buffer from memory pool
        BufferPtr            buffer;
        std::optional<float> loss;
    };
    using Buffers = std::vector<Buffer>;

    struct Meta {
        int64_t request_id;
    };

    using CallBack = std::function<void(bool success)>;

public:
    virtual bool init() = 0;

    virtual void asyncPut(const Buffers& buffers, const Meta& meta, const CallBack& callback)       = 0;
    virtual void asyncPrefixPut(const Buffers& buffers, const Meta& meta, const CallBack& callback) = 0;

    virtual void asyncGet(const Buffers& buffers, const Meta& meta, const CallBack& callback)       = 0;
    virtual void asyncPrefixGet(const Buffers& buffers, const Meta& meta, const CallBack& callback) = 0;

    virtual std::vector<bool> match(const std::vector<int64_t>& keys)       = 0;
    virtual int32_t           prefixMatch(const std::vector<int64_t>& keys) = 0;
};

}  // namespace rtp_llm