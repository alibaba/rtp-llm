#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    struct Buffer {
        BufferPtr            buffer;
        int64_t              key;
        int32_t              layer;
        std::optional<float> loss;
    } using Buffers = std::vector<Buffer>;

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

    virtual bool    match(int64_t key)                            = 0;
    virtual int32_t prefixMatch(const std::vector<int64_t>& keys) = 0;
};

}  // namespace rtp_llm