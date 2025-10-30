#pragma once

#include "rtp_llm/cpp/engine_base/resource/BatchKVCacheResource.h"

namespace rtp_llm {

class KVCacheReaderWriter {
public:
    KVCacheReaderWriter()          = default;
    virtual ~KVCacheReaderWriter() = default;

public:
    using CallBack = std::function<void(bool)>;

public:
    virtual bool init();

    virtual void asyncRead(const BatchKVCacheResourcePtr& resource, const CallBack& callback)                      = 0;
    virtual void asyncReadByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback) = 0;

    virtual void asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback)                      = 0;
    virtual void asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, const CallBack& callback) = 0;

    // virtual bool match(int64_t key) = 0;
    // virtual int32_t prefixMatch(const std::vector<int64_t> &keys) = 0;
};

}  // namespace rtp_llm