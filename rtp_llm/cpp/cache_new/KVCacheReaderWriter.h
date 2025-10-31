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
    virtual bool   init()                                                                        = 0;
    virtual void   asyncRead(const BatchKVCacheResourcePtr& resource, const CallBack& callback)  = 0;
    virtual void   asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback) = 0;
    virtual size_t match(const std::vector<int64_t>& keys) const;
};

}  // namespace rtp_llm