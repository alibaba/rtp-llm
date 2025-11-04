#pragma once

#include "rtp_llm/cpp/engine_base/resource/BatchKVCacheResource.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    using CallBack = std::function<void(bool)>;

public:
    virtual bool   init()                                                                      = 0;
    virtual void   asyncGet(const BatchKVCacheResourcePtr& resource, const CallBack& callback) = 0;
    virtual void   asyncPut(const BatchKVCacheResourcePtr& resource, const CallBack& callback) = 0;
};

}  // namespace rtp_llm