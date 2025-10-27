#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache_new/MemoryBlockPool.h"

namespace rtp_llm {

struct ConnectorBuffer {
    std::vector<BufferPtr> buffers;
    std::vector<int64_t> keys;
    std::vector<float> losses;
};

struct ConnectorMeta {
    int64_t request_id;
    std::vector<std::shared_ptr<KVCacheGroup>> groups;
};

struct ConnectorMatchInfo {
    std::vector<int64_t> keys;
    bool need_loss{false};
};

class ConnectorCallback {
public:
    void onSuccess() {}
    void onFailure() {}
};

class KVCacheConnector {
public:
    KVCacheConnector() = default;
    virtual ~KVCacheConnector() = default;

public:
    virtual bool init() = 0;
    virtual void asyncPut(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const std::shared_ptr<ConnectorCallback> &callback) = 0;
    virtual void asyncPrefixPut(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const std::shared_ptr<ConnectorCallback> &callback) = 0;
    virtual void asyncGet(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const std::shared_ptr<ConnectorCallback> &callback) = 0;
    virtual void asyncPrefixGet(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const std::shared_ptr<ConnectorCallback> &callback) = 0;
    virtual void asyncMatch(const ConnectorMatchInfo &match_info, const ConnectorMeta &meta, const std::shared_ptr<ConnectorCallback> &callback) = 0;
    virtual void asyncPrefixMatch(const ConnectorMatchInfo &match_info, const ConnectorMeta &meta, const std::shared_ptr<ConnectorCallback> &callback) = 0;
};

// for memory cache
class MemoryKVCacheConnector : public KVCacheConnector {
public:
    MemoryKVCacheConnector() = default;
    ~MemoryKVCacheConnector() override = default;

public:
    std::vector<std::shared_ptr<MemoryBlockPool>> mem_pools;
};

// for pd sep
class P2PKVCacheConnector : public KVCacheConnector {};

// for remote cache
class RemoteKVCacheConnector : public KVCacheConnector {};

}