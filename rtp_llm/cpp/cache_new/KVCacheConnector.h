#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache_new/MemoryBlockPool.h"

namespace rtp_llm {

struct ConnectorBuffer {
    std::vector<BufferPtr> buffers;  // 一个buffer对应一个block
    std::vector<int64_t> keys;
    std::vector<float> losses;
};

struct ConnectorMeta {
    int layer_idx;
    int64_t request_id;
    std::vector<std::shared_ptr<KVCacheGroup>> groups;
};

struct ConnectorMatchInfo {
    std::vector<int64_t> keys;
    bool need_loss{false};
};

using ConnectorCallBack = std::function<void(bool success)>;
using ConnectorMatchCallBack = std::function<void(const std::vector<bool> &results)>;

class KVCacheConnector {
public:
    KVCacheConnector() = default;
    virtual ~KVCacheConnector() = default;

public:
    virtual bool init() = 0;

    void asyncPut(const std::shared_ptr<BatchKVCacheResource> &resource, const ConnectorCallBack& callback) {

    }

    virtual void asyncPut(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const ConnectorCallBack &callback) = 0;
    virtual void asyncPrefixPut(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const ConnectorCallBack &callback) = 0;

    virtual void asyncGet(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const ConnectorCallBack &callback) = 0;
    virtual void asyncPrefixGet(const std::shared_ptr<std::vector<ConnectorBuffer>> &buffers, const ConnectorMeta &meta, const ConnectorCallBack &callback) = 0;

    virtual void asyncMatch(const ConnectorMatchInfo &match_info, const ConnectorCallBack &callback) = 0;
    virtual void asyncPrefixMatch(const ConnectorMatchInfo &match_info, const ConnectorCallBack &callback) = 0;
};

// for memory cache
class MemoryKVCacheConnector : public KVCacheConnector {
public:
    MemoryKVCacheConnector() = default;
    ~MemoryKVCacheConnector() override = default;

private:
    void asyncMatch(const ConnectorMatchInfo &match_info, const ConnectorCallBack &callback) override {
        std::vector<bool> results;
        for (auto &cache_key : match_info.keys) {
            bool result = matchOne(cache_key);
            results.push_back(result);
        }
        callback(results);
    }

private:
    bool matchOne(int64_t cache_key) const {
        for (auto &mem_cache : mem_caches_) {
            if (!mem_cache->match(cache_key)) {
                return false;
            }
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<MemoryBlockPool>> mem_pools_;
    std::vector<std::shared_ptr<MemoryBlockCache>> mem_caches_;
};

// for pd sep
class P2PKVCacheConnector : public KVCacheConnector {};

// for remote cache
class RemoteKVCacheConnector : public KVCacheConnector {};

}