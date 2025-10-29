#pragma once

#include <vector>

#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/engine_base/resource/BatchKVCacheResource.h"

namespace rtp_llm {

// class ReadCallBack {
// public:
//     virtual readDone(bool success) = 0;
// };

// struct WriteCallback {
// public:
//     virtual writeDone(bool success) = 0;
// };

struct MatchResult {
    std::vector<int64_t> cache_keys;
    std::vector<int> block_ids;
};

using CallBack = std::function<void(bool)>;

class KVCacheReaderWriter {
public:
    KVCacheReaderWriter() = default;
    virtual ~KVCacheReaderWriter() = default;

public:
    virtual bool init();
    virtual void asyncRead(const std::shared_ptr<BatchKVCacheResource> &resource, const CallBack &callback);
    virtual void asyncReadByLayer(const std::shared_ptr<BatchKVCacheResource> &resource, int layer_id, const CallBack &callback);
    virtual void asyncWrite(const std::shared_ptr<BatchKVCacheResource> &resource, const CallBack &callback);
    virtual void asyncWriteByLayer(const std::shared_ptr<BatchKVCacheResource> &resource, int layer_id, const CallBack &callback);
    virtual MatchResult match(const std::shared_ptr<BatchKVCacheResource> &resource);
    virtual MatchResult prefixMatch(const std::shared_ptr<BatchKVCacheResource> &resource);

private:
    // TODO(LXQ): 还是按照继承的方式实现, 不要放在一个基类中
    std::shared_ptr<KVCacheConnector> memory_connector_;
    // TODO(LXQ): 不应该感知group, 只感知layer
    std::vector<std::shared_ptr<KVCacheGroup>> groups_;
};

// // for memory cache
// class MemoryKVCacheReaderWriter : public KVCacheReaderWriter {
// public:
//     void bool match(const int64_t& cache_key) override {
//         return connector_->match(cache_key);
//     }
//     void write(const std::vector<int64_t>& cache_keys, const std::vector<int>& block_indices) override {
//         // 1. connector->match(cache_keys);
//         // 2. if not fully match, reader_writer_->write(cache_keys, block_indices);
//     }
// };

// // for pd sep
// class P2PKVCacheReaderWriter : public KVCacheReaderWriter {};

// // for remote cache
// class RemoteKVCacheReaderWriter : public KVCacheReaderWriter {};

// // for block meta data, eg: first token, loss etc
// class BlockMetaDataReaderWriter : public KVCacheReaderWriter {};

}  // namespace rtp_llm