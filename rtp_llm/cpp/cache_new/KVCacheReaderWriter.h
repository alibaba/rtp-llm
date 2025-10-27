#pragma once

#include <vector>

#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

class ReadCallBack {
public:
    virtual void onSuccess() = 0;
    virtual void onFailure() = 0;
};

struct WriteCallback {
public:
    virtual void onSuccess() = 0;
    virtual void onFailure() = 0;
};

struct MatchResult {
    std::vector<int64_t> cache_keys;
    std::vector<int> block_ids;
};

class KVCacheReaderWriter {
public:
    KVCacheReaderWriter() = default;
    virtual ~KVCacheReaderWriter() = default;

public:
    virtual bool init() = 0;
    virtual void read(const std::shared_ptr<GenerateStream> &stream, const std::shared_ptr<ReadCallBack& callback) = 0;
    virtual void readByLayer(const std::shared_ptr<GenerateStream> &stream, int layer_id, const std::shared_ptr<ReadCallBack& callback) = 0;
    virtual void write(const std::shared_ptr<GenerateStream> &stream, const std::shared_ptr<WriteCallBack& callback) = 0;
    virtual void writeByLayer(const std::shared_ptr<GenerateStream> &stream, int layer_id, const std::shared_ptr<WriteCallBack& callback) = 0;
    virtual MatchResult match(const std::shared_ptr<GenerateStream> &stream) = 0;
    virtual MatchResult prefixMatch(const std::shared_ptr<GenerateStream> &stream) = 0;

private:
    std::shared_ptr<KVCacheConnector> connector_;
    std::vector<std::shared_ptr<KVCacheGroup>> groups_;
};

// for memory cache
class MemoryKVCacheReaderWriter : public KVCacheReaderWriter {};

// for pd sep
class P2PKVCacheReaderWriter : public KVCacheReaderWriter {};

// for remote cache
class RemoteKVCacheReaderWriter : public KVCacheReaderWriter {};

// for block meta data, eg: first token, loss etc
class BlockMetaDataReaderWriter : public KVCacheReaderWriter {};

}  // namespace rtp_llm