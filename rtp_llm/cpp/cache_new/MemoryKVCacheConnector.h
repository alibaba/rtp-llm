#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <optional>

#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

// A simple in-memory KV cache connector. Thread-safe for concurrent reads/writes.
class MemoryKVCacheConnector final: public KVCacheConnector {
public:
    MemoryKVCacheConnector(BlockPoolPtr block_pool, BlockCachePtr block_cache, rtp_llm::DeviceBase* device);
    ~MemoryKVCacheConnector() override;

public:
    bool init() override;

    void asyncPut(const Buffers& buffers, const Meta& meta, const CallBack& callback) override;
    void asyncPrefixPut(const Buffers& buffers, const Meta& meta, const CallBack& callback) override;

    void asyncGet(const Buffers& buffers, const Meta& meta, const CallBack& callback) override;
    void asyncPrefixGet(const Buffers& buffers, const Meta& meta, const CallBack& callback) override;

    std::vector<bool> match(const std::vector<int64_t>& keys) override;
    int32_t           prefixMatch(const std::vector<int64_t>& keys) override;

private:
    bool copyBufferData(const BufferPtr& dst, const BufferPtr& src);

private:
    BlockPoolPtr         block_pool_;
    BlockCachePtr        block_cache_;
    rtp_llm::DeviceBase* device_;
};

}  // namespace rtp_llm
