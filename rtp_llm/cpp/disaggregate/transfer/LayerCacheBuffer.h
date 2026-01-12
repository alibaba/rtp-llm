#pragma once

#include "autil/Thread.h"
#include <mutex>
#include <thread>
#include <map>
#include <vector>
#include <memory>
#include <condition_variable>

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class LayerCacheBuffer {
public:
    LayerCacheBuffer(int layer_id);
    ~LayerCacheBuffer() = default;

public:
    void addBlockId(int64_t cache_key, int block_id);
    int  getBlockId(int64_t cache_key) const;
    int  getLayerId() const {
        return layer_id_;
    }
    const std::map<int64_t, int>& blockIdMap() const {
        return block_id_map_;
    }

private:
    int                    layer_id_;
    std::map<int64_t, int> block_id_map_;  // [cache_key, block_id]
};

class LayerCacheBufferStore {
public:
    LayerCacheBufferStore(uint64_t timeout_ms = 100 * 1000);
    ~LayerCacheBufferStore() = default;

public:
    void                              addLayerCacheBuffer(const std::string&                       unique_key,
                                                          const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer);
    std::shared_ptr<LayerCacheBuffer> getLayerCacheBuffer(const std::string& unique_key, int layer_id) const;
    void                              checkTimeout();

private:
    uint64_t timeout_ms_;

    mutable std::mutex mutex_;
    // [unique_key, [layer_id, LayerCacheBuffer]]
    std::map<std::string, std::map<int, std::shared_ptr<LayerCacheBuffer>>> layer_cache_buffer_map_;
    // [unique_key, expired_time]
    std::map<std::string, int64_t> expired_time_map_;
};
}  // namespace rtp_llm
