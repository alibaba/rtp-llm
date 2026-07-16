#pragma once

#include "autil/Thread.h"
#include <mutex>
#include <thread>
#include <map>
#include <vector>
#include <memory>
#include <condition_variable>

namespace rtp_llm {

class LayerCacheBuffer {
public:
    LayerCacheBuffer(int layer_id);
    ~LayerCacheBuffer() = default;

public:
    /// @brief 记录 cache_key 对应的 block_id
    void addBlockId(int64_t cache_key, int block_id);
    /// @brief 查询 cache_key 对应的 block_id，未找到返回 -1
    int getBlockId(int64_t cache_key) const;
    int getLayerId() const {
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
    /// timeout_ms 建议与 CacheStoreConfig::p2p_layer_cache_buffer_store_timeout_ms 一致
    explicit LayerCacheBufferStore(uint64_t timeout_ms);
    ~LayerCacheBufferStore() = default;

public:
    /// @brief 按 unique_key 存入指定层的缓冲区
    void addLayerCacheBuffer(const std::string&                       unique_key,
                             const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer);
    /// @brief 获取 unique_key 对应指定层的缓冲区，不存在返回 nullptr
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
