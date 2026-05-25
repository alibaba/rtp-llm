#pragma once

#include "autil/Thread.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include <mutex>
#include <thread>
#include <map>
#include <vector>
#include <memory>
#include <condition_variable>

namespace rtp_llm {

class LayerCacheBuffer {
public:
    LayerCacheBuffer(int layer_id, KVCacheRegionName region_name = KVCacheRegionName::DEFAULT);
    ~LayerCacheBuffer() = default;

public:
    void addBlockId(int64_t cache_key, int block_id);
    int  getBlockId(int64_t cache_key) const;
    int  getLayerId() const {
        return layer_id_;
    }
    KVCacheRegionName getRegionName() const {
        return region_name_;
    }
    int virtualLayerId() const {
        return layer_id_ * static_cast<int>(KVCacheRegionName::REGION_COUNT) + static_cast<int>(region_name_);
    }
    const std::map<int64_t, int>& blockIdMap() const {
        return block_id_map_;
    }

private:
    int               layer_id_;
    KVCacheRegionName region_name_;
    std::map<int64_t, int> block_id_map_;
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
