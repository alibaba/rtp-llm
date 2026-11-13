#pragma once

#include "autil/Thread.h"
#include <cstddef>
#include <mutex>
#include <thread>
#include <map>
#include <tuple>
#include <vector>
#include <memory>
#include <condition_variable>
#include <string>

namespace rtp_llm {

// Publication identity: owning module + module-local layer + globally unique tag.
// Two modules may both own local layer 0, so the module id is part of the key.
struct LayerCacheDescriptor {
    size_t      module_id = 0;
    int         layer_id  = 0;
    std::string cache_tag;

    bool operator<(const LayerCacheDescriptor& rhs) const {
        return std::tie(module_id, layer_id, cache_tag) < std::tie(rhs.module_id, rhs.layer_id, rhs.cache_tag);
    }
    bool operator==(const LayerCacheDescriptor& rhs) const {
        return module_id == rhs.module_id && layer_id == rhs.layer_id && cache_tag == rhs.cache_tag;
    }
};

class LayerCacheBuffer {
public:
    LayerCacheBuffer(int layer_id, std::string cache_tag, size_t module_id = 0);
    ~LayerCacheBuffer() = default;

public:
    /// @brief 记录 cache_key 对应的 block_id
    void addBlockId(int64_t cache_key, int block_id);
    /// @brief 查询 cache_key 对应的 block_id，未找到返回 -1
    int getBlockId(int64_t cache_key) const;
    int getLayerId() const {
        return layer_id_;
    }
    const std::string& cacheTag() const {
        return cache_tag_;
    }
    size_t moduleId() const {
        return module_id_;
    }
    LayerCacheDescriptor descriptor() const {
        return {module_id_, layer_id_, cache_tag_};
    }
    const std::map<int64_t, int>& blockIdMap() const {
        return block_id_map_;
    }

private:
    int                    layer_id_;
    std::string            cache_tag_;
    size_t                 module_id_;
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
    std::shared_ptr<LayerCacheBuffer> getLayerCacheBuffer(const std::string& unique_key,
                                                          int                layer_id,
                                                          const std::string& cache_tag,
                                                          size_t             module_id = 0) const;
    void                              checkTimeout();

private:
    uint64_t timeout_ms_;

    mutable std::mutex mutex_;
    // [unique_key, [(module_id, layer_id, tag), LayerCacheBuffer]]
    std::map<std::string, std::map<LayerCacheDescriptor, std::shared_ptr<LayerCacheBuffer>>> layer_cache_buffer_map_;
    // [unique_key, expired_time]
    std::map<std::string, int64_t> expired_time_map_;
};
}  // namespace rtp_llm
