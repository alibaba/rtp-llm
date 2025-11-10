#pragma once

#include "autil/Thread.h"
#include <mutex>
#include <thread>
#include <map>
#include <vector>
#include <memory>

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {
namespace cache_store {

struct BlockCacheBuffer {
    int64_t key;
    // not sure how to divide buffer according to layer type, will call allocator
    int block_id;

    BufferPtr buffer1;
    BufferPtr buffer2;

    BlockCacheBuffer(int64_t key, int block_id): key(key), block_id(block_id) {}
    BlockCacheBuffer(int64_t key, int block_id, BufferPtr buffer1, BufferPtr buffer2):
        key(key), block_id(block_id), buffer1(buffer1), buffer2(buffer2) {}
};

class LayerCacheBuffer {
public:
    LayerCacheBuffer(int layer_id);
    ~LayerCacheBuffer() = default;

public:
    void addBlockCacheBuffer(int64_t key, int block_id);
    void addBlockCacheBuffer(int64_t key, int block_id, BufferPtr buffer1, BufferPtr buffer2);
    std::shared_ptr<BlockCacheBuffer> getBlockCacheBuffer(int64_t key) const;
    int                               layerId() const {
        return layer_id_;
    }
    const std::map<int64_t, std::shared_ptr<BlockCacheBuffer>>& blockCacheBuffers() const {
        return block_cache_buffers_;
    }

private:
    int                                                  layer_id_;
    std::map<int64_t, std::shared_ptr<BlockCacheBuffer>> block_cache_buffers_;
};

class SingleLayerCacheBufferStore {
public:
    SingleLayerCacheBufferStore(int layer_id);
    ~SingleLayerCacheBufferStore() = default;

public:
    class Watcher {
    public:
        Watcher(int layer_id): layer_id_(layer_id) {}
        ~Watcher() = default;

    public:
        virtual bool notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) = 0;
        int          layerId() const {
            return layer_id_;
        }

    protected:
        int layer_id_;
    };

public:
    bool setLayerCacheBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    void setLayerCacheBufferWatchFunc(std::shared_ptr<Watcher> watcher, int64_t deadline_ms);

    void checkTimeout();

    uint32_t layerCacheBufferMapSize() const {
        return layer_cache_buffer_map_.size();
    }

    uint32_t layerCacheBufferWatcherMapSize() const {
        return layer_cache_buffer_watcher_map_.size();
    }

private:
    void notifyAllWatchers(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer);
    void watchAllCacheBuffers(const std::shared_ptr<Watcher>& watcher);
    void delLayerCacheBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer);
    void delLayerCacheBufferWatchFunc(std::shared_ptr<Watcher> watcher);

private:
    int layer_id_;

    std::mutex layer_cache_buffers_mutex_;
    // [layer_buffer, deadline_ms]
    std::map<std::shared_ptr<LayerCacheBuffer>, int64_t> layer_cache_buffer_map_;

    // [layer_id][watcher, timeout_ms]
    std::mutex layer_cache_buffer_watchers_mutex_;
    // [watcher, deadline_ms]
    std::map<std::shared_ptr<Watcher>, int64_t> layer_cache_buffer_watcher_map_;
};

class LayerCacheBufferStore {
public:
    LayerCacheBufferStore(int layer_num);
    ~LayerCacheBufferStore() {
        check_timeout_thread_stop_ = true;
        if (check_timeout_thread_.joinable()) {
            check_timeout_thread_.join();
        }
    }

public:
    std::shared_ptr<SingleLayerCacheBufferStore> getSingleLayerCacheBufferStore(int layer_id) const;

private:
    void checkTimeoutThread();

private:
    int                                                       layer_num_;
    std::vector<std::shared_ptr<SingleLayerCacheBufferStore>> single_layer_cache_buffer_stores_;
    bool                                                      check_timeout_thread_stop_{false};
    std::thread                                               check_timeout_thread_;
};

}  // namespace cache_store
}  // namespace rtp_llm