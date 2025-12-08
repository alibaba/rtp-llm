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

class LayerCacheBufferTask {
public:
    LayerCacheBufferTask(const std::map<int, std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                         int64_t                                                 deadline_ms):
        layer_cache_buffers_(layer_cache_buffers), deadline_ms_(deadline_ms) {}
    ~LayerCacheBufferTask() = default;

public:
    void setCancelled();
    // wait loading task done, may still have inflight request
    void waitDone();
    void notifyDone(int layer_id, bool success);

    // wait all inflight request finished
    void setLoading(int layer_id);

    // 在调用waitLoadingDone之前，需要保证不会有新的loading request进来
    // TODO: test this
    void waitLoadingDone();

    bool success() const;
    bool cancelled() const;

    std::shared_ptr<LayerCacheBuffer> getLayerCacheBuffer(int layer_id) const;

private:
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    std::set<int>                                    done_layer_ids_;
    std::set<int>                                    loading_layer_ids_;
    bool                                             all_success_ = true;
    bool                                             cancelled_   = false;
    mutable std::mutex                               mutex_;
    mutable std::condition_variable                  cond_;
    int64_t                                          deadline_ms_;
};

/// @brief 存储 unique_key 对应的多层的 LayerCacheBuffer
/// 这里的 LayerCacheBuffer 中存储的只有 BlockID，需要配合 request 中的 partition_count 和 partition_id
/// 调用 LayerBlockConvertor 的接口获取对应的 Buffer
class LayerCacheBufferTaskStore {
public:
    LayerCacheBufferTaskStore();
    ~LayerCacheBufferTaskStore() = default;

    std::shared_ptr<LayerCacheBufferTask>
    addTask(const std::string&                                      unique_key,
            const std::map<int, std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
            int64_t                                                 deadline_ms);

    std::shared_ptr<LayerCacheBufferTask> getTask(const std::string& unique_key) const;
    std::shared_ptr<LayerCacheBufferTask> stealTask(const std::string& unique_key);

private:
    mutable std::mutex mutex_;
    // [unique_key, LayerCacheBufferTask]
    std::map<std::string, std::shared_ptr<LayerCacheBufferTask>> task_map_;
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
