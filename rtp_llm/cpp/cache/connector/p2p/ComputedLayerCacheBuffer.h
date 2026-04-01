#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

class ComputedLayerCacheBuffer {
public:
    ComputedLayerCacheBuffer(int64_t                                  request_id,
                             const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                             int64_t                                  deadline_ms);

    /// @brief 追加一层 cache buffer 并更新 deadline
    void addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    /// @brief 返回当前已有层数及指定层集合对应的缓冲区列表
    std::pair<int, std::vector<std::shared_ptr<LayerCacheBuffer>>> getBuffers(const std::set<int>& layer_ids);

    /// @brief 阻塞等待层数变化，直到超过 last_layer_num 或 timeout_ms 超时
    void waitChange(int last_layer_num, int timeout_ms);

    int64_t deadlineMs() const {
        return deadline_ms_.load(std::memory_order_relaxed);
    }

private:
    int64_t                                          request_id_;
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    std::atomic<int64_t>                             deadline_ms_;

    std::mutex              mutex_;
    std::condition_variable condition_variable_;
};

class ComputedLayerCacheBufferStore {
public:
    ComputedLayerCacheBufferStore();
    ~ComputedLayerCacheBufferStore();

public:
    /// @brief 按 request_id 获取或创建对应的 ComputedLayerCacheBuffer 并追加首层数据
    std::shared_ptr<ComputedLayerCacheBuffer>
    addBuffer(int64_t request_id, const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    std::shared_ptr<ComputedLayerCacheBuffer> getBuffer(int64_t request_id) const;
    void                                      removeBuffer(int64_t request_id);
    void                                      checkTimeout();
    int64_t                                   getBuffersCount() const;

private:
    // stores layer cache buffer already computed
    mutable std::mutex                                                     computed_buffers_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<ComputedLayerCacheBuffer>> computed_buffers_;
};

}  // namespace rtp_llm