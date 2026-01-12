#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
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

    void addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    // 返回 <当前层数, 请求的层缓存列表>
    std::pair<int, std::vector<std::shared_ptr<LayerCacheBuffer>>> getBuffers(const std::set<int>& layer_ids);

    void waitChange(int last_layer_num, int timeout_ms);

    int64_t deadlineMs() const {
        return deadline_ms_;
    }

private:
    int64_t                                          request_id_;
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    int64_t                                          deadline_ms_;

    std::mutex              mutex_;
    std::condition_variable condition_variable_;
};

class ComputedLayerCacheBufferStore {
public:
    ComputedLayerCacheBufferStore();
    ~ComputedLayerCacheBufferStore();

public:
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