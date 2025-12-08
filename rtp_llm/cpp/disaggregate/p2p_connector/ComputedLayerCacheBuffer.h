#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"

namespace rtp_llm {

struct ComputedLayerCacheBuffer {
    int64_t                                          request_id;
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    int64_t                                          deadline_ms;

    ComputedLayerCacheBuffer(int64_t                                  request_id,
                             const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                             int64_t                                  deadline_ms):
        request_id(request_id),
        layer_cache_buffers({{layer_cache_buffer->getLayerId(), layer_cache_buffer}}),
        deadline_ms(deadline_ms) {}
};

class ComputedLayerCacheBufferStore {
public:
    ComputedLayerCacheBufferStore();
    ~ComputedLayerCacheBufferStore();

public:
    void
    addBuffer(int64_t request_id, const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);
    std::shared_ptr<ComputedLayerCacheBuffer> getBuffer(int64_t request_id) const;
    void                                      removeBuffer(int64_t request_id);
    void                                      checkTimeout();

private:
    // stores layer cache buffer already computed
    mutable std::mutex                                                     computed_buffers_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<ComputedLayerCacheBuffer>> computed_buffers_;
};

}  // namespace rtp_llm