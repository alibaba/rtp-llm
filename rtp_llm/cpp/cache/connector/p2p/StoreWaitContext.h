#pragma once

#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "autil/LoopThread.h"
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace rtp_llm {

struct StoreWaitContext {
    int64_t                                             request_id;
    AsyncEventPtr                                       event;
    std::shared_ptr<LayerCacheBuffer>                   layer_cache_buffer;
    int64_t                                             deadline_ms;
    std::shared_ptr<PrefillWorkerStoreMetricsCollector> collector;

    StoreWaitContext(int64_t                                             request_id,
                     AsyncEventPtr                                       event,
                     std::shared_ptr<LayerCacheBuffer>                   layer_cache_buffer,
                     int64_t                                             deadline_ms,
                     std::shared_ptr<PrefillWorkerStoreMetricsCollector> collector):
        request_id(request_id),
        event(event),
        layer_cache_buffer(layer_cache_buffer),
        deadline_ms(deadline_ms),
        collector(collector) {}
};

class StoreWaitContextChecker {
public:
    StoreWaitContextChecker(const kmonitor::MetricsReporterPtr&                   metrics_reporter,
                            const std::shared_ptr<ComputedLayerCacheBufferStore>& computed_buffers);
    ~StoreWaitContextChecker();

public:
    /// @brief 添加一个待等待 GPU event 的 store 上下文
    void addContext(const StoreWaitContext& context);

    size_t getContextCount() const;

    /// @brief 检查所有上下文，将 GPU event 已完成的写入 computed buffer
    void checkOnce();

private:
    kmonitor::MetricsReporterPtr                   metrics_reporter_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;

    mutable std::mutex            contexts_mutex_;
    std::vector<StoreWaitContext> contexts_;
};

}  // namespace rtp_llm
