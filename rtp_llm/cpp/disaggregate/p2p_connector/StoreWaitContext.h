#pragma once

#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "autil/LoopThread.h"
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace rtp_llm {

struct StoreWaitContext {
    int64_t                                                        request_id;
    DeviceEventPtr                                                 event;
    std::shared_ptr<LayerCacheBuffer>                              layer_cache_buffer;
    int64_t                                                        deadline_ms;
    std::shared_ptr<P2PConnectorServerWorkerStoreMetricsCollector> collector;

    StoreWaitContext(int64_t                                                        request_id,
                     DeviceEventPtr                                                 event,
                     std::shared_ptr<LayerCacheBuffer>                              layer_cache_buffer,
                     int64_t                                                        deadline_ms,
                     std::shared_ptr<P2PConnectorServerWorkerStoreMetricsCollector> collector):
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
    void addContext(const StoreWaitContext& context);

    size_t getContextCount() const;

    void checkOnce();

private:
    kmonitor::MetricsReporterPtr                   metrics_reporter_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;

    mutable std::mutex            contexts_mutex_;
    std::vector<StoreWaitContext> contexts_;
};

}  // namespace rtp_llm