#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/P2PConnectorWorkerPrefill.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/P2PConnectorWorkerDecode.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/P2PConnectorMetrics.h"
#include <c10/core/Event.h>
#include <optional>
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm::legacy::p2p {

/// @brief Facade: 统一外部接口，内部委托给 Prefill / Decode 子对象
class P2PConnectorWorker {
public:
    P2PConnectorWorker(P2PConnectorWorkerConfig                    config,
                       const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                       const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnectorWorker();

public:
    bool init(int64_t store_wait_timeout_ms = 10 * 1000);

public:
    bool
    writeByLayer(int layer_id, const KVCacheResourcePtr& resource, int64_t request_id, std::optional<c10::Event> event);

    ErrorInfo sendKVCache(int64_t                                              request_id,
                          const std::string&                                   unique_key,
                          int64_t                                              deadline_ms,
                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);

    ErrorInfo read(int64_t                                               request_id,
                   const std::string&                                    unique_key,
                   int64_t                                               deadline_ms,
                   const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                   int                                                   remote_tp_size = 1);

    bool cancelRead(const std::string& unique_key);
    bool cancelSend(const std::string& unique_key);

public:
    std::shared_ptr<ComputedLayerCacheBufferStore> getComputedBuffersStore() const;

    void setStoreWaitTimeoutMs(int64_t store_wait_timeout_ms);

private:
    P2PConnectorWorkerConfig             config_;
    std::shared_ptr<LayerBlockConverter> layer_block_converter_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;

    std::unique_ptr<P2PConnectorWorkerPrefill> prefill_;
    std::unique_ptr<P2PConnectorWorkerDecode>  decode_;
};

}  // namespace rtp_llm::legacy::p2p
