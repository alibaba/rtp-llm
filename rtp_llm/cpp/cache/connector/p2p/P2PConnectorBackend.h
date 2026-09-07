#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <memory>

namespace rtp_llm::p2p_internal {

transfer::TransferBackendPair createAndRegisterTransferBackend(
    const P2PConnectorWorkerConfig&             config,
    const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
    const kmonitor::MetricsReporterPtr&         metrics_reporter);

}  // namespace rtp_llm::p2p_internal
