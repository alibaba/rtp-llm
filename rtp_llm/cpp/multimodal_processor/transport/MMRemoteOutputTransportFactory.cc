#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransportFactory.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/MMTransportMode.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::unique_ptr<MMRemoteOutputTransport> createMMRemoteOutputTransport(const MMTransportConfig&     transport_config,
                                                                       kmonitor::MetricsReporterPtr reporter,
                                                                       int                          device_id) {
    const auto mode    = validateMMTransportMode(transport_config.mode);
    auto       metrics = std::make_shared<const MMTransportMetrics>(std::move(reporter));

    std::vector<std::unique_ptr<MMReceiptReader>> readers;
    if (mode == kMMTransportModeRdma) {
        // One dedicated provider per pool slot: providers serialize their own calls, so
        // a pool of N lets N multimodal reads run concurrently instead of queueing the
        // whole worker on a single serialization point. All-or-nothing initialization.
        const size_t pool_size = std::max<size_t>(1, transport_config.rdma.read_provider_count);
        std::vector<std::shared_ptr<rdma_transport::RdmaRead>> providers;
        providers.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            auto reader = rdma_transport::createRdmaRead(transport_config.rdma, device_id);
            if (reader == nullptr) {
                throw std::runtime_error("failed to initialize RDMA reader for multimodal output transport");
            }
            providers.push_back(std::move(reader));
        }
        RTP_LLM_LOG_INFO("mm transport mode=rdma: %zu pooled reader(s) initialized", pool_size);
        readers.push_back(std::make_unique<MMRdmaReader>(std::move(providers), transport_config.rdma));
    } else {
        RTP_LLM_LOG_INFO("mm transport mode=grpc: rdma disabled, use inline grpc");
        readers.push_back(createMMRdmaReader(nullptr));
    }

    return std::make_unique<MMRemoteOutputTransport>(
        std::move(readers),
        createGrpcInlineReceiptReader(),
        createGrpcMMControlClient(metrics, transport_config.control.release_timeout_ms),
        transport_config.default_rpc_timeout_ms,
        transport_config.rpc_timeout_margin_ms);
}

}  // namespace rtp_llm
