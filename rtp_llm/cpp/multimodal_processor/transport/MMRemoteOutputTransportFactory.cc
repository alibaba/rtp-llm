#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransportFactory.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/MMTransportMode.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"
#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmReader.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::unique_ptr<MMRemoteOutputTransport>
createMMRemoteOutputTransport(const MMTransportConfig&     transport_config,
                              kmonitor::MetricsReporterPtr reporter,
                              int                          device_id) {
    const auto mode = validateMMTransportMode(transport_config.mode);
    auto metrics = std::make_shared<const MMTransportMetrics>(std::move(reporter));

    std::vector<std::unique_ptr<MMReceiptReader>> readers;
    if (mode == kMMTransportModeRdma) {
        auto reader = rdma_transport::createRdmaRead(transport_config.rdma, device_id);
        if (reader == nullptr) {
            throw std::runtime_error("failed to initialize RDMA reader for multimodal output transport");
        }
        RTP_LLM_LOG_INFO("mm transport mode=rdma: reader initialized");
        readers.push_back(std::make_unique<MMRdmaReader>(std::move(reader), transport_config.rdma));
    } else {
        readers.push_back(createMMRdmaReader(nullptr));
    }
    if (mode == kMMTransportModeKvcm) {
        const auto config_error = validateMMKvcmConfig(transport_config.kvcm);
        if (!config_error.empty()) {
            throw std::invalid_argument(config_error);
        }
        auto client = createMMKvcmClient(transport_config.kvcm);
        if (client == nullptr) {
            if (!hasMMKvcmImplementation()) {
                throw std::runtime_error(
                    "KVCM EMB storage is not linked; rebuild with --define=use_kvcm_emb_storage=true");
            }
            throw std::runtime_error("failed to initialize KVCM EMB object client");
        }
        RTP_LLM_LOG_INFO("mm transport mode=kvcm: exact-size EMB object reader initialized");
        readers.push_back(std::make_unique<MMKvcmReader>(std::move(client), transport_config.kvcm, device_id));
    } else {
        readers.push_back(createMMKvcmReader(nullptr));
    }
    if (mode == kMMTransportModeGrpc) {
        RTP_LLM_LOG_INFO("mm transport mode=grpc: external data planes disabled, use inline grpc");
    }

    return std::make_unique<MMRemoteOutputTransport>(
        std::move(readers),
        createGrpcInlineReceiptReader(),
        createGrpcMMControlClient(metrics, transport_config.control.release_timeout_ms),
        transport_config.default_rpc_timeout_ms,
        transport_config.rpc_timeout_margin_ms);
}

}  // namespace rtp_llm
