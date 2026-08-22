#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransportFactory.h"

#include <utility>
#include <vector>

#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReceiptReader.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
inline constexpr const char* kMMTransportModeAuto = "auto";
inline constexpr const char* kMMTransportModeGrpc = "grpc";
}  // namespace

std::unique_ptr<MMRemoteOutputTransport>
createMMRemoteOutputTransport(const MMTransportConfig& transport_config, kmonitor::MetricsReporterPtr reporter) {
    auto metrics = std::make_shared<const MMTransportMetrics>(std::move(reporter));

    std::shared_ptr<MMRdmaTransport> rdma_transport;
    if (transport_config.mode == kMMTransportModeAuto) {
        rdma_transport = createMMRdmaClientTransport(transport_config.rdma);
    } else if (transport_config.mode == kMMTransportModeGrpc) {
        RTP_LLM_LOG_INFO("mm transport mode is grpc, skip rdma initialization");
    } else {
        RTP_LLM_LOG_WARNING("unknown mm transport mode '%s', fall back to grpc", transport_config.mode.c_str());
    }

    std::vector<std::unique_ptr<MMReceiptReader>> readers;
    readers.push_back(createMMRdmaReceiptReader(std::move(rdma_transport), metrics));

    return std::make_unique<MMRemoteOutputTransport>(
        std::move(readers),
        createGrpcInlineReceiptReader(),
        createGrpcMMControlClient(metrics, transport_config.control.release_timeout_ms),
        metrics);
}

}  // namespace rtp_llm
