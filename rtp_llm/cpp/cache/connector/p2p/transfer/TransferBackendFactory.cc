#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpBackendFactory.h"

#include <stdexcept>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {

TransferBackendPair createTransferBackend(TransferBackend                     backend,
                                          const TransferBackendConfig&        config,
                                          TransferBackendRole                 role,
                                          const kmonitor::MetricsReporterPtr& metrics_reporter) {
    switch (backend) {
        case TransferBackend::kTcp:
            return createTcpBackend(config, role, metrics_reporter);
        case TransferBackend::kBarexRdma:
            throw std::runtime_error("BarexRdma backend not supported in this build");
        default:
            RTP_LLM_LOG_ERROR("createTransferBackend: unknown backend");
            return {};
    }
}

}  // namespace transfer
}  // namespace rtp_llm
