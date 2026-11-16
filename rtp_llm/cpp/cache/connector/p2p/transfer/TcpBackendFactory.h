#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

namespace rtp_llm::transfer {

TransferBackendPair createTcpBackend(const TransferBackendConfig&        config,
                                     TransferBackendRole                 role,
                                     const kmonitor::MetricsReporterPtr& metrics_reporter);

}  // namespace rtp_llm::transfer
