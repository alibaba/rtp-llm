#pragma once

#include <memory>

#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"

namespace rtp_llm {

std::unique_ptr<MMControlClient> createGrpcMMControlClient(MMTransportMetricsPtr metrics,
                                                           int64_t release_timeout_ms);
std::unique_ptr<MMTerminalReceiptReader> createGrpcInlineReceiptReader();

}  // namespace rtp_llm
