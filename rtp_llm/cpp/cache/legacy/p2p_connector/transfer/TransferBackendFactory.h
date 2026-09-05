#pragma once

#include "rtp_llm/cpp/cache/legacy/p2p_connector/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <memory>

namespace rtp_llm::legacy::p2p {
namespace transfer {

enum class TransferBackend {
    kTcp,
    kBarexRdma,
};

struct TransferBackendPair {
    IKVCacheSenderPtr   sender;
    IKVCacheReceiverPtr receiver;
};

TransferBackendPair createTransferBackend(TransferBackend                     backend,
                                          const TransferBackendConfig&        config,
                                          const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);

}  // namespace transfer
}  // namespace rtp_llm::legacy::p2p
