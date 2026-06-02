#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <memory>

namespace rtp_llm {
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
}  // namespace rtp_llm
