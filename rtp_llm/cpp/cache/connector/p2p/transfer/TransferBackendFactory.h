#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <cstdint>
#include <memory>

namespace rtp_llm {
namespace transfer {

enum class TransferBackend {
    kTcp,
    kBarexRdma,
};

enum class TransferBackendLifecycleState {
    kUninitialized,
    kRunning,
    kCheckpointing,
    kSleeping,
    kRestoring,
    kRestored,
    kFailed,
};

class ITransferBackendLifecycle {
public:
    virtual ~ITransferBackendLifecycle() = default;

    // Level 3 checkpoints the transport only: the caller quiesces transfer
    // requests, then stopTransportForCheckpoint tears down QPs/MRs/CQs while the
    // backing MR owner stays alive until the controller deregisters KV MRs in the
    // following phase. Backends must retain the connector lease through every
    // physical completion callback before advertising this capability.
    // restoreTransportAfterCheckpoint rebuilds transport after CUDA checkpoint
    // restore has recreated every registered backing allocation; the restored
    // backend stays gated until the unified wake path calls resume().
    virtual bool supportsTransportOnlyCheckpoint() const {
        return false;
    }
    virtual bool stopTransportForCheckpoint() {
        return false;
    }
    virtual bool restoreTransportAfterCheckpoint() {
        return false;
    }
    virtual bool resume() = 0;
};

struct TransferBackendPair {
    IKVCacheSenderPtr   sender;
    IKVCacheReceiverPtr receiver;

    // Stateless backends (currently TCP) intentionally treat lifecycle calls as no-ops.
    bool resume() const {
        auto lifecycle = getLifecycle();
        return lifecycle ? lifecycle->resume() : true;
    }

    bool supportsTransportOnlyCheckpoint() const {
        auto lifecycle = getLifecycle();
        return lifecycle && lifecycle->supportsTransportOnlyCheckpoint();
    }

    bool stopTransportForCheckpoint() const {
        auto lifecycle = getLifecycle();
        return lifecycle && lifecycle->stopTransportForCheckpoint();
    }

    bool restoreTransportAfterCheckpoint() const {
        auto lifecycle = getLifecycle();
        return lifecycle && lifecycle->restoreTransportAfterCheckpoint();
    }

private:
    std::shared_ptr<ITransferBackendLifecycle> getLifecycle() const {
        if (sender) {
            if (auto lifecycle = std::dynamic_pointer_cast<ITransferBackendLifecycle>(sender)) {
                return lifecycle;
            }
        }
        if (receiver) {
            return std::dynamic_pointer_cast<ITransferBackendLifecycle>(receiver);
        }
        return nullptr;
    }
};

TransferBackendPair createTransferBackend(TransferBackend                     backend,
                                          const TransferBackendConfig&        config,
                                          const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);

}  // namespace transfer
}  // namespace rtp_llm
