#pragma once

#include <cstddef>
#include <functional>

namespace rtp_llm::kvcm {

// RTP-internal capability; does not change the external SDK TransferClient ABI.
// Called once per attempted SDK operation, independently of later GPU scatter
// or URI validation. Callbacks must be thread-safe and must not reenter I/O.
using SdkTransferObserver = std::function<void(bool is_read, size_t completed_bytes, bool completion_unknown)>;

class SdkTransferObservable {
public:
    virtual ~SdkTransferObservable()                                  = default;
    virtual void setSdkTransferObserver(SdkTransferObserver observer) = 0;
};

}  // namespace rtp_llm::kvcm
