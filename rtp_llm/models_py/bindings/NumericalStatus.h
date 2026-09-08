#pragma once

#include <c10/core/Stream.h>
#include <cstdint>
#include <memory>
#include <torch/extension.h>

namespace rtp_llm {

enum class NumericalStatusScope : uint8_t {
    NONE       = 0,
    BATCH      = 1,
    ORIGIN_ROW = 2,
};

class NumericalStatusLease {
public:
    virtual ~NumericalStatusLease() = default;

    // The single device consumer waits before reading the snapshot and should
    // record completion after its final read. If it omits markConsumed(), lease
    // destruction records a fence on the stream passed to waitReady(). Neither
    // operation waits on the host.
    virtual void waitReady(const c10::Stream& stream)    = 0;
    virtual void markConsumed(const c10::Stream& stream) = 0;
};

struct NumericalStatusView {
    torch::Tensor                        values;
    int64_t                              live_rows = 0;
    NumericalStatusScope                 scope     = NumericalStatusScope::NONE;
    uint64_t                             epoch     = 0;
    std::shared_ptr<torch::Event>         ready_event;
    std::shared_ptr<NumericalStatusLease> lease;

    bool defined() const {
        return scope != NumericalStatusScope::NONE && values.defined();
    }

    void waitReady(const c10::Stream& stream) const {
        if (lease) {
            lease->waitReady(stream);
        }
    }

    void markConsumed(const c10::Stream& stream) const {
        if (lease) {
            lease->markConsumed(stream);
        }
    }
};

}  // namespace rtp_llm
