#pragma once

#include <memory>

namespace rtp_llm {

// Device-agnostic async event interface used by connector-side wait logic.
struct AsyncEvent {
    virtual ~AsyncEvent()               = default;
    virtual void synchronize() const    = 0;
    virtual bool checkReadiness() const = 0;
};

using AsyncEventPtr = std::shared_ptr<AsyncEvent>;

}  // namespace rtp_llm
