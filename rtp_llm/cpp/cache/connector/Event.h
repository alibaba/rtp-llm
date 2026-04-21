#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <sstream>

namespace rtp_llm {

// AsyncEvent is similar to cudaEvent_t, but it is device-independent.
struct AsyncEvent {
    virtual ~AsyncEvent()               = default;
    virtual void synchronize() const    = 0;
    virtual bool checkReadiness() const = 0;
};

using AsyncEventPtr = std::shared_ptr<AsyncEvent>;

}  // namespace rtp_llm
