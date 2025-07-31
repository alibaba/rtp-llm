#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <sstream>

namespace rtp_llm {

// DeviceEvent is similar to cudaEvent_t, but it is device-independent.
struct DeviceEvent {
    virtual ~DeviceEvent()           = default;
    virtual void synchronize() const = 0;
    virtual bool checkReadiness() const = 0;
};

using DeviceEventPtr = std::shared_ptr<DeviceEvent>;

// Event hook is used for hooking communication streams or other io operations
// that is simultaneous with some computation but needs to be synchronized before another computation
struct DeviceHook {
    virtual ~DeviceHook()          = default;
    virtual void hook_sync() const = 0;
};

using DeviceHookPtr = std::shared_ptr<DeviceHook>;

}  // namespace rtp_llm
