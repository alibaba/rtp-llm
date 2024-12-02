#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <sstream>

namespace fastertransformer {

// DeviceEvent is similar to cudaEvent_t, but it is device-independent.
struct DeviceEvent {
    virtual ~DeviceEvent() = default;
    virtual void synchronize() const = 0;
};

using DeviceEventPtr = std::unique_ptr<DeviceEvent>;

} // namespace fastertransformer

