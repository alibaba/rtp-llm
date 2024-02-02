#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"

namespace fastertransformer {

class DeviceBase : public DeviceOps {
public:
    DeviceBase();

    virtual std::string type() const = 0;
    virtual IAllocator* getAllocator() = 0;
    virtual IAllocator* getHostAllocator() = 0;

private:
    DeviceBase(const DeviceBase&) = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&) = delete;

};

};  // namespace fastertransformer
