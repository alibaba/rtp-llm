#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"

namespace fastertransformer {

class DeviceBase : public DeviceOps {
public:
    DeviceBase();

    virtual std::string type() const = 0;
    IAllocator*         getAllocator();

private:
    DeviceBase(const DeviceBase&) = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&) = delete;

protected:
    std::unique_ptr<IAllocator> allocator_;
};

};  // namespace fastertransformer
