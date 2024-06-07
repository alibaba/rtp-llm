#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

class ROCmDevice : public DeviceBase {
public:
    ROCmDevice(const DeviceInitParams& params);
    ~ROCmDevice();

public:
    DeviceProperties getDeviceProperties() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return allocator_.get(); }

public:
    void copy(const CopyParams& params);

private:
    std::unique_ptr<IAllocator> allocator_;
};

} // namespace fastertransformer

