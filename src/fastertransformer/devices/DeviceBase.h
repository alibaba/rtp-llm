#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"
#include "src/fastertransformer/devices/BufferManager.h"

namespace fastertransformer {

class DeviceBase : public DeviceOps {
public:
    DeviceBase();

    void init();
    std::unique_ptr<Buffer> allocateBuffer(const BufferParams& params, const BufferHints& hints = {});
    virtual std::string type() const = 0;

public:
    // target independence ops
    FfnLayerOutput ffnLayer(const FfnLayerParams& params);
    LoraLinearOutput loraLinear(const LoraLinearParams& params);

private:
    DeviceBase(const DeviceBase&) = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&) = delete;

private:
    virtual IAllocator* getAllocator() = 0;
    virtual IAllocator* getHostAllocator() = 0;

private:
    int device_id_;
    std::unique_ptr<BufferManager> buffer_manager_;
};

};  // namespace fastertransformer
