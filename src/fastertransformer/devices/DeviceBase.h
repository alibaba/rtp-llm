#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"
#include "src/fastertransformer/devices/BufferManager.h"

namespace fastertransformer {

class DeviceBase : public DeviceOps {
public:
    DeviceBase();

    void init();
    std::shared_ptr<Tensor> allocateBuffer(const BufferParams& params, const BufferHints& hints);
    virtual std::string type() const = 0;

private:
    DeviceBase(const DeviceBase&) = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&) = delete;

private:
    virtual IAllocator* getAllocator() = 0;
    virtual IAllocator* getHostAllocator() = 0;

private:
    std::unique_ptr<BufferManager> buffer_manager_;
};

};  // namespace fastertransformer
