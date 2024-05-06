#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"
#include "src/fastertransformer/devices/DeviceData.h"
#include "src/fastertransformer/devices/BufferManager.h"

namespace fastertransformer {

class DeviceBase : public DeviceOps {
public:
    DeviceBase();

    void init();
    virtual DeviceProperties getDeviceProperties() = 0;
    virtual DeviceStatus getDeviceStatus();
    std::unique_ptr<Buffer> allocateBuffer(const BufferParams& params, const BufferHints& hints = {});
    std::unique_ptr<Buffer> allocateBufferLike(const Buffer& buffer, const BufferHints& hints = {});
    virtual void syncAndCheck();

public:
    // device-independence op implementations
    CloneOutput clone(const CloneParams& params) override;
    AttentionLayerOutput attentionLayer(const AttentionLayerParams& params) override;
    FfnLayerOutput ffnLayer(const FfnLayerParams& params) override;
    LoraLinearOutput loraLinear(const LoraLinearParams& params) override;

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
