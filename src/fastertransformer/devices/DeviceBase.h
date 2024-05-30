#pragma once

#include "src/fastertransformer/devices/DeviceOps.h"
#include "src/fastertransformer/devices/DeviceData.h"
#include "src/fastertransformer/devices/BufferManager.h"

namespace fastertransformer {

class DeviceBase : public DeviceOps {
public:
    DeviceBase(const DeviceInitParams& params);

    virtual void init();
    virtual DeviceProperties getDeviceProperties() = 0;
    virtual DeviceStatus getDeviceStatus();
    void traceMemoryUsage();
    BufferPtr allocateBuffer(const BufferParams& params, const BufferHints& hints = {});
    BufferPtr allocateBufferLike(const Buffer& buffer, const BufferHints& hints = {});
    virtual void syncAndCheck();
    virtual void syncCommunication();

public:
    // device-independence op implementations
    CloneOutput clone(const CloneParams& params) override;
    AttentionLayerOutput attentionLayer(const AttentionLayerParams& params) override;
    FfnLayerOutput ffnLayer(const FfnLayerParams& params) override;
    LoraLinearOutput loraLinear(const LoraLinearParams& params) override;

protected:
    BufferStatus queryBufferStatus();
    AllocationType getMemAllocationType(const MemoryType type);

private:
    DeviceBase(const DeviceBase&) = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&) = delete;

private:
    virtual IAllocator* getAllocator() = 0;
    virtual IAllocator* getHostAllocator() = 0;

protected:
    int device_id_;
    DeviceInitParams init_params_;

private:
    std::unique_ptr<BufferManager> buffer_manager_;
};

};  // namespace fastertransformer
