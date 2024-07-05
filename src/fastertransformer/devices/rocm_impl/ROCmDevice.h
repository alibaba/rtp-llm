#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/rocm/hip_utils.h"

namespace fastertransformer {

class ROCmDevice: public DeviceBase {
public:
    ROCmDevice(const DeviceInitParams& params);
    ~ROCmDevice();

    DeviceProperties getDeviceProperties() override;
    IAllocator*      getAllocator() override {
        return allocator_.get();
    }
    IAllocator* getHostAllocator() override {
        return hostAllocator_.get();
    }
    
    void                  copy(const CopyParams& params) override;
    void                  syncAndCheck() override;
    BufferPtr             gemm(const GemmParams& params) override;
    SelectOutput          select(const SelectParams& params) override;
    BufferPtr             embeddingLookup(const EmbeddingLookupParams& params) override;
    LayernormOutput       layernorm(const LayernormParams& params) override;
    void                  activation(const ActivationParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;

public:
    BufferPtr        testVecAdd(const BufferPtr a, const BufferPtr b);
    hipDeviceProp_t* getRocmDeviceProperties() {
        return &rocmDevProp;
    }

private:
    hipDeviceProp_t             rocmDevProp;
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> hostAllocator_;
    hipStream_t                 stream_ = nullptr;
};

}  // namespace fastertransformer
