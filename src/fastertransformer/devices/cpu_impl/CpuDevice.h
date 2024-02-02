#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

class CpuDevice : public DeviceBase {
public:
    CpuDevice();
    ~CpuDevice();

public:
    std::string type() const override { return "cpu"; }
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return allocator_.get(); }

public:
    void layernorm(LayernormParams& params);
    void gemm(GemmParams& params);
    void contextAttention(AttentionModuleParams& params);
    void decoderSelfAttention(AttentionModuleParams& params);
    void allocateBuffers(AllocateBufferParams& params);
    void attentionLayer(AttentionLayerParams& params);
    void ffnLayer(FfnLayerParams& params);
    void sampleTopP(SamplerParams& params);
    void sampleTopK(SamplerParams& params);
    void broadcast(BroadcastParams& params);
    void allReduceSum(AllReduceParams& params);

private:
    std::unique_ptr<IAllocator> allocator_;
};

} // namespace fastertransformer

