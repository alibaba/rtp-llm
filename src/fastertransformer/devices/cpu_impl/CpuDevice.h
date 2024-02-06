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
    OpStatus copy();
    OpStatus layernorm(LayernormParams& params);
    OpStatus gemm(GemmParams& params);
    OpStatus groupedGemm(GroupedGemmParams& params);
    OpStatus contextAttention(AttentionModuleParams& params);
    OpStatus decoderSelfAttention(AttentionModuleParams& params);
    OpStatus attentionLayer(AttentionLayerParams& params);
    OpStatus ffnLayer(FfnLayerParams& params);
    OpStatus sampleTopP(SamplerParams& params);
    OpStatus sampleTopK(SamplerParams& params);
    OpStatus broadcast(BroadcastParams& params);
    OpStatus allReduceSum(AllReduceParams& params);

private:
    std::unique_ptr<IAllocator> allocator_;
};

} // namespace fastertransformer

