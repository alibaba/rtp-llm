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
    OpStatus copy(const CopyParams& params);
    OpStatus layernorm(const LayernormParams& params);
    OpStatus gemm(const GemmParams& params);
    OpStatus groupedGemm(const GroupedGemmParams& params);
    OpStatus contextAttention(const AttentionModuleParams& params);
    OpStatus decoderSelfAttention(const AttentionModuleParams& params);
    OpStatus attentionLayer(const AttentionLayerParams& params);
    OpStatus ffnLayer(const FfnLayerParams& params);
    OpStatus sampleTopP(const SamplerParams& params);
    OpStatus sampleTopK(const SamplerParams& params);
    OpStatus broadcast(const BroadcastParams& params);
    OpStatus allReduceSum(const AllReduceParams& params);

private:
    std::unique_ptr<IAllocator> allocator_;
};

} // namespace fastertransformer

