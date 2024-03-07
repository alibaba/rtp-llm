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
    void copy(const CopyParams& params);
    void layernorm(const LayernormParams& params);
    void gemm(const GemmParams& params);
    void groupedGemm(const GroupedGemmParams& params);
    void embeddingLookup(const EmbeddingLookupParams& params);
    void contextAttention(const AttentionModuleParams& params);
    void decoderSelfAttention(const AttentionModuleParams& params);
    void attentionLayer(const AttentionLayerParams& params);
    void ffnLayer(const FfnLayerParams& params);
    void sampleTopP(const SamplerParams& params);
    void sampleTopK(const SamplerParams& params);
    void broadcast(const BroadcastParams& params);
    void allReduceSum(const AllReduceParams& params);

private:
    std::unique_ptr<IAllocator> allocator_;
};

} // namespace fastertransformer

