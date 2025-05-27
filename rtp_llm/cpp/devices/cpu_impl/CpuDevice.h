#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

class CpuDevice: public DeviceBase {
public:
    CpuDevice(const DeviceInitParams& params);
    ~CpuDevice();

public:
    DeviceProperties getDeviceProperties() override;
    IAllocator*      getAllocator() override {
        return allocator_.get();
    }
    IAllocator* getHostAllocator() override {
        return allocator_.get();
    }
    torch::Device getTorchDevice() override {
        return torch::Device(torch::kCPU);
    };

public:
    void                  copy(const CopyParams& params);
    LayernormOutput       layernorm(const LayernormParams& params);
    BufferPtr             gemm(const GemmParams& params);
    GroupedGemmOutput     groupedGemm(const GroupedGemmParams& params);
    BufferPtr             embeddingLookup(const EmbeddingLookupParams& params);
    BufferPtr             activation(const ActivationParams& params);
    BufferPtr             softmax(const SoftmaxParams& params);
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params);
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params);
    AttentionLayerOutput  attentionLayer(const AttentionLayerParams& params);
    FfnLayerOutput        ffnLayer(const FfnLayerParams& params);
    GreedyOutput          sampleGreedy(const GreedyParams& params);
    BeamSearchOutput      sampleBeamSearch(const BeamSearchParams& params);
    void                  broadcast(const BroadcastParams& params);
    AllReduceOutput       allReduce(const AllReduceParams& params);

private:
    std::unique_ptr<IAllocator> allocator_;
};

}  // namespace rtp_llm
