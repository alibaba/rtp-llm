#pragma once

#include "src/fastertransformer/devices/OpData.h"

namespace fastertransformer {

class DeviceOps {
public:
    DeviceOps();
    virtual ~DeviceOps();

public:
    // tensor ops
    virtual void copy(const CopyParams& params);
    virtual CloneOutput clone(const CloneParams& params);
    virtual TransposeOutput transpose(const TransposeParams& params);
    virtual ConvertOutput convert(const ConvertParams& params);
    virtual SelectOutput select(const SelectParams& params);

    // basic compuation ops
    virtual LayernormOutput layernorm(const LayernormParams& params);
    virtual BufferPtr gemm(const GemmParams& params);
    virtual GroupedGemmOutput groupedGemm(const GroupedGemmParams& params);
    virtual BufferPtr embeddingLookup(const EmbeddingLookupParams& params);
    virtual void activation(const ActivationParams& params);
    virtual BufferPtr softmax(const SoftmaxParams& params);

    // dedicated attention ops
    virtual AttentionModuleOutput contextAttention(const AttentionModuleParams& params);
    virtual AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params);

    // Top level model ops
    virtual AttentionLayerOutput attentionLayer(const AttentionLayerParams& params);
    virtual FfnLayerOutput ffnLayer(const FfnLayerParams& params);
    virtual LoraLinearOutput loraLinear(const LoraLinearParams& params);

    // for sampler
    virtual void sampleGreedy(const GreedyParams& params);
    virtual void sampleBeamSearch(const BeamSearchParams& params);

    // for device communication
    virtual void broadcast(const BroadcastParams& params);
    virtual void allReduce(const AllReduceParams& params);
    virtual void allGather(const AllGatherParams& params);
};

}  // namespace fastertransformer
