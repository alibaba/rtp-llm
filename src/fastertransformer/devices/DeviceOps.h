#pragma once

#include "src/fastertransformer/devices/ModelInfo.h"
#include "src/fastertransformer/devices/QueryInfo.h"
#include "src/fastertransformer/devices/OpData.h"

namespace fastertransformer {

class DeviceOps {
public:
    DeviceOps();
    virtual ~DeviceOps();

public:
    virtual size_t getKvCacheBlockSize(const ModelInfo& model) const ;

    // tensor ops
    virtual void copy(const CopyParams& params)                                             = 0;

    // basic compuation ops
    virtual LayernormOutput layernorm(const LayernormParams& params)                        = 0;
    virtual BufferPtr gemm(const GemmParams& params)                                        = 0;
    virtual GroupedGemmOutput groupedGemm(const GroupedGemmParams& params)                  = 0;
    virtual BufferPtr embeddingLookup(const EmbeddingLookupParams& params)                  = 0;
    virtual void activation(const ActivationParams& params)                                 = 0;
    virtual BufferPtr softmax(const SoftmaxParams& params)                                  = 0;

    // dedicated attention ops
    virtual AttentionModuleOutput contextAttention(const AttentionModuleParams& params)     = 0;
    virtual AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) = 0;

    // Top level model ops
    virtual AttentionLayerOutput attentionLayer(const AttentionLayerParams& params)         = 0;
    virtual FfnLayerOutput ffnLayer(const FfnLayerParams& params)                           = 0;
    virtual LoraLinearOutput loraLinear(const LoraLinearParams& params)                     = 0;

    // for sampler
    virtual void sampleGreedy(const GreedyParams& params)                                   = 0;
    virtual void sampleBeamSearch(const BeamSearchParams& params)                           = 0;

    // for device communication
    virtual void broadcast(const BroadcastParams& params)                                   = 0;
    virtual void allReduceSum(const AllReduceParams& params)                                = 0;
};

}  // namespace fastertransformer
