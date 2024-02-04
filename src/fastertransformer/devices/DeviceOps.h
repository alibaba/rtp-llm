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

    // virtual void

    // basic ops
    virtual void layernorm(LayernormParams& params) = 0;
    virtual void gemm(GemmParams& params)           = 0;

    // dedicated attention ops
    virtual void contextAttention(AttentionModuleParams& params)     = 0;
    virtual void decoderSelfAttention(AttentionModuleParams& params) = 0;

    // Top level model ops
    virtual void attentionLayer(AttentionLayerParams& params) = 0;
    virtual void ffnLayer(FfnLayerParams& params)             = 0;

    // for sampler
    virtual void sampleTopP(SamplerParams& params) = 0;
    virtual void sampleTopK(SamplerParams& params) = 0;

    // for device communication
    virtual void broadcast(BroadcastParams& params) = 0;
    virtual void allReduceSum(AllReduceParams& params) = 0;
};

}  // namespace fastertransformer
