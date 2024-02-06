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

    // basic ops
    virtual OpStatus layernorm(LayernormParams& params)        = 0;
    virtual OpStatus gemm(GemmParams& params)                  = 0;
    virtual OpStatus groupedGemm(GroupedGemmParams& params)    = 0;

    // dedicated attention ops
    virtual OpStatus contextAttention(AttentionModuleParams& params)     = 0;
    virtual OpStatus decoderSelfAttention(AttentionModuleParams& params) = 0;

    // Top level model ops
    virtual OpStatus attentionLayer(AttentionLayerParams& params) = 0;
    virtual OpStatus ffnLayer(FfnLayerParams& params)             = 0;

    // for sampler
    virtual OpStatus sampleTopP(SamplerParams& params) = 0;
    virtual OpStatus sampleTopK(SamplerParams& params) = 0;

    // for device communication
    virtual OpStatus broadcast(BroadcastParams& params) = 0;
    virtual OpStatus allReduceSum(AllReduceParams& params) = 0;
};

}  // namespace fastertransformer
