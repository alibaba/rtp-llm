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
    virtual OpStatus copy(const CopyParams& params) = 0;

    // basic compuation ops
    virtual OpStatus layernorm(const LayernormParams& params)        = 0;
    virtual OpStatus gemm(const GemmParams& params)                  = 0;
    virtual OpStatus groupedGemm(const GroupedGemmParams& params)    = 0;

    // dedicated attention ops
    virtual OpStatus contextAttention(const AttentionModuleParams& params)     = 0;
    virtual OpStatus decoderSelfAttention(const AttentionModuleParams& params) = 0;

    // Top level model ops
    virtual OpStatus attentionLayer(const AttentionLayerParams& params) = 0;
    virtual OpStatus ffnLayer(const FfnLayerParams& params)             = 0;

    // for sampler
    virtual OpStatus sampleTopP(const SamplerParams& params) = 0;
    virtual OpStatus sampleTopK(const SamplerParams& params) = 0;

    // for device communication
    virtual OpStatus broadcast(const BroadcastParams& params) = 0;
    virtual OpStatus allReduceSum(const AllReduceParams& params) = 0;
};

}  // namespace fastertransformer
