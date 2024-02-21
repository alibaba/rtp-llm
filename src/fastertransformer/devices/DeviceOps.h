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
    virtual void copy(const CopyParams& params) = 0;

    // basic compuation ops
    virtual void layernorm(const LayernormParams& params)        = 0;
    virtual void gemm(const GemmParams& params)                  = 0;
    virtual void groupedGemm(const GroupedGemmParams& params)    = 0;

    // dedicated attention ops
    virtual void contextAttention(const AttentionModuleParams& params)     = 0;
    virtual void decoderSelfAttention(const AttentionModuleParams& params) = 0;

    // Top level model ops
    virtual void attentionLayer(const AttentionLayerParams& params) = 0;
    virtual void ffnLayer(const FfnLayerParams& params)             = 0;

    // for sampler
    virtual void sampleTopP(const SamplerParams& params) = 0;
    virtual void sampleTopK(const SamplerParams& params) = 0;

    // for device communication
    virtual void broadcast(const BroadcastParams& params) = 0;
    virtual void allReduceSum(const AllReduceParams& params) = 0;
};

}  // namespace fastertransformer
