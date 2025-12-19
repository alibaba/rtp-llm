#pragma once

#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include <optional>

namespace rtp_llm {

class FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs);
    TRTAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);

protected:
    AttentionConfigs attn_configs_;
    CudaDevice*      device_;
};

class FusedRopeKVCachePrefillOpQKVOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQKVOut(const AttentionConfigs& attn_configs);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const TRTAttnPtr&                 params);
};

class FusedRopeKVCachePrefillOpQOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQOut(const AttentionConfigs& attn_configs);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const TRTAttnPtr&                 params);
};

class FusedRopeKVCacheDecodeOp {
public:
    FusedRopeKVCacheDecodeOp(const AttentionConfigs& attn_configs);
    TRTAttnPtr    prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const TRTAttnPtr&                 params);

protected:
    AttentionConfigs attn_configs_;
    CudaDevice*      device_;
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm
