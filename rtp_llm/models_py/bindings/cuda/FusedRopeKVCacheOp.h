#pragma once

#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <optional>

namespace rtp_llm {

class FusedRopeKVCachePrefillOpBase: public FMHACudaBase {
public:
    FusedRopeKVCachePrefillOpBase(const GptInitParameter& gpt_init_parameter);
    TRTAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
};

class FusedRopeKVCachePrefillOpQKVOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQKVOut(const GptInitParameter& gpt_init_parameter);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const TRTAttnPtr&                 params);
};

class FusedRopeKVCachePrefillOpQOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQOut(const GptInitParameter& gpt_init_parameter);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const TRTAttnPtr&                 params);
};

class FusedRopeKVCacheDecodeOp: public FMHACudaBase {
public:
    FusedRopeKVCacheDecodeOp(const GptInitParameter& gpt_init_parameter);
    TRTAttnPtr    prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const TRTAttnPtr&                 params);
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm
