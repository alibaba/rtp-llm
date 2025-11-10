#pragma once

#include "rtp_llm/models_py/bindings/rocm/FMHARocmBase.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class FusedRopeKVCachePrefillOp: public FMHARocmBase {
public:
    FusedRopeKVCachePrefillOp(const GptInitParameter& gpt_init_parameter);
    CKAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache_base,
                          const CKAttnPtr&                  params);
};

class FusedRopeKVCacheDecodeOp: public FMHARocmBase {
public:
    FusedRopeKVCacheDecodeOp(const GptInitParameter& gpt_init_parameter);
    CKAttnPtr     prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache_base,
                          const CKAttnPtr&                  params);
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm