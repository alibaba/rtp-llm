#pragma once

#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/cuda/cufmha/TRTAttn.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <optional>

namespace rtp_llm {

class FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs,
                                  size_t                  max_seq_len  = 0,
                                  bool                    use_fp8_fmha = false);
    TRTAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);

protected:
    AttentionConfigs attn_configs_;
    size_t           max_seq_len_;
    bool             use_fp8_fmha_;
};

class FusedRopeKVCachePrefillOpQKVOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQKVOut(const AttentionConfigs& attn_configs,
                                    size_t                  max_seq_len  = 0,
                                    bool                    use_fp8_fmha = false);
    torch::Tensor
    forward(const torch::Tensor& qkv, std::optional<torch_ext::LayerKVCache> kv_cache, const TRTAttnPtr& params);
};

class FusedRopeKVCachePrefillOpQOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQOut(const AttentionConfigs& attn_configs,
                                  size_t                  max_seq_len  = 0,
                                  bool                    use_fp8_fmha = false);
    torch::Tensor
    forward(const torch::Tensor& qkv, std::optional<torch_ext::LayerKVCache> kv_cache, const TRTAttnPtr& params);
};

class FusedRopeKVCacheDecodeOp {
public:
    FusedRopeKVCacheDecodeOp(const AttentionConfigs& attn_configs, size_t max_seq_len = 0, bool use_fp8_fmha = false);
    TRTAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor
    forward(const torch::Tensor& qkv, std::optional<torch_ext::LayerKVCache> kv_cache, const TRTAttnPtr& params);

protected:
    AttentionConfigs attn_configs_;
    size_t           max_seq_len_;
    bool             use_fp8_fmha_;
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm
