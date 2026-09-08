#pragma once

#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <optional>

namespace rtp_llm {

struct PpuFusedRopeParams: public ParamsBase {
    KVBlockArray  kv_block_array;
    torch::Tensor kv_cache_offset;
    torch::Tensor padding_offset;
    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor prefix_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor position_ids;
    int           max_seq_len       = 0;
    int           max_prefix_length = 0;
};

using PpuFusedRopeParamsPtr = std::shared_ptr<PpuFusedRopeParams>;

class FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs,
                                  size_t                  max_seq_len  = 0,
                                  bool                    use_fp8_fmha = false);
    PpuFusedRopeParamsPtr prepare(torch_ext::PyAttentionInputs attn_inputs);

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
    torch::Tensor forward(const torch::Tensor&                   qkv,
                          std::optional<torch_ext::LayerKVCache> kv_cache,
                          const PpuFusedRopeParamsPtr&           params);
};

class FusedRopeKVCachePrefillOpQOut: public FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpQOut(const AttentionConfigs& attn_configs,
                                  size_t                  max_seq_len  = 0,
                                  bool                    use_fp8_fmha = false);
    torch::Tensor forward(const torch::Tensor&                   qkv,
                          std::optional<torch_ext::LayerKVCache> kv_cache,
                          const PpuFusedRopeParamsPtr&           params);
};

class FusedRopeKVCacheDecodeOp {
public:
    FusedRopeKVCacheDecodeOp(const AttentionConfigs& attn_configs, size_t max_seq_len = 0, bool use_fp8_fmha = false);
    PpuFusedRopeParamsPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor         forward(const torch::Tensor&                   qkv,
                                  std::optional<torch_ext::LayerKVCache> kv_cache,
                                  const PpuFusedRopeParamsPtr&           params);

protected:
    AttentionConfigs attn_configs_;
    size_t           max_seq_len_;
    bool             use_fp8_fmha_;
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm
