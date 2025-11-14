#pragma once

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class FusedRopeKVCachePrefillOp {
public:
    FusedRopeKVCachePrefillOp(const AttentionConfigs& attn_configs, int layer_num, const HWKernelConfig& hw_kernel_config);
    CKAttnPtr                                               prepare(torch_ext::PyAttentionInputs attn_inputs);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor&              qkv,
                                                                    FMHAType                          fmha_type,
                                                                    std::optional<torch_ext::KVCache> kv_cache_base,
                                                                    const CKAttnPtr&                  params);

protected:
    AttentionConfigs attn_configs_;
    int              layer_num_;
    HWKernelConfig   hw_kernel_config_;
    ROCmDevice*      device_;
};

class FusedRopeKVCacheDecodeOp {
public:
    FusedRopeKVCacheDecodeOp(const AttentionConfigs& attn_configs, int layer_num, const HWKernelConfig& hw_kernel_config);
    CKAttnPtr     prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor forward(const torch::Tensor&              qkv,
                          FMHAType                          fmha_type,
                          std::optional<torch_ext::KVCache> kv_cache_base,
                          const CKAttnPtr&                  params);

protected:
    AttentionConfigs attn_configs_;
    int              layer_num_;
    HWKernelConfig   hw_kernel_config_;
    ROCmDevice*      device_;
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm