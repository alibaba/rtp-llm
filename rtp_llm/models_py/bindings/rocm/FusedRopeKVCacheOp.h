#pragma once

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {

// Base class for Prefill operations
class FusedRopeKVCachePrefillOpBase {
public:
    FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs);
    CKAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& qkv, std::optional<torch_ext::KVCache> kv_cache_base, const CKAttnPtr& params);

protected:
    AttentionConfigs attn_configs_;
    ROCmDevice*      device_;  // Only used for PrepareCKAttn
    virtual bool     use_asm() const = 0;
};

// ASM version of Prefill operation
class FusedRopeKVCachePrefillOpAsm: public FusedRopeKVCachePrefillOpBase {
protected:
    bool use_asm() const override {
        return true;
    }

public:
    FusedRopeKVCachePrefillOpAsm(const AttentionConfigs& attn_configs);
};

// Non-ASM version of Prefill operation
class FusedRopeKVCachePrefillOpNonAsm: public FusedRopeKVCachePrefillOpBase {
protected:
    bool use_asm() const override {
        return false;
    }

public:
    FusedRopeKVCachePrefillOpNonAsm(const AttentionConfigs& attn_configs);
};

// Base class for Decode operations
class FusedRopeKVCacheDecodeOpBase {
public:
    FusedRopeKVCacheDecodeOpBase(const AttentionConfigs& attn_configs);
    CKAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
    torch::Tensor
    forward(const torch::Tensor& qkv, std::optional<torch_ext::KVCache> kv_cache_base, const CKAttnPtr& params);

protected:
    AttentionConfigs attn_configs_;
    ROCmDevice*      device_;  // Only used for PrepareCKAttn
    virtual bool     use_asm() const = 0;
};

// ASM version of Decode operation
class FusedRopeKVCacheDecodeOpAsm: public FusedRopeKVCacheDecodeOpBase {
protected:
    bool use_asm() const override {
        return true;
    }

public:
    FusedRopeKVCacheDecodeOpAsm(const AttentionConfigs& attn_configs);
};

// Non-ASM version of Decode operation
class FusedRopeKVCacheDecodeOpNonAsm: public FusedRopeKVCacheDecodeOpBase {
protected:
    bool use_asm() const override {
        return false;
    }

public:
    FusedRopeKVCacheDecodeOpNonAsm(const AttentionConfigs& attn_configs);
};

void registerFusedRopeKVCacheOp(const py::module& m);

}  // namespace rtp_llm
