#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/utils/AttentionConfig.h"

namespace torch_ext {

struct KVCache {
    torch::Tensor k_cache_base;
    torch::Tensor v_cache_base;
    torch::Tensor k_scale_base;
    torch::Tensor v_scale_base;

    KVCache getLayerCache(int idx) {
        KVCache layer_cache;
        layer_cache.k_cache_base = k_cache_base[idx];
        layer_cache.v_cache_base = v_cache_base[idx];
        if (k_scale_base.defined() && k_scale_base.numel() > 0) {
            layer_cache.k_scale_base = k_scale_base[idx];
            layer_cache.v_scale_base = v_scale_base[idx];
        }
        return layer_cache;
    }
};

struct PyModelInitResources {
    KVCache kv_cache;
};

struct PyAttentionInputs {
    bool             is_prefill;
    torch::Tensor    prefix_lengths;
    torch::Tensor    sequence_lengths;
    torch::Tensor    input_lengths;
    torch::Tensor    kv_cache_block_id_host;
    torch::Tensor    kv_cache_block_id_device;
    caffe2::TypeMeta dtype;
    int              kv_block_offset;
    // for `FusedRopeKVCacheDecodeOp`.
    torch::Tensor cu_seqlens;
};

struct PyModelInputs {
    torch::Tensor     input_ids;
    PyAttentionInputs attention_inputs;
};

struct PyModelOutputs {
    torch::Tensor hidden_states;
};

void registerPyOpDefs(pybind11::module& m);

}  // namespace torch_ext
