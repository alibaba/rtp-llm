#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/utils/Logger.h"
namespace torch_ext {

struct KVCache {
    torch::Tensor k_cache_base;
    torch::Tensor v_cache_base;
    torch::Tensor k_scale_base;
    torch::Tensor v_scale_base;
    int           layer_id = -1;
    KVCache       getLayerCache(int idx) {
        KVCache layer_cache;
        layer_cache.k_cache_base = k_cache_base[idx];
        layer_cache.v_cache_base = v_cache_base[idx];
        if (k_scale_base.defined() && k_scale_base.numel() > 0) {
            layer_cache.k_scale_base = k_scale_base[idx];
            layer_cache.v_scale_base = v_scale_base[idx];
        }
        layer_cache.layer_id = idx;
        return layer_cache;
    }
};

struct PyModelInitResources {
    std::optional<KVCache> kv_cache;
};

struct PyCacheStoreInputs {
    size_t                   context_batch_size = 0;
    size_t                   decoder_batch_size = 0;
    torch::Tensor            request_id;
    torch::Tensor            request_pd_separation;
    std::vector<std::string> cache_keys;  // [context_batch_size]
    size_t                   tokens_per_block;
    size_t                   k_block_size;
    size_t                   v_block_size;
    size_t                   scale_block_size;
    bool                     pd_separation   = false;
    size_t                   model_id        = 0;
    bool                     decode_entrance = false;
    bool                     warmup          = false;
    bool                     mla_kvcache     = false;
};

struct PyAttentionInputs {
    bool             is_prefill;
    torch::Tensor    prefix_lengths;
    torch::Tensor    sequence_lengths;
    torch::Tensor    input_lengths;
    torch::Tensor    kv_cache_block_id_host;
    torch::Tensor    kv_cache_block_id_device;
    caffe2::TypeMeta dtype;
    int              kv_block_offset = 0;
    // for `FusedRopeKVCacheDecodeOp`.
    torch::Tensor cu_seqlens;
    torch::Tensor padding_offset;

    // for write cache store
    std::optional<PyCacheStoreInputs> cache_store_inputs;
};

struct PyModelInputs {
    torch::Tensor     input_ids;
    PyAttentionInputs attention_inputs;
};

struct PyModelOutputs {
    torch::Tensor          hidden_states;
    rtp_llm::ParamsBasePtr params_ptr{nullptr};

    PyModelOutputs() = default;
    PyModelOutputs(torch::Tensor hidden_states, std::shared_ptr<rtp_llm::ParamsBase> params_ptr):
        hidden_states(std::move(hidden_states)), params_ptr(std::move(params_ptr)) {}

    // Constructor with default values
    PyModelOutputs(torch::Tensor hidden_states): hidden_states(std::move(hidden_states)), params_ptr(nullptr) {}

    // Constructor with default hidden_states
    PyModelOutputs(std::shared_ptr<rtp_llm::ParamsBase> params_ptr):
        hidden_states(torch::Tensor()), params_ptr(std::move(params_ptr)) {}
};

void registerPyOpDefs(pybind11::module& m);

}  // namespace torch_ext
