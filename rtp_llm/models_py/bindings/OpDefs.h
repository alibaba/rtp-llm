#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/utils/Logger.h"
namespace torch_ext {

struct KVCache {
    torch::Tensor kv_cache_base;
    torch::Tensor kv_scale_base;
    int           seq_size_per_block;
    int           layer_id = -1;
    KVCache       getLayerCache(int idx) {
        KVCache layer_cache;
        layer_cache.kv_cache_base      = kv_cache_base[idx];
        layer_cache.seq_size_per_block = seq_size_per_block;
        if (kv_scale_base.defined() && kv_scale_base.numel() > 0) {
            layer_cache.kv_scale_base = kv_scale_base[idx];
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
    size_t                   kv_block_stride_bytes;
    size_t                   kv_scale_stride_bytes;
    bool                     pd_separation   = false;
    size_t                   model_id        = 0;
    bool                     decode_entrance = false;
    bool                     warmup          = false;
    bool                     mla_kvcache     = false;
};

// for cuda grpah capture
struct PyCaptureMetaData {
    int capture_batch_size{1};
};

struct PyPrefillCudaGaphCopyParams {
    // for embedding model cuda graph capture, the attenton batch size is padded to max_batch_size,
    // so we can't get the real batch size for `copy kernel` using `input_lengths.size(0)`(which is max_batch_size).
    torch::Tensor cuda_graph_prefill_batch_size = torch::empty(0);
    int           max_seq_len                   = 0;
    int           max_batch_size                = 0;
};

struct PyAttentionInputs {
    bool             is_prefill{false};
    torch::Tensor    prefix_lengths;
    torch::Tensor    sequence_lengths;
    torch::Tensor    input_lengths;
    torch::Tensor    kv_cache_block_id_host;
    torch::Tensor    kv_cache_block_id_device;
    caffe2::TypeMeta dtype;
    // for `FusedRopeKVCacheDecodeOp`.
    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor decode_cu_seqlens_host;

    int           context_total_kv_length = 0;
    int           total_tokens            = 0;
    torch::Tensor padding_offset;

    // for write cache store
    std::optional<PyCacheStoreInputs> cache_store_inputs;

    std::optional<PyPrefillCudaGaphCopyParams> prefill_cuda_graph_copy_params;
    bool                                       is_s_padded = false;
    // deivce tensor
    torch::Tensor prefix_lengths_d;
    torch::Tensor sequence_lengths_plus_1_d;
    torch::Tensor input_lengths_d;
    torch::Tensor decode_cu_seqlens_d;

    // CUDA Graph mode flag
    bool is_cuda_graph = false;
};

struct BertEmbeddingInputs {
    torch::Tensor combo_position_ids;
    torch::Tensor position_encoding;
    torch::Tensor combo_tokens_type_ids;
    torch::Tensor token_type_embedding;
    float         input_embedding_scalar{1.0};
};

struct PyModelInputs {
    torch::Tensor       input_ids;
    torch::Tensor       input_hiddens;
    PyAttentionInputs   attention_inputs;
    BertEmbeddingInputs bert_embedding_inputs;
    bool                has_prefill_global{false};
};

struct PyModelOutputs {
    torch::Tensor          hidden_states;
    rtp_llm::ParamsBasePtr params_ptr{nullptr};
    py::object             py_attn_params{py::none()};

    PyModelOutputs() = default;

    // Constructor with default hidden_states
    PyModelOutputs(torch::Tensor hidden_states):
        hidden_states(std::move(hidden_states)), params_ptr(nullptr), py_attn_params(py::none()) {}

    PyModelOutputs(torch::Tensor                        hidden_states,
                   std::shared_ptr<rtp_llm::ParamsBase> params_ptr,
                   py::object                           py_params = py::none()):
        hidden_states(std::move(hidden_states)),
        params_ptr(std::move(params_ptr)),
        py_attn_params(std::move(py_params)) {}
};

void registerPyOpDefs(pybind11::module& m);

}  // namespace torch_ext
