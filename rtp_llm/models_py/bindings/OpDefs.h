#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/utils/Logger.h"
namespace torch_ext {

// Per-layer KV cache view. Returned by KVCache::getLayerCache().
// When kernel_seq_size_per_block < seq_size_per_block the tensor is presented at
// kernel-block granularity:
//   MHA: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
//   MLA: [kernel_block_num, kernel_seq_size_per_block, kv_lora_rank + rope_head_dim]
struct LayerKVCache {
    torch::Tensor kv_cache_base;
    torch::Tensor kv_scale_base;
    int           seq_size_per_block = 0;
    int           layer_id           = -1;
};

// Whole-model KV cache holding tensors for all layers.
// Call getLayerCache(global_layer_id) to obtain a per-layer LayerKVCache.
struct KVCache {
    // Per-layer views
    std::vector<torch::Tensor> kv_cache_base_by_layer;
    std::vector<torch::Tensor> kv_scale_base_by_layer;
    int                        seq_size_per_block        = 0;
    int                        kernel_seq_size_per_block = 0;
    int                        num_kv_heads              = 0;
    int                        head_dim                  = 0;
    bool                       use_mla                   = false;
    int                        kv_lora_rank              = 0;
    int                        rope_head_dim             = 0;

    // Per-layer attention type (CacheGroupType::FULL or LINEAR).
    std::vector<rtp_llm::CacheGroupType> layer_attn_types;

    LayerKVCache getLayerCache(int idx) {
        LayerKVCache layer_cache;
        layer_cache.layer_id = idx;

        // Determine whether this layer is a full-attention layer.
        if (idx < 0 || static_cast<size_t>(idx) >= layer_attn_types.size())
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        auto          base = kv_cache_base_by_layer[idx];
        torch::Tensor scale;
        if (!kv_scale_base_by_layer.empty()) {
            scale = kv_scale_base_by_layer[idx];
        }

        const bool is_full = layer_attn_types[static_cast<size_t>(idx)] == rtp_llm::CacheGroupType::FULL;

        if (!is_full) {
            // Linear/SSM attention layer: return the raw cache tensor unchanged.
            // Use the physical block size so the layer sees the full per-block storage.
            layer_cache.seq_size_per_block = seq_size_per_block;
            layer_cache.kv_cache_base      = base;
            layer_cache.kv_scale_base      = scale;
        } else {
            layer_cache.seq_size_per_block =
                kernel_seq_size_per_block > 0 ? kernel_seq_size_per_block : seq_size_per_block;
            const int64_t kernel_blocks_per_kv_block =
                kernel_seq_size_per_block > 0 ? (int64_t)seq_size_per_block / (int64_t)kernel_seq_size_per_block : 1;

            // [block_num, kv_block_stride_elems] shared by all layer types.
            if (base.defined() && base.dim() == 2) {
                const int64_t physical_block_num = base.size(0);
                const int64_t kernel_block_num   = physical_block_num * kernel_blocks_per_kv_block;
                if (use_mla && kv_lora_rank > 0 && rope_head_dim > 0) {
                    // MLA layout: [kernel_block_num, kernel_seq_size_per_block, kv_lora_rank + rope_head_dim]
                    layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                                              (int64_t)kernel_seq_size_per_block,
                                                              (int64_t)(kv_lora_rank + rope_head_dim)});
                } else if (num_kv_heads > 0 && head_dim > 0) {
                    // MHA layout: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
                    layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                                              2,
                                                              (int64_t)num_kv_heads,
                                                              (int64_t)kernel_seq_size_per_block,
                                                              (int64_t)head_dim});
                } else {
                    layer_cache.kv_cache_base = base;
                }
            } else {
                layer_cache.kv_cache_base = base;
            }

            if (scale.defined()) {
                // Keep kv_scale_base aligned with kernel-block view of kv_cache_base.
                const int64_t physical_block_num = base.size(0);
                const int64_t kernel_block_num   = physical_block_num * kernel_blocks_per_kv_block;

                if (use_mla) {
                    layer_cache.kv_scale_base =
                        scale.reshape({kernel_block_num, (int64_t)kernel_seq_size_per_block, scale.size(2)});
                } else {
                    layer_cache.kv_scale_base =
                        scale.reshape({kernel_block_num, scale.size(1) / kernel_blocks_per_kv_block});
                }
            }
        }
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
    torch::Tensor            kv_cache_layer_to_group;
    torch::Tensor            kv_cache_group_types;
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

struct PyContextParallelParams {
    torch::Tensor prefill_cp_padding_lengths;
    torch::Tensor prefill_cp_chunk_lengths;
    torch::Tensor prefill_shuffle_indices;
    torch::Tensor prefill_qkv_restore_indice;
    torch::Tensor prefill_qkv_padding_mask;
    torch::Tensor prefill_actual_input_lengths_cpu;
};

struct PyAttentionInputs {
    bool          is_prefill{false};
    bool          is_target_verify{false};
    torch::Tensor prefix_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor input_lengths;
    // Kernel-granularity block IDs for attention compute.
    // Shape: [group, batch, max_kernel_blocks] or [batch, max_kernel_blocks].
    torch::Tensor kv_cache_kernel_block_id_host;
    torch::Tensor kv_cache_kernel_block_id_device;
    // Physical block IDs dedicated for cache store.
    // Shape: [group, batch, max_blocks] or [batch, max_blocks].
    torch::Tensor kv_cache_block_id_host;
    torch::Tensor kv_cache_block_id_device;
    // Hybrid cache support:
    // - kv_cache_kernel_block_id_*_by_group: vector of 2-D kernel block tables, each [batch, max_kernel_blocks].
    std::vector<torch::Tensor> kv_cache_kernel_block_id_host_by_group;
    std::vector<torch::Tensor> kv_cache_kernel_block_id_device_by_group;
    torch::Tensor              kv_cache_layer_to_group;
    caffe2::TypeMeta           dtype;
    // for `FusedRopeKVCacheDecodeOp`.
    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor decode_cu_seqlens_host;

    int           context_total_kv_length = 0;
    int           total_tokens            = 0;
    torch::Tensor padding_offset;
    torch::Tensor position_ids;

    // for write cache store
    std::optional<PyCacheStoreInputs> cache_store_inputs;

    std::optional<PyPrefillCudaGaphCopyParams> prefill_cuda_graph_copy_params;
    bool                                       is_s_padded = false;
    // deivce tensor
    torch::Tensor prefix_lengths_d;
    torch::Tensor sequence_lengths_plus_1_d;
    torch::Tensor input_lengths_d;
    torch::Tensor decode_cu_seqlens_d;

    // CUDA Graph mode flags
    bool is_cuda_graph = false;  // True when running in CUDA graph mode (capture or replay)

    std::optional<PyContextParallelParams> context_parallel_info;
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
