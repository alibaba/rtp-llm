#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

// Forward declare for opaque pointers in PyCacheStoreInputs
namespace rtp_llm {
class CacheStore;
class CacheStoreAsyncWriter;
}  // namespace rtp_llm

namespace torch_ext {

// Per-layer KV cache view. Returned by KVCache::getLayerCache().
// When kernel_seq_size_per_block < seq_size_per_block the tensor is presented at
// kernel-block granularity:
//   MHA: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
//   MLA: [kernel_block_num, kernel_seq_size_per_block, kv_lora_rank + rope_head_dim]
struct LayerKVCache {
    torch::Tensor              kv_cache_base;
    torch::Tensor              kv_scale_base;
    int                        seq_size_per_block = 0;
    int                        layer_id           = -1;
    int                        group_id           = -1;
    std::string                tag;
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
    std::vector<rtp_llm::CacheGroupType>    layer_group_types;
    std::vector<rtp_llm::CacheGroupType>    group_types;
    std::vector<std::string>                group_tags;
    std::vector<int>                        group_seq_size_per_block;
    std::vector<std::vector<int>>           layer_to_group_ids;
    std::vector<std::map<std::string, int>> layer_tag_to_group_id;
    std::vector<std::vector<torch::Tensor>> kv_cache_base_by_layer_group;
    std::vector<std::vector<torch::Tensor>> kv_scale_base_by_layer_group;

    LayerKVCache getLayerCache(int idx) {
        LayerKVCache layer_cache;
        layer_cache.layer_id = idx;

        // Determine whether this layer is a full-attention layer.
        if (idx < 0 || static_cast<size_t>(idx) >= layer_group_types.size())
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        auto          base = kv_cache_base_by_layer[idx];
        torch::Tensor scale;
        if (!kv_scale_base_by_layer.empty()) {
            scale = kv_scale_base_by_layer[idx];
        }

        const bool is_full = layer_group_types[static_cast<size_t>(idx)] == rtp_llm::CacheGroupType::FULL;

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
        const auto layer = static_cast<size_t>(idx);
        if (!layer_to_group_ids.empty() && layer < layer_to_group_ids.size()
            && layer_to_group_ids[layer].size() == 1) {
            layer_cache.group_id = layer_to_group_ids[layer].front();
        } else {
            layer_cache.group_id = 0;
        }
        if (layer_cache.group_id >= 0 && static_cast<size_t>(layer_cache.group_id) < group_tags.size()) {
            layer_cache.tag = group_tags[static_cast<size_t>(layer_cache.group_id)];
        }
        return layer_cache;
    }

    int groupSeqSizePerBlock(int group_id) const {
        if (group_id >= 0 && static_cast<size_t>(group_id) < group_seq_size_per_block.size()
            && group_seq_size_per_block[static_cast<size_t>(group_id)] > 0) {
            return group_seq_size_per_block[static_cast<size_t>(group_id)];
        }
        return seq_size_per_block;
    }

    LayerKVCache getLayerCacheByGroup(int idx, int gid) {
        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= kv_cache_base_by_layer_group.size()) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        }
        if (gid < 0 || static_cast<size_t>(gid) >= kv_cache_base_by_layer_group[layer].size()) {
            throw std::runtime_error("Invalid KV cache group id: " + std::to_string(gid));
        }
        if (!layer_to_group_ids.empty()) {
            if (layer >= layer_to_group_ids.size()
                || std::find(layer_to_group_ids[layer].begin(), layer_to_group_ids[layer].end(), gid)
                       == layer_to_group_ids[layer].end()) {
                throw std::runtime_error("Layer " + std::to_string(idx) + " does not own KV cache group "
                                         + std::to_string(gid));
            }
        }

        auto base = kv_cache_base_by_layer_group[layer][static_cast<size_t>(gid)];
        if (!base.defined()) {
            throw std::runtime_error("Missing KV cache tensor for layer " + std::to_string(idx) + ", group "
                                     + std::to_string(gid));
        }

        LayerKVCache layer_cache;
        layer_cache.layer_id      = idx;
        layer_cache.group_id      = gid;
        if (static_cast<size_t>(gid) < group_tags.size()) {
            layer_cache.tag = group_tags[static_cast<size_t>(gid)];
        }
        const bool is_full_group = gid >= 0 && static_cast<size_t>(gid) < group_types.size()
                                       && group_types[static_cast<size_t>(gid)] == rtp_llm::CacheGroupType::FULL;
        layer_cache.seq_size_per_block =
            is_full_group && kernel_seq_size_per_block > 0 ? kernel_seq_size_per_block :
                                                             groupSeqSizePerBlock(layer_cache.group_id);
        layer_cache.kv_cache_base = base;
        if (!kv_scale_base_by_layer_group.empty() && layer < kv_scale_base_by_layer_group.size()
            && static_cast<size_t>(gid) < kv_scale_base_by_layer_group[layer].size()) {
            layer_cache.kv_scale_base = kv_scale_base_by_layer_group[layer][static_cast<size_t>(gid)];
        }
        return layer_cache;
    }

    LayerKVCache getLayerCache(int idx, const std::string& tag) {
        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= layer_tag_to_group_id.size()) {
            throw std::runtime_error("Invalid layer index for cache tag lookup: " + std::to_string(idx));
        }
        const auto it = layer_tag_to_group_id[layer].find(tag);
        if (it == layer_tag_to_group_id[layer].end() || it->second < 0) {
            throw std::runtime_error("Layer " + std::to_string(idx) + " does not own KV cache tag " + tag);
        }
        const int gid = it->second;
        if (gid < 0 || static_cast<size_t>(gid) >= group_tags.size()) {
            throw std::runtime_error("KV cache tag " + tag + " maps to invalid group " + std::to_string(gid));
        }
        return getLayerCacheByGroup(idx, gid);
    }

    std::vector<LayerKVCache> getLayerCaches(int idx) {
        if (layer_to_group_ids.empty() || group_tags.empty()) {
            return {getLayerCache(idx)};
        }
        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= layer_to_group_ids.size()) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        }

        std::vector<LayerKVCache> layer_caches;
        for (int gid : layer_to_group_ids[layer]) {
            layer_caches.push_back(getLayerCacheByGroup(idx, gid));
        }
        return layer_caches;
    }

    // Return raw [total_blocks, stride_bytes] tensor for a specific pool,
    // without MHA reshape. Used by DSV4 gather/scatter which needs raw uint8 access.
    torch::Tensor getRawPoolTensor(int layer_id, const std::string& tag) {
        if (kv_cache_base_by_layer_group.empty()) {
            throw std::runtime_error("kv_cache_base_by_layer_group is empty");
        }
        return getLayerCache(layer_id, tag).kv_cache_base;
    }
};

struct PyModelInitResources {
    std::optional<KVCache> kv_cache;
    bool                   is_speculative         = false;
    bool                   is_decode_role         = false;
    int64_t                max_context_batch_size = 1;
};

struct PyCacheStoreInputs {
    size_t                   context_batch_size = 0;
    size_t                   decoder_batch_size = 0;
    torch::Tensor            request_id;
    torch::Tensor            request_pd_separation;
    torch::Tensor            kv_cache_group_types;
    std::vector<std::string> cache_keys;  // [context_batch_size]
    // Pinned-host mirrors of device length tensors for cache store consumption.
    // Populated via non-blocking D2H in prepareWriteCacheParams so that
    // background cache-store threads never issue a synchronous .cpu() copy.
    torch::Tensor input_lengths_host;
    torch::Tensor prefix_lengths_host;
    size_t        tokens_per_block;
    size_t        kv_block_stride_bytes;
    size_t        kv_scale_stride_bytes;
    bool          pd_separation             = false;
    size_t        model_id                  = 0;
    bool          decode_entrance           = false;
    bool          warmup                    = false;
    bool          use_opaque_kv_cache_store = false;
    bool          mla_kvcache               = false;

    // Opaque cache_store reference (C++ only; passes through Python without inspection)
    std::shared_ptr<rtp_llm::CacheStore> cache_store;
    rtp_llm::CacheStoreAsyncWriter*      cache_store_async_writer = nullptr;

    // CP-page-RR sharding context. (1, 0) = no sharding (legacy path).
    int cp_size = 1;
    int cp_rank = 0;
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
    // Hybrid cache support: per-group CUDA kernel block tables.
    std::vector<torch::Tensor> kv_cache_kernel_block_id_device_by_group;
    caffe2::TypeMeta           dtype;
    // Cumulative sequence lengths for attention kernels (e.g. FusedRopeKVCacheDecodeOp).
    // cu_seqlens lives on CUDA device; cu_seqlens_host is its pinned-memory CPU mirror
    // used for CUDA graph replay (write host → async copy to device, avoiding GPU-side fills).
    torch::Tensor cu_seqlens;
    torch::Tensor cu_seqlens_host;
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
    torch::Tensor                              sequence_lengths_plus_1_d;
    torch::Tensor                              decode_cu_seqlens_d;

    // CUDA Graph mode flags
    bool is_cuda_graph = false;  // True when running in CUDA graph mode (capture or replay)

    std::optional<PyContextParallelParams> context_parallel_info;

    // Headwise attention config (Python dict or None).
    py::object headwise_config{py::none()};
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
