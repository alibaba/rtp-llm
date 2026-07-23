#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
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
//   MLA: [kernel_block_num, kernel_seq_size_per_block, physical_elements_per_token]
struct LayerKVCache {
    torch::Tensor kv_cache_base;
    torch::Tensor kv_scale_base;
    int           seq_size_per_block = 0;
    int           layer_id           = -1;
    std::string   tag                = "default";

    LayerKVCache() = default;

    LayerKVCache(torch::Tensor kv_cache_base,
                 int           seq_size_per_block,
                 int           layer_id      = -1,
                 std::string   tag           = "default",
                 torch::Tensor kv_scale_base = {}):
        kv_cache_base(std::move(kv_cache_base)),
        kv_scale_base(std::move(kv_scale_base)),
        seq_size_per_block(seq_size_per_block),
        layer_id(layer_id),
        tag(std::move(tag)) {}
};

// Whole-model KV cache holding tensors for all layers.
// Call getLayerCache(global_layer_id) to obtain a per-layer LayerKVCache.
class KVCache {
public:
    explicit KVCache(rtp_llm::GroupedCacheLayerLayout grouped_layout): grouped_layout_(std::move(grouped_layout)) {}

    LayerKVCache getLayerCache(int layer_id) const {
        validateLayer(layer_id);
        const auto& group = grouped_layout_.topology().soleGroupForLayer(layer_id);
        return getLayerCache(layer_id, group.tag);
    }

    LayerKVCache getLayerCache(int layer_id, const std::string& tag) const {
        validateLayer(layer_id);
        const auto& group        = grouped_layout_.topology().groupForLayer(layer_id, tag);
        const auto& group_layout = grouped_layout_.group(tag);
        const auto  layer        = static_cast<size_t>(layer_id);
        if (group_layout.empty() || !group_layout.hasLayer(layer)) {
            throw std::runtime_error("Layer " + std::to_string(layer_id) + " has no KV cache tensor for tag " + tag);
        }
        return makeLayerCache(layer_id, group, group_layout.at(layer));
    }

    std::vector<LayerKVCache> getLayerCacheGroups(int layer_id) const {
        validateLayer(layer_id);
        const auto  layer = static_cast<size_t>(layer_id);
        const auto& tags  = grouped_layout_.topology().layer(layer_id).group_tags;

        std::vector<LayerKVCache> layer_caches;
        layer_caches.reserve(tags.size());
        for (const auto& tag : tags) {
            const auto& group_layout = grouped_layout_.group(tag);
            if (group_layout.empty() || !group_layout.hasLayer(layer)) {
                continue;
            }
            layer_caches.push_back(getLayerCache(layer_id, tag));
        }
        return layer_caches;
    }

    const std::vector<std::string>& groupTags() const {
        return grouped_layout_.topology().groupTagsSnapshot();
    }

    size_t layerCount() const {
        return grouped_layout_.topology().layers().size();
    }

    int getSeqSizePerBlock(const std::string& tag) const {
        return static_cast<int>(grouped_layout_.topology().group(tag).seq_size_per_block);
    }

    int getKernelSeqSizePerBlock(const std::string& tag) const {
        return static_cast<int>(grouped_layout_.topology().group(tag).kernel_seq_size_per_block);
    }

private:
    void validateLayer(int layer_id) const {
        if (layer_id < 0 || static_cast<size_t>(layer_id) >= layerCount()) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(layer_id));
        }
    }

    static int64_t kernelBlocksPerPhysicalBlock(const rtp_llm::GroupBase& group) {
        RTP_LLM_CHECK_WITH_INFO(group.kernel_seq_size_per_block > 0
                                    && group.seq_size_per_block % group.kernel_seq_size_per_block == 0,
                                "invalid block subdivision for tag=%s physical=%zu kernel=%zu",
                                group.tag.c_str(),
                                group.seq_size_per_block,
                                group.kernel_seq_size_per_block);
        return static_cast<int64_t>(group.seq_size_per_block / group.kernel_seq_size_per_block);
    }

    static torch::Tensor reshapeMlaTensor(const torch::Tensor& tensor,
                                          int64_t              physical_block_num,
                                          int64_t              kernel_block_num,
                                          int64_t              kernel_seq_size,
                                          const char*          tensor_name,
                                          const std::string&   tag) {
        if (!tensor.defined()) {
            return {};
        }
        RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s for tag=%s must be contiguous", tensor_name, tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(tensor.dim() > 0 && tensor.size(0) == physical_block_num,
                                "%s physical block count mismatch for tag=%s: got=%ld expected=%ld",
                                tensor_name,
                                tag.c_str(),
                                tensor.dim() > 0 ? tensor.size(0) : -1,
                                physical_block_num);
        const int64_t page_elements = kernel_block_num * kernel_seq_size;
        RTP_LLM_CHECK_WITH_INFO(page_elements > 0 && tensor.numel() % page_elements == 0,
                                "%s elements=%ld must be divisible by kernel page elements=%ld for tag=%s",
                                tensor_name,
                                tensor.numel(),
                                page_elements,
                                tag.c_str());
        return tensor.view({kernel_block_num, kernel_seq_size, tensor.numel() / page_elements});
    }

    LayerKVCache
    makeLayerCache(int layer_id, const rtp_llm::GroupBase& group, const rtp_llm::BlockBufferPtrInfo& buffers) const {
        RTP_LLM_CHECK_WITH_INFO(buffers.kv_addr.defined(),
                                "KV cache tensor must be defined for layer=%d tag=%s",
                                layer_id,
                                group.tag.c_str());

        LayerKVCache result(
            buffers.kv_addr, static_cast<int>(group.seq_size_per_block), layer_id, group.tag, buffers.kv_scale_addr);

        const auto spec_type = group.spec->type;
        if (group.policy.group_type != rtp_llm::CacheGroupType::FULL
            || spec_type == rtp_llm::KVCacheSpecType::LinearAttention
            || spec_type == rtp_llm::KVCacheSpecType::OpaqueState) {
            return result;
        }

        const int64_t physical_block_num  = buffers.kv_addr.size(0);
        const int64_t blocks_per_physical = kernelBlocksPerPhysicalBlock(group);
        const int64_t kernel_block_num    = physical_block_num * blocks_per_physical;
        const int64_t kernel_seq_size     = static_cast<int64_t>(group.kernel_seq_size_per_block);
        result.seq_size_per_block         = static_cast<int>(kernel_seq_size);

        if (spec_type == rtp_llm::KVCacheSpecType::MultiHeadAttention) {
            const int64_t local_kv_heads = static_cast<int64_t>(group.local_kv_head_num);
            RTP_LLM_CHECK_WITH_INFO(local_kv_heads > 0, "MHA tag=%s has no local KV heads", group.tag.c_str());
            const int64_t physical_seq_size = static_cast<int64_t>(group.seq_size_per_block);
            const int64_t k_block_elems     = static_cast<int64_t>(group.spec->k_block_size());
            RTP_LLM_CHECK_WITH_INFO(k_block_elems > 0 && k_block_elems % (local_kv_heads * physical_seq_size) == 0,
                                    "MHA tag=%s cannot derive head dimension from k_block_size=%ld heads=%ld seq=%ld",
                                    group.tag.c_str(),
                                    k_block_elems,
                                    local_kv_heads,
                                    physical_seq_size);
            const int64_t head_dim = k_block_elems / (local_kv_heads * physical_seq_size);
            RTP_LLM_CHECK_WITH_INFO(
                buffers.kv_addr.is_contiguous(), "MHA KV cache base for tag=%s must be contiguous", group.tag.c_str());
            const int64_t expected_numel = kernel_block_num * 2 * local_kv_heads * kernel_seq_size * head_dim;
            RTP_LLM_CHECK_WITH_INFO(buffers.kv_addr.numel() == expected_numel,
                                    "MHA KV cache elements=%ld expected=%ld for layer=%d tag=%s",
                                    buffers.kv_addr.numel(),
                                    expected_numel,
                                    layer_id,
                                    group.tag.c_str());
            result.kv_cache_base =
                buffers.kv_addr.view({kernel_block_num, 2, local_kv_heads, kernel_seq_size, head_dim});
            if (buffers.kv_scale_addr.defined()) {
                RTP_LLM_CHECK_WITH_INFO(buffers.kv_scale_addr.is_contiguous() && buffers.kv_scale_addr.dim() > 0
                                            && buffers.kv_scale_addr.size(0) == physical_block_num
                                            && buffers.kv_scale_addr.numel() % kernel_block_num == 0,
                                        "MHA scale tensor cannot be expanded for layer=%d tag=%s",
                                        layer_id,
                                        group.tag.c_str());
                result.kv_scale_base =
                    buffers.kv_scale_addr.view({kernel_block_num, buffers.kv_scale_addr.numel() / kernel_block_num});
            }
            return result;
        }

        if (spec_type == rtp_llm::KVCacheSpecType::MultiHeadLatentAttention) {
            result.kv_cache_base = reshapeMlaTensor(buffers.kv_addr,
                                                    physical_block_num,
                                                    kernel_block_num,
                                                    kernel_seq_size,
                                                    "MLA KV cache tensor",
                                                    group.tag);
            result.kv_scale_base = reshapeMlaTensor(buffers.kv_scale_addr,
                                                    physical_block_num,
                                                    kernel_block_num,
                                                    kernel_seq_size,
                                                    "MLA scale/indexer tensor",
                                                    group.tag);
            return result;
        }

        if (spec_type == rtp_llm::KVCacheSpecType::OpaqueKV) {
            RTP_LLM_CHECK_WITH_INFO(buffers.kv_addr.is_contiguous() && buffers.kv_addr.numel() % kernel_block_num == 0,
                                    "opaque KV cache cannot be expanded for layer=%d tag=%s",
                                    layer_id,
                                    group.tag.c_str());
            result.kv_cache_base = buffers.kv_addr.view({kernel_block_num, buffers.kv_addr.numel() / kernel_block_num});
        }
        return result;
    }

    const rtp_llm::GroupedCacheLayerLayout grouped_layout_;
};

struct PyModelInitResources {
    std::optional<KVCache> kv_cache;
    bool                   is_speculative         = false;
    bool                   is_decode_role         = false;
    int64_t                max_context_batch_size = 1;
};

struct PyCacheStoreInputs {
    size_t                                           context_batch_size = 0;
    size_t                                           decoder_batch_size = 0;
    torch::Tensor                                    request_id;
    torch::Tensor                                    request_pd_separation;
    std::map<std::string, rtp_llm::CacheGroupPolicy> kv_cache_group_policies;
    std::map<std::string, size_t>                    tokens_per_block_by_tag;
    // Physical address step and logical transfer length are different for a
    // shared pool: blocks are max-group-stride apart, while each tag transfers
    // only the bytes described by its own cache group.
    std::map<std::string, size_t> kv_block_stride_bytes_by_tag;
    std::map<std::string, size_t> kv_scale_stride_bytes_by_tag;
    std::map<std::string, size_t> kv_block_transfer_bytes_by_tag;
    std::map<std::string, size_t> kv_scale_transfer_bytes_by_tag;
    std::vector<std::string>      cache_keys;  // [context_batch_size]
    size_t                        tokens_per_block = 0;
    // Physical KV-manager block strides, supplied by CacheConfig rather than inferred from tensor views.
    size_t kv_block_stride_bytes     = 0;
    size_t kv_scale_stride_bytes     = 0;
    bool   pd_separation             = false;
    size_t model_id                  = 0;
    bool   decode_entrance           = false;
    bool   warmup                    = false;
    bool   use_opaque_kv_cache_store = false;
    bool   mla_kvcache               = false;

    // Cache store reference (C++ only; passes through Python without inspection)
    std::shared_ptr<rtp_llm::CacheStore> cache_store;
    rtp_llm::CacheStoreAsyncWriter*      cache_store_async_writer = nullptr;

    // CP-page-RR sharding context. (1, 0) = no sharding.
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

// Naming convention: the host (pinned CPU) tensor uses the bare name; its device (CUDA)
// counterpart carries a _device suffix.
struct PyAttentionInputs {
    bool          is_prefill{false};
    bool          is_target_verify{false};
    torch::Tensor prefix_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor input_lengths;
    // Group-local kernel-granularity block IDs for attention compute.
    // Shape: [batch, max_kernel_blocks].
    torch::Tensor kv_cache_kernel_block_id;
    torch::Tensor kv_cache_kernel_block_id_device;
    // Group-local physical block IDs dedicated for cache store.
    // Shape: [batch, max_blocks].
    torch::Tensor    kv_cache_block_id;
    torch::Tensor    kv_cache_block_id_device;
    caffe2::TypeMeta dtype;
    // Cumulative sequence lengths for attention kernels (e.g. FusedRopeKVCacheDecodeOp).
    // cu_seqlens_device lives on CUDA device; cu_seqlens is its pinned-memory CPU mirror
    // used for CUDA graph replay (write host -> async copy to device, avoiding GPU-side fills).
    torch::Tensor cu_seqlens;
    torch::Tensor cu_seqlens_device;
    torch::Tensor cu_kv_seqlens_device;  // device only (no host mirror needed)
    torch::Tensor decode_cu_seqlens;
    int           context_total_kv_length = 0;
    int           total_tokens            = 0;
    torch::Tensor padding_offset;
    torch::Tensor combo_position_ids;

    // for write cache store
    std::optional<PyCacheStoreInputs> cache_store_inputs;

    std::optional<PyPrefillCudaGaphCopyParams> prefill_cuda_graph_copy_params;
    bool                                       is_s_padded = false;
    // Device-side mirrors of host tensors, managed by C++ for fused D2D copy in CUDA graph.
    torch::Tensor prefix_lengths_device;
    torch::Tensor sequence_lengths_plus_1_device;
    torch::Tensor input_lengths_device;
    torch::Tensor decode_cu_seqlens_device;

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

struct PyEmbeddingInputs {
    torch::Tensor combo_tokens_type_ids;
    torch::Tensor text_tokens_mask;
};

struct PyMultimodalInputs {
    std::vector<torch::Tensor> multimodal_features;
    torch::Tensor              mm_features_locs;
    std::vector<torch::Tensor> mm_extra_input;
};

using AttentionInputsByTag = std::map<std::string, PyAttentionInputs>;

struct PyModelInputs {
    torch::Tensor      input_ids;
    torch::Tensor      input_hiddens;
    torch::Tensor      combo_position_ids;
    PyEmbeddingInputs  embedding_inputs;
    PyMultimodalInputs multimodal_inputs;
    // C++ common/single-group fast path. Python sees this field through a
    // property which returns either this object or attention_inputs_by_tag.
    PyAttentionInputs    attention_inputs;
    AttentionInputsByTag attention_inputs_by_tag;
    BertEmbeddingInputs  bert_embedding_inputs;

    bool hasAttentionInputsByTag() const {
        return !attention_inputs_by_tag.empty();
    }
};

struct PyModelOutputs {
    torch::Tensor hidden_states;

    PyModelOutputs() = default;

    PyModelOutputs(torch::Tensor hidden_states): hidden_states(std::move(hidden_states)) {}
};

void registerPyOpDefs(pybind11::module& m);

}  // namespace torch_ext
