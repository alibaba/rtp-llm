#include "OpDefs.h"

namespace torch_ext {
namespace {

void registerCacheGroupType(pybind11::module& m) {
    try {
        auto config_m            = pybind11::module_::import("libth_transformer_config");
        m.attr("CacheGroupType") = config_m.attr("CacheGroupType");
        return;
    } catch (const pybind11::error_already_set& e) {
        PyErr_Clear();
    }

    pybind11::enum_<rtp_llm::CacheGroupType>(m, "CacheGroupType")
        .value("LINEAR", rtp_llm::CacheGroupType::LINEAR)
        .value("FULL", rtp_llm::CacheGroupType::FULL)
        .value("SWA", rtp_llm::CacheGroupType::SWA)
        .export_values();
}

}  // namespace

void registerPyOpDefs(pybind11::module& m) {
    registerCacheGroupType(m);

    pybind11::class_<LayerKVCache>(m, "LayerKVCache")
        .def(pybind11::init<>())
        .def(
            pybind11::init([](torch::Tensor    kv_cache_base,
                              int              seq_size_per_block,
                              int              layer_id,
                              int              group_id,
                              std::string      tag,
                              pybind11::object kv_scale_base) {
                torch::Tensor scale;
                if (!kv_scale_base.is_none()) {
                    scale = kv_scale_base.cast<torch::Tensor>();
                }
                return LayerKVCache(
                    std::move(kv_cache_base), seq_size_per_block, layer_id, group_id, std::move(tag), std::move(scale));
            }),
            pybind11::arg("kv_cache_base"),
            pybind11::arg("seq_size_per_block"),
            pybind11::arg("layer_id")      = -1,
            pybind11::arg("group_id")      = -1,
            pybind11::arg("tag")           = "default",
            pybind11::arg("kv_scale_base") = pybind11::none())
        .def_readwrite("kv_cache_base", &LayerKVCache::kv_cache_base, "Key/value cache tensor (per-layer view)")
        .def_readwrite("kv_scale_base", &LayerKVCache::kv_scale_base, "Key/value cache scale tensor")
        .def_readonly("seq_size_per_block", &LayerKVCache::seq_size_per_block, "Sequence size per block")
        .def_readonly("layer_id", &LayerKVCache::layer_id, "Global layer id")
        .def_readonly("group_id", &LayerKVCache::group_id, "Cache group id (-1 = default)")
        .def_readonly("tag", &LayerKVCache::tag, "Cache group tag");

    pybind11::class_<KVCache>(m, "KVCache")
        .def_property_readonly("group_tags", &KVCache::groupTags, "Cache group tags in topology slot order")
        .def_property_readonly("layer_count", &KVCache::layerCount, "Number of model-local cache layers")
        .def("get_layer_cache",
             static_cast<LayerKVCache (KVCache::*)(int) const>(&KVCache::getLayerCache),
             "Return a per-layer LayerKVCache for the given global layer id")
        .def("get_layer_cache",
             static_cast<LayerKVCache (KVCache::*)(int, const std::string&) const>(&KVCache::getLayerCache),
             "Return a LayerKVCache for the given layer and tag")
        .def("get_layer_cache_by_group",
             &KVCache::getLayerCacheByGroup,
             "Compatibility accessor using a CacheTopology slot")
        .def("get_layer_cache_groups",
             &KVCache::getLayerCacheGroups,
             "Return every valid LayerKVCache group owned by the layer")
        .def("get_seq_size_per_block",
             &KVCache::getSeqSizePerBlock,
             "Return the physical sequence size per block for a cache tag")
        .def("get_kernel_seq_size_per_block",
             &KVCache::getKernelSeqSizePerBlock,
             "Return the kernel sequence size per block for a cache tag");

    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readonly("kv_cache", &PyModelInitResources::kv_cache, "KV cache for all layers")
        .def_readonly("is_speculative", &PyModelInitResources::is_speculative)
        .def_readonly("is_decode_role", &PyModelInitResources::is_decode_role)
        .def_readonly("max_context_batch_size", &PyModelInitResources::max_context_batch_size);

    pybind11::class_<caffe2::TypeMeta>(m, "TypeMeta").def(pybind11::init<>());

    m.def(
        "get_typemeta",
        [](const torch::Tensor& tensor) { return torch::scalarTypeToTypeMeta(tensor.scalar_type()); },
        "Convert tensor dtype to TypeMeta");

    m.def(
        "get_scalar_type",
        [](caffe2::TypeMeta dtype) { return dtype.toScalarType(); },
        "Convert TypeMeta to scalar type");

    pybind11::class_<PyCacheStoreInputs>(m, "PyCacheStoreInputs").def(pybind11::init<>());

    pybind11::class_<rtp_llm::ParamsBase, std::shared_ptr<rtp_llm::ParamsBase>>(m, "ParamsBase")
        .def(pybind11::init<>())
        .def(
            "fill_params",
            [](rtp_llm::ParamsBase& self,
               torch::Tensor        sequence_lengths,
               torch::Tensor        input_lengths,
               torch::Tensor        kv_cache_block_id_host,
               int                  batch_size,
               int                  seq_size_per_block) {
                self.fillParams(
                    sequence_lengths, input_lengths, kv_cache_block_id_host, batch_size, seq_size_per_block);
            },
            pybind11::arg("sequence_lengths"),
            pybind11::arg("input_lengths"),
            pybind11::arg("kv_cache_block_id_host"),
            pybind11::arg("batch_size"),
            pybind11::arg("seq_size_per_block"),
            "Fill parameters for CUDA graph execution");

    pybind11::class_<PyPrefillCudaGaphCopyParams>(m, "PyPrefillCudaGaphCopyParams")
        .def(pybind11::init<>())
        .def_readwrite("cuda_graph_prefill_batch_size", &PyPrefillCudaGaphCopyParams::cuda_graph_prefill_batch_size)
        .def_readwrite("max_seq_len", &PyPrefillCudaGaphCopyParams::max_seq_len)
        .def_readwrite("max_batch_size", &PyPrefillCudaGaphCopyParams::max_batch_size);

    pybind11::class_<PyContextParallelParams>(m, "PyContextParallelParams")
        .def(pybind11::init<>())
        .def_readwrite("prefill_cp_padding_lengths", &PyContextParallelParams::prefill_cp_padding_lengths)
        .def_readwrite("prefill_cp_chunk_lengths", &PyContextParallelParams::prefill_cp_chunk_lengths)
        .def_readwrite("prefill_shuffle_indices", &PyContextParallelParams::prefill_shuffle_indices)
        .def_readwrite("prefill_qkv_restore_indice", &PyContextParallelParams::prefill_qkv_restore_indice)
        .def_readwrite("prefill_qkv_padding_mask", &PyContextParallelParams::prefill_qkv_padding_mask)
        .def_readwrite("prefill_actual_input_lengths_cpu", &PyContextParallelParams::prefill_actual_input_lengths_cpu);

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
        .def_readwrite("is_cuda_graph", &PyAttentionInputs::is_cuda_graph)
        .def_readwrite("is_target_verify", &PyAttentionInputs::is_target_verify)
        .def_readwrite("prefix_lengths", &PyAttentionInputs::prefix_lengths)
        .def_readwrite("sequence_lengths", &PyAttentionInputs::sequence_lengths)
        .def_readwrite("input_lengths", &PyAttentionInputs::input_lengths)
        .def_readwrite("kv_cache_kernel_block_id", &PyAttentionInputs::kv_cache_kernel_block_id)
        .def_readwrite("kv_cache_kernel_block_id_device", &PyAttentionInputs::kv_cache_kernel_block_id_device)
        .def_readwrite("kv_cache_block_id", &PyAttentionInputs::kv_cache_block_id)
        .def_readwrite("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readwrite("dtype", &PyAttentionInputs::dtype)
        .def_readwrite("cu_seqlens_device", &PyAttentionInputs::cu_seqlens_device)
        .def_readwrite("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readwrite("cu_kv_seqlens_device", &PyAttentionInputs::cu_kv_seqlens_device)
        .def_readwrite("context_total_kv_length", &PyAttentionInputs::context_total_kv_length)
        .def_readwrite("total_tokens", &PyAttentionInputs::total_tokens)
        .def_readwrite("padding_offset", &PyAttentionInputs::padding_offset)
        .def_readwrite("is_s_padded", &PyAttentionInputs::is_s_padded)
        .def_readonly("prefix_lengths_device", &PyAttentionInputs::prefix_lengths_device)
        .def_readwrite("sequence_lengths_plus_1_device", &PyAttentionInputs::sequence_lengths_plus_1_device)
        .def_readonly("input_lengths_device", &PyAttentionInputs::input_lengths_device)
        .def_readwrite("decode_cu_seqlens_device", &PyAttentionInputs::decode_cu_seqlens_device)
        .def_readwrite("decode_cu_seqlens", &PyAttentionInputs::decode_cu_seqlens)
        .def_readwrite("cache_store_inputs", &PyAttentionInputs::cache_store_inputs)
        .def_readwrite("context_parallel_info", &PyAttentionInputs::context_parallel_info)
        .def_readwrite("combo_position_ids", &PyAttentionInputs::combo_position_ids)
        .def("__repr__", [](const PyAttentionInputs& self) { return "PyAttentionInputs"; })
        .def_readwrite("prefill_cuda_graph_copy_params", &PyAttentionInputs::prefill_cuda_graph_copy_params)
        .def_readwrite("headwise_config", &PyAttentionInputs::headwise_config)
        .def("__copy__", [](const PyAttentionInputs& self) { return PyAttentionInputs(self); });

    pybind11::class_<BertEmbeddingInputs>(m, "BertEmbeddingInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float>(),
             pybind11::arg("combo_position_ids")     = torch::empty(0),
             pybind11::arg("position_encoding")      = torch::empty(0),
             pybind11::arg("combo_tokens_type_ids")  = torch::empty(0),
             pybind11::arg("token_type_embedding")   = torch::empty(0),
             pybind11::arg("input_embedding_scalar") = 1.0f)
        .def_readwrite("combo_position_ids", &BertEmbeddingInputs::combo_position_ids, "Combined position IDs tensor")
        .def_readwrite("position_encoding", &BertEmbeddingInputs::position_encoding, "Position encoding tensor")
        .def_readwrite(
            "combo_tokens_type_ids", &BertEmbeddingInputs::combo_tokens_type_ids, "Combined token type IDs tensor")
        .def_readwrite(
            "token_type_embedding", &BertEmbeddingInputs::token_type_embedding, "Token type embedding tensor")
        .def_readwrite(
            "input_embedding_scalar", &BertEmbeddingInputs::input_embedding_scalar, "Input embedding scalar value")
        .def("__repr__", [](const BertEmbeddingInputs& self) { return "BertEmbeddingInputs"; });

    pybind11::class_<PyEmbeddingInputs>(m, "PyEmbeddingInputs")
        .def(pybind11::init<>())
        .def_readwrite(
            "combo_tokens_type_ids", &PyEmbeddingInputs::combo_tokens_type_ids, "Combined token type IDs tensor")
        .def_readwrite("text_tokens_mask", &PyEmbeddingInputs::text_tokens_mask, "Text tokens mask tensor")
        .def("__repr__", [](const PyEmbeddingInputs& self) { return "PyEmbeddingInputs"; });

    pybind11::class_<PyMultimodalInputs>(m, "PyMultimodalInputs")
        .def(pybind11::init<>())
        .def_readwrite("multimodal_features", &PyMultimodalInputs::multimodal_features, "Multimodal features tensor")
        .def_readwrite(
            "mm_features_locs", &PyMultimodalInputs::mm_features_locs, "Multimodal features locations tensor")
        .def_readwrite(
            "mm_extra_input", &PyMultimodalInputs::mm_extra_input, "Multimodal model-specific extra input tensor")
        .def("__repr__", [](const PyMultimodalInputs& self) { return "PyMultimodalInputs"; });

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def(pybind11::init([](torch::Tensor       input_ids,
                               torch::Tensor       input_hiddens,
                               torch::Tensor       combo_position_ids,
                               PyEmbeddingInputs   embedding_inputs,
                               PyMultimodalInputs  multimodal_inputs,
                               pybind11::object    attention_inputs,
                               BertEmbeddingInputs bert_embedding_inputs) {
                 PyModelInputs result;
                 result.input_ids             = std::move(input_ids);
                 result.input_hiddens         = std::move(input_hiddens);
                 result.combo_position_ids    = std::move(combo_position_ids);
                 result.embedding_inputs      = std::move(embedding_inputs);
                 result.multimodal_inputs     = std::move(multimodal_inputs);
                 result.bert_embedding_inputs = std::move(bert_embedding_inputs);
                 if (pybind11::isinstance<PyAttentionInputs>(attention_inputs)) {
                     result.attention_inputs = attention_inputs.cast<PyAttentionInputs>();
                 } else {
                     result.attention_inputs_by_tag = attention_inputs.cast<AttentionInputsByTag>();
                     RTP_LLM_CHECK_WITH_INFO(!result.attention_inputs_by_tag.empty(),
                                             "attention_inputs tag map must not be empty");
                     result.attention_inputs = result.attention_inputs_by_tag.begin()->second;
                 }
                 return result;
             }),
             pybind11::arg("input_ids")             = torch::empty(0),
             pybind11::arg("input_hiddens")         = torch::empty(0),
             pybind11::arg("combo_position_ids")    = torch::empty(0),
             pybind11::arg("embedding_inputs")      = PyEmbeddingInputs(),
             pybind11::arg("multimodal_inputs")     = PyMultimodalInputs(),
             pybind11::arg("attention_inputs")      = PyAttentionInputs(),
             pybind11::arg("bert_embedding_inputs") = BertEmbeddingInputs())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("input_hiddens", &PyModelInputs::input_hiddens, "Input hidden states tensor")
        .def_readwrite("combo_position_ids", &PyModelInputs::combo_position_ids, "Combo position IDs tensor")
        .def_readwrite("embedding_inputs", &PyModelInputs::embedding_inputs, "Embedding inputs structure")
        .def_readwrite("multimodal_inputs", &PyModelInputs::multimodal_inputs, "Multimodal inputs structure")
        .def_property(
            "attention_inputs",
            [](PyModelInputs& self) -> pybind11::object {
                if (!self.attention_inputs_by_tag.empty()) {
                    pybind11::dict result;
                    for (auto& [tag, inputs] : self.attention_inputs_by_tag) {
                        result[pybind11::str(tag)] = pybind11::cast(
                            &inputs, pybind11::return_value_policy::reference_internal, pybind11::cast(&self));
                    }
                    return std::move(result);
                }
                return pybind11::cast(
                    &self.attention_inputs, pybind11::return_value_policy::reference_internal, pybind11::cast(&self));
            },
            [](PyModelInputs& self, pybind11::object value) {
                if (pybind11::isinstance<PyAttentionInputs>(value)) {
                    self.attention_inputs        = value.cast<PyAttentionInputs>();
                    self.attention_inputs_by_tag = {};
                    return;
                }
                auto by_tag = value.cast<AttentionInputsByTag>();
                RTP_LLM_CHECK_WITH_INFO(!by_tag.empty(), "attention_inputs tag map must not be empty");
                self.attention_inputs        = by_tag.begin()->second;
                self.attention_inputs_by_tag = std::move(by_tag);
            },
            "A PyAttentionInputs value or a tag-to-PyAttentionInputs mapping")
        .def_readwrite(
            "bert_embedding_inputs", &PyModelInputs::bert_embedding_inputs, "BERT embedding inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<>(), "Default constructor")
        .def(pybind11::init<torch::Tensor>(),
             pybind11::arg("hidden_states"),
             "Initialize with hidden states tensor only (params_ptr defaults to nullptr)")
        .def(pybind11::init([](torch::Tensor hidden_states, pybind11::object params_obj) {
                 // PyModelOutputs may be destroyed by an engine thread without the
                 // GIL. Keep its lifetime purely C++ and discard non-ParamsBase
                 // Python objects instead of retaining them in the output.
                 std::shared_ptr<rtp_llm::ParamsBase> params_ptr = nullptr;
                 if (!params_obj.is_none()) {
                     try {
                         params_ptr = pybind11::cast<std::shared_ptr<rtp_llm::ParamsBase>>(params_obj);
                     } catch (const pybind11::cast_error&) {
                         // Some attention implementations expose Python-only params.
                         // They are owned by the FMHA implementation during forward.
                     }
                 }
                 return PyModelOutputs(std::move(hidden_states), std::move(params_ptr));
             }),
             pybind11::arg("hidden_states"),
             pybind11::arg("params_ptr"),
             "Initialize with hidden states tensor and params pointer")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor")
        .def_readwrite("params_ptr", &PyModelOutputs::params_ptr, "Parameters pointer");
}

}  // namespace torch_ext
