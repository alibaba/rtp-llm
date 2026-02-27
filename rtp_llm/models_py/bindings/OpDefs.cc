#include "OpDefs.h"

namespace torch_ext {

void registerPyOpDefs(pybind11::module& m) {
    pybind11::class_<LayerKVCache>(m, "LayerKVCache")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_base", &LayerKVCache::kv_cache_base, "Key/value cache tensor (per-layer view)")
        .def_readwrite("kv_scale_base", &LayerKVCache::kv_scale_base, "Key/value cache scale tensor")
        .def_readonly("seq_size_per_block", &LayerKVCache::seq_size_per_block, "Sequence size per block")
        .def_readonly("layer_id", &LayerKVCache::layer_id, "Global layer id");

    pybind11::class_<KVCache>(m, "KVCache")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_base", &KVCache::kv_cache_base, "Full multi-layer KV cache tensor")
        .def_readwrite("kv_scale_base", &KVCache::kv_scale_base, "Full multi-layer KV scale tensor")
        .def_readonly("seq_size_per_block", &KVCache::seq_size_per_block, "Sequence size per block")
        .def_readonly("num_kv_heads", &KVCache::num_kv_heads, "Number of KV heads per TP rank")
        .def_readonly("head_dim", &KVCache::head_dim, "Head dimension")
        .def_readonly("use_mla", &KVCache::use_mla, "Whether MLA cache layout is used")
        .def_readonly("kv_lora_rank", &KVCache::kv_lora_rank, "MLA KV LoRA rank")
        .def_readonly("rope_head_dim", &KVCache::rope_head_dim, "MLA RoPE head dimension")
        .def("get_layer_cache",
             &KVCache::getLayerCache,
             "Return a per-layer LayerKVCache for the given global layer id");

    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readonly("kv_cache", &PyModelInitResources::kv_cache, "KV cache for all layers");

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
    pybind11::class_<PyCaptureMetaData>(m, "PyCaptureMetaData").def(pybind11::init<>());

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

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
        .def_readwrite("is_cuda_graph", &PyAttentionInputs::is_cuda_graph)
        .def_readwrite("prefix_lengths", &PyAttentionInputs::prefix_lengths)
        .def_readwrite("sequence_lengths", &PyAttentionInputs::sequence_lengths)
        .def_readwrite("input_lengths", &PyAttentionInputs::input_lengths)
        .def_readwrite("kv_cache_block_id_host", &PyAttentionInputs::kv_cache_block_id_host)
        .def_readwrite("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readwrite("kv_cache_block_id_host_by_group", &PyAttentionInputs::kv_cache_block_id_host_by_group)
        .def_readwrite("kv_cache_block_id_device_by_group", &PyAttentionInputs::kv_cache_block_id_device_by_group)
        .def_readwrite("kv_cache_layer_to_group", &PyAttentionInputs::kv_cache_layer_to_group)
        .def_readwrite("dtype", &PyAttentionInputs::dtype)
        .def_readwrite("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readwrite("cu_kv_seqlens", &PyAttentionInputs::cu_kv_seqlens)
        .def_readwrite("context_total_kv_length", &PyAttentionInputs::context_total_kv_length)
        .def_readwrite("total_tokens", &PyAttentionInputs::total_tokens)
        .def_readwrite("padding_offset", &PyAttentionInputs::padding_offset)
        .def_readwrite("is_s_padded", &PyAttentionInputs::is_s_padded)
        .def_readonly("prefix_lengths_d", &PyAttentionInputs::prefix_lengths_d)
        .def_readwrite("sequence_lengths_plus_1_d", &PyAttentionInputs::sequence_lengths_plus_1_d)
        .def_readonly("input_lengths_d", &PyAttentionInputs::input_lengths_d)
        .def_readwrite("decode_cu_seqlens_d", &PyAttentionInputs::decode_cu_seqlens_d)
        .def_readonly("decode_cu_seqlens_host", &PyAttentionInputs::decode_cu_seqlens_host)
        .def_readwrite("cache_store_inputs", &PyAttentionInputs::cache_store_inputs)
        .def("__repr__", [](const PyAttentionInputs& self) { return "PyAttentionInputs"; })
        .def_readwrite("prefill_cuda_graph_copy_params", &PyAttentionInputs::prefill_cuda_graph_copy_params);

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

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, torch::Tensor, PyAttentionInputs, BertEmbeddingInputs>(),
             pybind11::arg("input_ids")             = torch::empty(0),
             pybind11::arg("input_hiddens")         = torch::empty(0),
             pybind11::arg("attention_inputs")      = PyAttentionInputs(),
             pybind11::arg("bert_embedding_inputs") = BertEmbeddingInputs())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("input_hiddens", &PyModelInputs::input_hiddens, "Input hidden states tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs, "Attention inputs structure")
        .def_readwrite(
            "bert_embedding_inputs", &PyModelInputs::bert_embedding_inputs, "BERT embedding inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<>(), "Default constructor")
        .def(pybind11::init<torch::Tensor>(),
             pybind11::arg("hidden_states"),
             "Initialize with hidden states tensor only (params_ptr defaults to nullptr)")
        .def(pybind11::init([](torch::Tensor hidden_states, pybind11::object params_obj) {
                 // Try to cast to shared_ptr, return nullptr if conversion fails
                 std::shared_ptr<rtp_llm::ParamsBase> params_ptr     = nullptr;
                 py::object                           py_attn_params = py::none();
                 try {
                     params_ptr = pybind11::cast<std::shared_ptr<rtp_llm::ParamsBase>>(params_obj);
                 } catch (const pybind11::cast_error& e) {
                     // Conversion failed, params_ptr remains nullptr
                     //  RTP_LLM_LOG_INFO("Failed to cast params_obj to shared_ptr<ParamsBase>: %s", e.what());
                     py_attn_params = params_obj;
                 }
                 return PyModelOutputs(hidden_states, params_ptr, py_attn_params);
             }),
             pybind11::arg("hidden_states"),
             pybind11::arg("params_ptr"),
             "Initialize with hidden states tensor and params pointer")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor")
        .def_readwrite("params_ptr", &PyModelOutputs::params_ptr, "Parameters pointer");
}

}  // namespace torch_ext
