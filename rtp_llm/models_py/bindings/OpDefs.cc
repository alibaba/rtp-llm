#include "OpDefs.h"

namespace torch_ext {

void registerPyOpDefs(pybind11::module& m) {
    pybind11::class_<MlaParams>(m, "MlaParams")
        .def(pybind11::init<>())
        .def_readonly("batch_indice", &MlaParams::batch_indice)
        .def_readonly("positions", &MlaParams::positions)
        .def_readonly("paged_kv_last_page_len", &MlaParams::paged_kv_last_page_len)
        .def_readonly("kvlen", &MlaParams::kvlen)
        .def_readonly("page_indice", &MlaParams::page_indice)
        .def_readonly("reuse_cache_page_indice", &MlaParams::reuse_cache_page_indice)
        .def_readonly("decode_page_indptr", &MlaParams::decode_page_indptr)
        .def_readonly("prefill_page_indptr", &MlaParams::prefill_page_indptr)
        .def_readonly("qo_indptr", &MlaParams::qo_indptr)
        .def_readonly("batch_reuse_info_vec", &MlaParams::batch_reuse_info_vec);

    pybind11::class_<KVCache>(m, "KVCache")
        .def(pybind11::init<>())
        .def_readwrite("k_cache_base", &KVCache::k_cache_base, "Key cache base tensor")
        .def_readwrite("v_cache_base", &KVCache::v_cache_base, "Value cache base tensor")
        .def_readwrite("k_scale_base", &KVCache::k_scale_base, "Key cache scale tensor")
        .def_readwrite("v_scale_base", &KVCache::v_scale_base, "Value cache scale tensor")
        .def_readonly("layer_id", &KVCache::layer_id, "kv cache layer id")
        .def("get_layer_cache", &KVCache::getLayerCache);

    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readonly("kv_cache", &PyModelInitResources::kv_cache, "kv cache");

    pybind11::class_<caffe2::TypeMeta>(m, "TypeMeta").def(pybind11::init<>());

    m.def(
        "get_typemeta",
        [](const torch::Tensor& tensor) { return torch::scalarTypeToTypeMeta(tensor.scalar_type()); },
        "Convert tensor dtype to TypeMeta");

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
        .def_readonly("cuda_graph_prefill_batch_size", &PyPrefillCudaGaphCopyParams::cuda_graph_prefill_batch_size)
        .def_readonly("max_seq_len", &PyPrefillCudaGaphCopyParams::max_seq_len)
        .def_readonly("hidden_size", &PyPrefillCudaGaphCopyParams::hidden_size)
        .def_readonly("max_batch_size", &PyPrefillCudaGaphCopyParams::max_batch_size);

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
        .def_readwrite("prefix_lengths", &PyAttentionInputs::prefix_lengths)
        .def_readwrite("sequence_lengths", &PyAttentionInputs::sequence_lengths)
        .def_readwrite("input_lengths", &PyAttentionInputs::input_lengths)
        .def_readwrite("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readwrite("kv_cache_block_id_host", &PyAttentionInputs::kv_cache_block_id_host)
        .def_readwrite("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readwrite("dtype", &PyAttentionInputs::dtype)
        .def_readwrite("kv_block_offset", &PyAttentionInputs::kv_block_offset)
        .def_readwrite("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readwrite("padding_offset", &PyAttentionInputs::padding_offset)
        .def_readwrite("cache_store_inputs", &PyAttentionInputs::cache_store_inputs)
        .def("__repr__", [](const PyAttentionInputs& self) { return "PyAttentionInputs"; })
        .def_readonly("prefill_cuda_graph_copy_params", &PyAttentionInputs::prefill_cuda_graph_copy_params);

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
        .def(pybind11::init<torch::Tensor, PyAttentionInputs, BertEmbeddingInputs>(),
             pybind11::arg("input_ids")             = torch::empty(0),
             pybind11::arg("attention_inputs")      = PyAttentionInputs(),
             pybind11::arg("bert_embedding_inputs") = BertEmbeddingInputs())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs, "Attention inputs structure")
        .def_readwrite(
            "bert_embedding_inputs", &PyModelInputs::bert_embedding_inputs, "BERT embedding inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<>(), "Default constructor")
        .def(pybind11::init<torch::Tensor, std::shared_ptr<rtp_llm::ParamsBase>>(),
             pybind11::arg("hidden_states"),
             pybind11::arg("params_ptr"),
             "Initialize with hidden states tensor and params pointer")
        .def(pybind11::init<torch::Tensor>(),
             pybind11::arg("hidden_states"),
             "Initialize with hidden states tensor only (params_ptr defaults to nullptr)")
        .def(pybind11::init<std::shared_ptr<rtp_llm::ParamsBase>>(),
             pybind11::arg("params_ptr"),
             "Initialize with params pointer only (hidden_states defaults to empty tensor)")
        .def(pybind11::init([](torch::Tensor hidden_states, pybind11::object params_obj) {
                 // Try to cast to shared_ptr, return nullptr if conversion fails
                 std::shared_ptr<rtp_llm::ParamsBase> params_ptr = nullptr;
                 try {
                     params_ptr = pybind11::cast<std::shared_ptr<rtp_llm::ParamsBase>>(params_obj);
                 } catch (const pybind11::cast_error& e) {
                     // Conversion failed, params_ptr remains nullptr
                     RTP_LLM_LOG_INFO("Failed to cast params_obj to shared_ptr<ParamsBase>: %s", e.what());
                 }
                 return PyModelOutputs(hidden_states, params_ptr);
             }),
             pybind11::arg("hidden_states"),
             pybind11::arg("params_ptr"),
             "Initialize with hidden states tensor and params pointer")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor")
        .def_readwrite("params_ptr", &PyModelOutputs::params_ptr, "Parameters pointer");
}

}  // namespace torch_ext
