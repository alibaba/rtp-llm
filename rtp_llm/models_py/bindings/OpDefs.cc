#include "OpDefs.h"

namespace torch_ext {

void registerPyOpDefs(pybind11::module& m) {
    pybind11::class_<KVCache>(m, "KVCache")
        .def(pybind11::init<>())
        .def_readonly("k_cache_base", &KVCache::k_cache_base, "Key cache base tensor")
        .def_readonly("v_cache_base", &KVCache::v_cache_base, "Value cache base tensor")
        .def_readonly("k_scale_base", &KVCache::k_scale_base, "Key cache scale tensor")
        .def_readonly("v_scale_base", &KVCache::v_scale_base, "Value cache scale tensor")
        .def_readonly("layer_id", &KVCache::layer_id, "kv cache layer id")
        .def("get_layer_cache", &KVCache::getLayerCache);

    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readonly("kv_cache", &PyModelInitResources::kv_cache, "kv cache");

    pybind11::class_<caffe2::TypeMeta>(m, "TypeMeta").def(pybind11::init<>());

    pybind11::class_<PyCacheStoreInputs>(m, "PyCacheStoreInputs").def(pybind11::init<>());

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def_readonly("is_prefill", &PyAttentionInputs::is_prefill)
        .def_readonly("prefix_lengths", &PyAttentionInputs::prefix_lengths)
        .def_readonly("sequence_lengths", &PyAttentionInputs::sequence_lengths)
        .def_readonly("input_lengths", &PyAttentionInputs::input_lengths)
        .def_readonly("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readonly("kv_cache_block_id_host", &PyAttentionInputs::kv_cache_block_id_host)
        .def_readonly("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readonly("kv_block_offset", &PyAttentionInputs::kv_block_offset)
        .def_readonly("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readonly("padding_offset", &PyAttentionInputs::padding_offset)
        .def_readonly("cache_store_inputs", &PyAttentionInputs::cache_store_inputs);

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, PyAttentionInputs>(),
             pybind11::arg("input_ids")        = torch::empty(0),
             pybind11::arg("attention_inputs") = PyAttentionInputs())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs, "Attention inputs structure");

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
