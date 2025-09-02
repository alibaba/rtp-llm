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
        .def_readonly("kv_cache_block_id_host", &PyAttentionInputs::kv_cache_block_id_host)
        .def_readonly("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readonly("kv_block_offset", &PyAttentionInputs::kv_block_offset)
        .def_readonly("dtype", &PyAttentionInputs::dtype)
        .def_readonly("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readonly("cache_store_inputs", &PyAttentionInputs::cache_store_inputs);

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, PyAttentionInputs>(),
             pybind11::arg("input_ids")        = torch::empty(0),
             pybind11::arg("attention_inputs") = PyAttentionInputs())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs, "Attention inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<torch::Tensor>(), pybind11::arg("hidden_states"), "Initialize with hidden states tensor")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor");
}

}  // namespace torch_ext
