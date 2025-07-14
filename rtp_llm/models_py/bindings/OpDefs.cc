#include "OpDefs.h"

namespace torch_ext {

void registerPyOpDefs(pybind11::module& m) {
    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readwrite("k_cache_base", &PyModelInitResources::k_cache_base, "Key cache base tensor")
        .def_readwrite("v_cache_base", &PyModelInitResources::v_cache_base, "Value cache base tensor");

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def(
            "get_prefill_flash_infer_attn",
            [](const PyAttentionInputs& self) -> pybind11::capsule {
                if (self.prefill_flash_infer_attn) {
                    return pybind11::capsule(self.prefill_flash_infer_attn.get(), "prefill_flash_infer_attn");
                }
                return pybind11::capsule(nullptr, "prefill_flash_infer_attn");
            },
            "Get prefill flash infer attention as capsule")
        .def(
            "get_decode_flash_infer_attn",
            [](const PyAttentionInputs& self) -> pybind11::capsule {
                if (self.decode_flash_infer_attn) {
                    return pybind11::capsule(self.decode_flash_infer_attn.get(), "decode_flash_infer_attn");
                }
                return pybind11::capsule(nullptr, "decode_flash_infer_attn");
            },
            "Get decode flash infer attention as capsule");

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs, "Attention inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<torch::Tensor>(), pybind11::arg("hidden_states"), "Initialize with hidden states tensor")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor");
}

}  // namespace torch_ext
