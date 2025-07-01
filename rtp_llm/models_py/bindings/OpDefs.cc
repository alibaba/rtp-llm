#include "OpDefs.h"

namespace torch_ext {

void registerPyOpDefs(pybind11::module &m) {
    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readwrite("k_cache_base", &PyModelInitResources::k_cache_base,
                       "Key cache base tensor")
        .def_readwrite("v_cache_base", &PyModelInitResources::v_cache_base,
                       "Value cache base tensor");

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        // Workaround for std::shared_ptr<void> - provide getter/setter methods
        // .def("set_prefill_flash_infer_attn", [](PyAttentionInputs& self, pybind11::capsule cap) {
        //     self.prefill_flash_infer_attn = std::static_pointer_cast<void>(
        //         std::shared_ptr<void>(cap.get_pointer(), [cap](void*) { /* capsule manages lifetime */ }));
        // }, "Set prefill flash infer attention from capsule")
        // .def("set_decode_flash_infer_attn", [](PyAttentionInputs& self, pybind11::capsule cap) {
        //     self.decode_flash_infer_attn = std::static_pointer_cast<void>(
        //         std::shared_ptr<void>(cap.get_pointer(), [cap](void*) { /* capsule manages lifetime */ }));
        // }, "Set decode flash infer attention from capsule")
        .def("get_prefill_flash_infer_attn", [](const PyAttentionInputs& self) -> pybind11::capsule {
            if (self.prefill_flash_infer_attn) {
                return pybind11::capsule(self.prefill_flash_infer_attn.get(), "prefill_flash_infer_attn");
            }
            return pybind11::capsule(nullptr, "prefill_flash_infer_attn");
        }, "Get prefill flash infer attention as capsule")
        .def("get_decode_flash_infer_attn", [](const PyAttentionInputs& self) -> pybind11::capsule {
            if (self.decode_flash_infer_attn) {
                return pybind11::capsule(self.decode_flash_infer_attn.get(), "decode_flash_infer_attn");
            }
            return pybind11::capsule(nullptr, "decode_flash_infer_attn");
        }, "Get decode flash infer attention as capsule")
        // .def("has_prefill_flash_infer_attn", [](const PyAttentionInputs& self) -> bool {
        //     return self.prefill_flash_infer_attn != nullptr;
        // }, "Check if prefill flash infer attention is set")
        // .def("has_decode_flash_infer_attn", [](const PyAttentionInputs& self) -> bool {
        //     return self.decode_flash_infer_attn != nullptr;
        // }, "Check if decode flash infer attention is set")
        ;

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def_readwrite("input_ids", &PyModelInputs::input_ids,
                      "Input token IDs tensor")
        .def_readwrite("attention_inputs", &PyModelInputs::attention_inputs,
                      "Attention inputs structure");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<torch::Tensor>(), pybind11::arg("hidden_states"),
             "Initialize with hidden states tensor")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states,
                      "Hidden states output tensor");
}

}

