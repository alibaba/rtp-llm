#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/cache/types.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

void registerMultimodalInput(const py::module& m) {
    pybind11::class_<MultimodalInput>(m, "MultimodalInput")
        .def(pybind11::init<std::string, torch::Tensor, int32_t>(),
             py::arg("url"),
             py::arg("tensor"),
             py::arg("mm_type"))
        .def_readwrite("url", &MultimodalInput::url)
        .def_readwrite("mm_type", &MultimodalInput::mm_type)
        .def_readwrite("tensor", &MultimodalInput::tensor);
}

void registerEmbeddingOutput(const py::module& m) {
    py::class_<TypedOutput>(m, "TypedOutput")
        .def(py::init<>())
        .def_readwrite("isTensor", &TypedOutput::isTensor)
        .def_property(
            "t",
            [](const TypedOutput& self) -> py::object {
                return self.t.has_value() ? py::cast(self.t.value()) : py::none();
            },
            [](TypedOutput& self, const at::Tensor& tensor) { self.setTensorOuput(tensor); })
        .def_property(
            "map",
            [](const TypedOutput& self) -> py::object {
                return self.map.has_value() ? py::cast(self.map.value()) : py::none();
            },
            // FIX: Take by value instead of const reference
            [](TypedOutput& self, std::vector<std::map<std::string, at::Tensor>> map_val) {
                self.setMapOutput(map_val);
            });

    py::class_<EmbeddingOutput>(m, "EmbeddingCppOutput")
        .def(py::init<>())
        .def_readwrite("output", &EmbeddingOutput::output)
        .def_readwrite("error_info", &EmbeddingOutput::error_info)
        .def("setTensorOutput", &EmbeddingOutput::setTensorOutput)
        .def("setMapOutput", &EmbeddingOutput::setMapOutput)
        .def("setError", &EmbeddingOutput::setError);
}

PYBIND11_MODULE(libth_transformer, m) {
    registerRtpLLMOp(m);
    registerMultimodalInput(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingOutput(m);
}

}  // namespace rtp_llm
