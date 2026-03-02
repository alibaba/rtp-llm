#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

void registerMultimodal(const py::module& m) {
    pybind11::class_<MultimodalInput>(m, "MultimodalInput")
        .def(pybind11::init<std::string, int32_t, torch::Tensor, MMPreprocessConfig>(),
             py::arg("url"),
             py::arg("mm_type"),
             py::arg("tensor"),
             py::arg("mm_preprocess_config"))
        .def_readwrite("url", &MultimodalInput::url)
        .def_readwrite("mm_type", &MultimodalInput::mm_type)
        .def_readwrite("tensor", &MultimodalInput::tensor)
        .def_readwrite("mm_preprocess_config", &MultimodalInput::mm_preprocess_config)
        .def("to_string", &MultimodalInput::to_string)
        .def(pybind11::pickle(
            [](const MultimodalInput& m) {  // __getstate__
                return py::make_tuple(m.url, m.mm_type, m.tensor, m.mm_preprocess_config);
            },
            [](py::tuple t) {  // __setstate__
                return MultimodalInput(t[0].cast<std::string>(),
                                       t[1].cast<int32_t>(),
                                       t[2].cast<torch::Tensor>(),
                                       t[3].cast<MMPreprocessConfig>());
            }));
    pybind11::class_<MMPreprocessConfig>(m, "MMPreprocessConfig")
        .def(pybind11::
                 init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, std::vector<float>, int32_t>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("min_pixels"),
             py::arg("max_pixels"),
             py::arg("fps"),
             py::arg("min_frames"),
             py::arg("max_frames"),
             py::arg("crop_positions"),
             py::arg("mm_timeout_ms"))
        .def_readwrite("width", &MMPreprocessConfig::width)
        .def_readwrite("height", &MMPreprocessConfig::height)
        .def_readwrite("min_pixels", &MMPreprocessConfig::min_pixels)
        .def_readwrite("max_pixels", &MMPreprocessConfig::max_pixels)
        .def_readwrite("fps", &MMPreprocessConfig::fps)
        .def_readwrite("min_frames", &MMPreprocessConfig::min_frames)
        .def_readwrite("max_frames", &MMPreprocessConfig::max_frames)
        .def_readwrite("crop_positions", &MMPreprocessConfig::crop_positions)
        .def_readwrite("mm_timeout_ms", &MMPreprocessConfig::mm_timeout_ms)
        .def("to_string", &MMPreprocessConfig::to_string)
        .def(pybind11::pickle(
            [](const MMPreprocessConfig& m) {  // __getstate__
                return py::make_tuple(m.width,
                                      m.height,
                                      m.min_pixels,
                                      m.max_pixels,
                                      m.fps,
                                      m.min_frames,
                                      m.max_frames,
                                      m.crop_positions,
                                      m.mm_timeout_ms);
            },
            [](py::tuple t) {  // __setstate__
                return MMPreprocessConfig(t[0].cast<int32_t>(),
                                          t[1].cast<int32_t>(),
                                          t[2].cast<int32_t>(),
                                          t[3].cast<int32_t>(),
                                          t[4].cast<int32_t>(),
                                          t[5].cast<int32_t>(),
                                          t[6].cast<int32_t>(),
                                          t[7].cast<std::vector<float>>(),
                                          t[8].cast<int32_t>());
            }));
    pybind11::class_<MultimodalOutput>(m, "MultimodalOutput")
        .def(pybind11::init<>())
        .def_readwrite("mm_features", &MultimodalOutput::mm_features)
        .def_readwrite("mm_position_ids", &MultimodalOutput::mm_position_ids)
        .def_readwrite("mm_deepstack_embeds", &MultimodalOutput::mm_deepstack_embeds);
    pybind11::class_<MultimodalFeature>(m, "MultimodalFeature")
        .def(pybind11::init<>())
        .def_readwrite("features", &MultimodalFeature::features)
        .def_readwrite("inputs", &MultimodalFeature::inputs)
        .def_readwrite("text_tokens_mask", &MultimodalFeature::text_tokens_mask)
        .def_readwrite("locs", &MultimodalFeature::locs)
        .def_readwrite("expanded_ids", &MultimodalFeature::expanded_ids);
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
        .def("setTensorOutput", &EmbeddingOutput::setTensorOutput)
        .def("setMapOutput", &EmbeddingOutput::setMapOutput);
}

PYBIND11_MODULE(libth_transformer, m) {
    registerRtpLLMOp(m);
    registerMultimodal(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingOutput(m);
}

}  // namespace rtp_llm
