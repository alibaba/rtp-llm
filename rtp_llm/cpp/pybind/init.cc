#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/awi/AwiCollector.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

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

void registerAwiObservationHooks(py::module& m) {
    // Startup facts for the AWI adaptation-evaluator observations. The Python
    // layer measures/derives these (weight placement from real allocations,
    // network-fetch audit count, build identity) and injects them once; the
    // C++ collector serves them verbatim through the AwiObservations RPC.
    auto awi = m.def_submodule("awi", "AWI adaptation-evaluator observation hooks");
    awi.def("set_build_info",
            [](std::string internal_sha, std::string public_sha) {
                AwiCollector::instance().setBuildInfo(std::move(internal_sha), std::move(public_sha));
            });
    awi.def("set_configuration_json",
            [](std::string json) { AwiCollector::instance().setConfigurationJson(std::move(json)); });
    awi.def("set_weights_json",
            [](std::string json) { AwiCollector::instance().setWeightsJson(std::move(json)); });
    awi.def("set_startup_network_fetches",
            [](int64_t count) { AwiCollector::instance().setStartupNetworkFetches(count); });
}

PYBIND11_MODULE(libth_transformer, m) {
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingOutput(m);
    registerAwiObservationHooks(m);
    m.def("get_multimodal_feature_hash", &getMultimodalFeatureHash, py::call_guard<py::gil_scoped_release>());
    m.def("get_multimodal_feature_hashes", &getMultimodalFeatureHashes, py::call_guard<py::gil_scoped_release>());
}

}  // namespace rtp_llm
