#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/awi/AwiObservationRegistry.h"
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

PYBIND11_MODULE(libth_transformer, m) {
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingOutput(m);
    m.def("get_multimodal_feature_hash", &getMultimodalFeatureHash, py::call_guard<py::gil_scoped_release>());
    m.def("get_multimodal_feature_hashes", &getMultimodalFeatureHashes, py::call_guard<py::gil_scoped_release>());

    // AWI evaluator instrumentation surface. The registry lives in this .so's
    // dependency graph (engine + scheduler + graph runner all write it), so
    // these accessors see the same singleton the C++ taps record into.
    m.def(
        "awi_query_observations",
        [](const std::string& operation_id) {
            return AwiObservationRegistry::instance().queryJson(operation_id);
        },
        py::arg("operation_id"),
        py::call_guard<py::gil_scoped_release>());
    m.def(
        "awi_set_build_info",
        [](const std::string& internal_sha, const std::string& public_sha) {
            AwiObservationRegistry::instance().setBuildInfo(internal_sha, public_sha);
        },
        py::arg("internal_sha"),
        py::arg("public_sha"));
    m.def(
        "awi_set_configuration",
        [](int64_t context_tokens, bool memory_kv_reuse, const std::string& graph_backend) {
            AwiObservationRegistry::instance().setConfiguration(context_tokens, memory_kv_reuse, graph_backend);
        },
        py::arg("context_tokens"),
        py::arg("memory_kv_reuse"),
        py::arg("graph_backend"));
    m.def(
        "awi_set_rank",
        [](int64_t world_rank) {
            AwiObservationRegistry::instance().setRank(world_rank);
        },
        py::arg("world_rank"));
    m.def(
        "awi_set_weights_placement",
        [](const std::string& weights_json_array) {
            AwiObservationRegistry::instance().setWeightsPlacementJson(weights_json_array);
        },
        py::arg("weights_json_array"));
    m.def("awi_record_startup_network_fetch",
          []() {
              AwiObservationRegistry::instance().recordStartupNetworkFetch();
          });
}

}  // namespace rtp_llm
