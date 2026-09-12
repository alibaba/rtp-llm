#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <array>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace rtp_llm {
namespace {

using TensorField = std::pair<const char*, torch::Tensor GptModelInputs::*>;
const std::array<TensorField, 18> kTensorFields{{
    {"combo_tokens", &GptModelInputs::combo_tokens},
    {"input_lengths", &GptModelInputs::input_lengths},
    {"sequence_lengths", &GptModelInputs::sequence_lengths},
    {"prefix_lengths", &GptModelInputs::prefix_lengths},
    {"lm_output_indexes", &GptModelInputs::lm_output_indexes},
    {"combo_position_ids", &GptModelInputs::combo_position_ids},
    {"text_tokens_mask", &GptModelInputs::text_tokens_mask},
    {"mm_features_locs", &GptModelInputs::mm_features_locs},
    {"mm_features_spans", &GptModelInputs::mm_features_spans},
    {"request_id", &GptModelInputs::request_id},
    {"request_pd_separation", &GptModelInputs::request_pd_separation},
    {"v41_token_types", &GptModelInputs::v41_token_types},
    {"v41_token_valid", &GptModelInputs::v41_token_valid},
    {"engram_history_ids", &GptModelInputs::engram_history_ids},
    {"engram_history_valid", &GptModelInputs::engram_history_valid},
    {"v41_request_id", &GptModelInputs::v41_request_id},
    {"v41_state_ready", &GptModelInputs::v41_state_ready},
    {"v41_is_fake", &GptModelInputs::v41_is_fake},
}};

using BoolField = std::pair<const char*, bool GptModelInputs::*>;
const std::array<BoolField, 3> kBoolFields{{
    {"need_all_logits", &GptModelInputs::need_all_logits},
    {"need_all_hidden_states", &GptModelInputs::need_all_hidden_states},
    {"is_fake_stream", &GptModelInputs::is_fake_stream},
}};

// Keep the same receiver across calls so stale fields cannot be hidden by a
// test-side reconstruction of GptModelInputs before each broadcast.
class ModelInputsTpSyncTestState {
public:
    void replace(const py::dict& values) {
        inputs_ = GptModelInputs{};
        for (const auto& [name, member] : kTensorFields) {
            if (values.contains(name) && !values[name].is_none()) {
                inputs_.*member = values[name].cast<torch::Tensor>();
            }
        }
        for (const auto& [name, member] : kBoolFields) {
            if (values.contains(name)) {
                inputs_.*member = values[name].cast<bool>();
            }
        }
        if (values.contains("multimodal_features")) {
            setFeatures(values["multimodal_features"]);
        }
    }

    void setFeatures(const py::object& features) {
        if (features.is_none()) {
            inputs_.multimodal_features.reset();
        } else {
            inputs_.multimodal_features = features.cast<std::vector<torch::Tensor>>();
        }
    }

    void sync(int rank, int size) {
        ParallelismConfig config;
        config.tp_rank = rank;
        config.tp_size = size;
        tpSyncModelInputs(inputs_, config);
    }

    py::dict snapshot() const {
        py::dict result;
        for (const auto& [name, member] : kTensorFields) {
            const auto& tensor = inputs_.*member;
            result[name]       = tensor.defined() ? py::cast(tensor) : py::none();
        }
        for (const auto& [name, member] : kBoolFields) {
            result[name] = inputs_.*member;
        }
        result["multimodal_features"] =
            inputs_.multimodal_features ? py::cast(*inputs_.multimodal_features) : py::none();
        return result;
    }

private:
    GptModelInputs inputs_{};
};

}  // namespace
}  // namespace rtp_llm

PYBIND11_MODULE(libtp_sync_model_inputs_test_ops, module) {
    py::class_<rtp_llm::ModelInputsTpSyncTestState>(module, "ModelInputsTpSyncTestState")
        .def(py::init<>())
        .def("replace", &rtp_llm::ModelInputsTpSyncTestState::replace)
        .def("set_features", &rtp_llm::ModelInputsTpSyncTestState::setFeatures)
        .def("snapshot", &rtp_llm::ModelInputsTpSyncTestState::snapshot)
        .def("sync", &rtp_llm::ModelInputsTpSyncTestState::sync, py::call_guard<py::gil_scoped_release>());
    module.def("cpu_broadcaster_initialized", &rtp_llm::isCpuTpBroadcasterInitialized);
}
