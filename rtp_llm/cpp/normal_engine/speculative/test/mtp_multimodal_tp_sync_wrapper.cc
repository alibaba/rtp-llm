#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace rtp_llm {
// Register callbacks in the same linked ExecOps instance that tpSync uses.
// This declaration mirrors the existing ExecOps binding registration entry.
void registerExecCtxOps(py::module& module);

namespace {

torch::Tensor hostCopy(const torch::Tensor& tensor) {
    return tensor.defined() ? tensor.cpu().clone() : torch::empty({0}, torch::kInt32);
}

class MtpMultimodalTpFixture {
public:
    explicit MtpMultimodalTpFixture(int rank): rank_(rank) {
        RTP_LLM_CHECK_WITH_INFO(rank == 0 || rank == 1, "this regression requires exactly two ranks");
        parallelism_.tp_rank = rank;
        parallelism_.tp_size = 2;
        parallelism_.world_rank = rank;
        parallelism_.world_size = 2;
        parallelism_.local_rank = rank;
        parallelism_.local_world_size = 2;

        model_.model_type = "qwen35_moe";
        model_.vocab_size = 2048;
        model_.num_layers = 1;
        model_.max_seq_len = 32;
        const auto cache = test::makeSimpleMhaCacheConfig(1, 2, 1, TYPE_FP16);
        SpeculativeExecutionConfig speculative;
        // The actual Qwen3.5 launcher uses the EAGLE executor for this MTP descriptor.
        speculative.type = SP_TYPE_EAGLE;
        speculative.model_type = "qwen35_moe_mtp";
        processor_ = std::make_unique<MtpBatchStreamProcessor>(
            model_, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, speculative, false);

        input_.combo_tokens = torch::tensor({1000, 5}, torch::kInt32).pin_memory();
        input_.input_lengths = torch::tensor({2}, torch::kInt32).pin_memory();
        input_.sequence_lengths = torch::empty({0}, torch::kInt32).pin_memory();
        input_.prefix_lengths = torch::tensor({0}, torch::kInt32).pin_memory();
        input_.request_id = torch::tensor({123}, torch::kInt64).pin_memory();
        input_.request_pd_separation = torch::tensor({false}, torch::kBool).pin_memory();
        input_.lm_output_indexes = torch::tensor({1}, torch::kInt32).pin_memory();
        // Device-generated one-step mRoPE positions must stay on the GPU on
        // every TP rank. Start the peer on CPU with poison values so both the
        // value transfer and allocation-device hint are exercised.
        input_.combo_position_ids = rank == 0 ?
            torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt32).cuda() :
            torch::tensor({-1, -2, -3, -4, -5, -6}, torch::kInt32).pin_memory();
        // The peer starts with poison data, proving the first round transferred values.
        input_.text_tokens_mask = torch::tensor(rank == 0 ? std::vector<int>{0, 1} : std::vector<int>{9, 9},
                                                torch::kInt32).pin_memory();
        input_.mm_features_locs = torch::tensor({rank == 0 ? 0 : 9}, torch::kInt32).pin_memory();
        const float value = rank == 0 ? 1.0f : -1.0f;
        input_.multimodal_features = std::vector<torch::Tensor>{
            (torch::tensor({17.0f, 19.0f}).reshape({1, 2}) * value).cuda()};
        input_.mm_extra_input = std::vector<torch::Tensor>{
            (torch::tensor({101.0f, 102.0f, 201.0f, 202.0f}) * value).cuda()};
    }

    void sync() {
        tpSyncModelInputs(input_, parallelism_);
    }

    void shiftRoot() {
        RTP_LLM_CHECK_WITH_INFO(rank_ == 0, "only rank 0 may shift the global MTP input");
        auto query = std::make_shared<GenerateInput>();
        query->input_ids = torch::tensor({1000, 5}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        auto stream = std::make_shared<NormalGenerateStream>(
            query, model_, RuntimeConfig{}, ResourceContext{}, nullptr);
        const StreamGroups streams({stream});
        GptModelOutputs output;
        output.all_hidden_states = torch::zeros({2, 2}, torch::TensorOptions().device(torch::kCUDA));
        SamplerOutput sampled;
        sampled.token_ids = torch::tensor({7}, torch::kInt32).reshape({1, 1});
        // The prefill shifter consumes host positions. This fixture injects
        // CUDA positions only to exercise the TP transport contract; the
        // processor test covers production one-step position generation.
        input_.combo_position_ids = torch::Tensor();
        // Exercise the real entry point, including tokens and the descriptor gate.
        processor_->updatePrefillPostDraftModelInput(streams, input_, output, sampled, holder_);
        input_.combo_position_ids = torch::tensor({101, 102, 103, 104, 105, 106}, torch::kInt32).cuda();
    }

    py::dict snapshot() const {
        py::dict result;
        result["tokens"] = hostCopy(input_.combo_tokens);
        result["positions"] = hostCopy(input_.combo_position_ids);
        result["positions_on_cuda"] = input_.combo_position_ids.is_cuda();
        const auto hints = getModelInputShapeHints(input_);
        result["position_device_hint"] =
            (hints[GptModelInputIndex::tensorDeviceMap] & GptModelInputDeviceBit::kDeviceBitComboPositionIds) != 0;
        result["mask"] = hostCopy(input_.text_tokens_mask);
        result["locs"] = hostCopy(input_.mm_features_locs);
        result["features_present"] = input_.multimodal_features.has_value();
        result["extra_present"] = input_.mm_extra_input.has_value();
        py::list features;
        if (input_.multimodal_features.has_value()) {
            for (const auto& feature : input_.multimodal_features.value()) {
                features.append(hostCopy(feature));
            }
        }
        py::list extra;
        if (input_.mm_extra_input.has_value()) {
            for (const auto& tensor : input_.mm_extra_input.value()) {
                extra.append(hostCopy(tensor));
            }
        }
        result["features"] = features;
        result["extra"] = extra;
        return result;
    }

private:
    int rank_;
    ModelConfig model_;
    ParallelismConfig parallelism_;
    GptModelInputs input_;
    TensorHolder holder_;
    std::unique_ptr<MtpBatchStreamProcessor> processor_;
};

}  // namespace
}  // namespace rtp_llm

PYBIND11_MODULE(libmtp_multimodal_tp_sync_wrapper, module) {
    rtp_llm::registerExecCtxOps(module);
    module.def("cpu_broadcaster_initialized", &rtp_llm::isCpuTpBroadcasterInitialized);
    py::class_<rtp_llm::MtpMultimodalTpFixture>(module, "Fixture")
        .def(py::init<int>())
        .def("sync", &rtp_llm::MtpMultimodalTpFixture::sync, py::call_guard<py::gil_scoped_release>())
        .def("shift_root", &rtp_llm::MtpMultimodalTpFixture::shiftRoot, py::call_guard<py::gil_scoped_release>())
        .def("snapshot", &rtp_llm::MtpMultimodalTpFixture::snapshot);
}
