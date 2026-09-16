#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/schedulers/PPScheduler.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

template<typename T>
std::vector<T> tensorToVector(const torch::Tensor& tensor) {
    const auto cpu = tensor.cpu().contiguous();
    return std::vector<T>(cpu.data_ptr<T>(), cpu.data_ptr<T>() + cpu.numel());
}

torch::Tensor intTensor(std::vector<int32_t> values) {
    return torch::tensor(std::move(values), torch::kInt32);
}

class RecordingDraftModel: public ModelBase {
public:
    GptModelOutputs forwardPP(const GptModelInputs& input,
                              const PPIntermediateTensors*,
                              PPIntermediateTensors*) override {
        return forward(input);
    }

    GptModelOutputs forward(const GptModelInputs& input) override {
        auto recorded_input = input;
        if (input.last_hidden_states.defined()) {
            recorded_input.last_hidden_states = input.last_hidden_states.clone();
        }
        inputs.push_back(std::move(recorded_input));
        const auto step = static_cast<int64_t>(inputs.size() - 1);
        const auto rows = input.combo_tokens.numel();
        const auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
        if (!hidden_buffer.defined() || hidden_buffer.size(0) < rows) {
            hidden_buffer = torch::empty({rows, 4}, options);
        }
        hidden_buffer.narrow(0, 0, rows).copy_(torch::arange(rows * 4, options).reshape({rows, 4}) + step * 100);
        if (input.last_hidden_states.defined()) {
            EXPECT_TRUE(torch::equal(input.last_hidden_states, inputs.back().last_hidden_states));
        }
        GptModelOutputs output;
        output.all_hidden_states = hidden_buffer.narrow(0, 0, rows).narrow(1, 0, 2);
        const int64_t logit_rows = next_tokens.empty() ? input.input_lengths.numel() : next_tokens.size();
        output.logits = torch::zeros({logit_rows, 64}, options);
        for (int64_t row = 0; row < logit_rows; ++row) {
            output.logits[row][next_tokens.empty() ? 10 + step * 2 + row : next_tokens[row]] = 100;
        }
        if (invalid_logits) {
            output.logits = output.logits.narrow(0, 0, 0);
        }
        return output;
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t rows) override {
        return hidden_buffer.narrow(0, 0, rows);
    }

    std::vector<GptModelInputs> inputs;
    /** Script target logits while keeping real sampling, verification and dispatch in the test. */
    std::vector<int32_t> next_tokens;
    torch::Tensor hidden_buffer;
    bool invalid_logits = false;
};

class RecordingDSparkModel: public ModelBase {
public:
    explicit RecordingDSparkModel(int64_t propose_step): propose_step_(propose_step) {}

    GptModelOutputs
    forwardPP(const GptModelInputs& input, const PPIntermediateTensors*, PPIntermediateTensors*) override {
        return forward(input);
    }

    GptModelOutputs forward(const GptModelInputs& input) override {
        auto       recorded_input   = input;
        const auto clone_if_defined = [](const torch::Tensor& tensor) {
            return tensor.defined() ? tensor.clone() : torch::Tensor();
        };
        recorded_input.combo_tokens          = clone_if_defined(input.combo_tokens);
        recorded_input.input_lengths         = clone_if_defined(input.input_lengths);
        recorded_input.sequence_lengths      = clone_if_defined(input.sequence_lengths);
        recorded_input.prefix_lengths        = clone_if_defined(input.prefix_lengths);
        recorded_input.lm_output_indexes     = clone_if_defined(input.lm_output_indexes);
        recorded_input.last_hidden_states    = clone_if_defined(input.last_hidden_states);
        recorded_input.request_id            = clone_if_defined(input.request_id);
        recorded_input.request_pd_separation = clone_if_defined(input.request_pd_separation);
        recorded_input.cache_keys            = clone_if_defined(input.cache_keys);
        inputs.push_back(std::move(recorded_input));

        GptModelOutputs output;
        if (input.dspark_call_phase == DSparkCallPhase::PROPOSE) {
            const auto batch_size = input.input_lengths.numel();
            output.draft_tokens   = torch::arange(100,
                                                100 + batch_size * propose_step_,
                                                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
                                      .reshape({batch_size, propose_step_});
        }
        return output;
    }

    int64_t                     propose_step_;
    std::vector<GptModelInputs> inputs;
};

class RecordingLogitsProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t start, size_t finish) override {
        process_intervals.emplace_back(start, finish);
        return process_error;
    }

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override {
        ++verify_calls;
        if (verify_error.has_value()) {
            return verify_error.value();
        }
        return static_cast<int>(request.propose_step);
    }

    void updateMultiSeqStatus(const std::vector<int>&) override {}

    MtpProcessorCapability mtpCapability() const override {
        return {mtp_supported ? MtpProcessorMode::SPEC_VERIFY : MtpProcessorMode::UNSUPPORTED,
                "test processor capability"};
    }

    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t) override {
        if (on_update) {
            on_update();
        }
        committed_tokens.push_back(new_tokens.clone());
        return update_error;
    }

    std::vector<torch::Tensor> committed_tokens;
    std::vector<std::pair<size_t, size_t>> process_intervals;
    int verify_calls = 0;
    std::function<void()> on_update;
    bool mtp_supported = true;
    std::optional<ErrorInfo> process_error;
    std::optional<ErrorInfo> verify_error;
    std::optional<ErrorInfo> update_error;
};

class InMemoryP2PWork final: public P2PWork {
public:
    explicit InMemoryP2PWork(torch::Tensor tensor): tensor_(std::move(tensor)) {}

    /** The test transport copies CPU tensors before returning the ticket. */
    void wait() override {}

private:
    torch::Tensor tensor_;
};

class InMemoryPPTransport: public PPTransport {
public:
    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override {
        sent_tensors.push_back(tensor.clone());
        return std::make_unique<PPCommTicket>(std::make_unique<InMemoryP2PWork>(tensor));
    }

    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override {
        tensor.copy_(received_tensors.at(receive_index++));
        return std::make_unique<PPCommTicket>(std::make_unique<InMemoryP2PWork>(tensor));
    }

    std::vector<torch::Tensor> received_tensors;
    std::vector<torch::Tensor> sent_tensors;
    size_t receive_index = 0;
};

}  // namespace

class PPBatchStreamProcessorTest: public DeviceTestBase {
protected:
    static ModelConfig makeModelConfig() {
        ModelConfig model_config;
        model_config.max_seq_len      = 32;
        model_config.vocab_size       = 64;
        model_config.input_vocab_size = 64;
        model_config.num_layers       = 1;
        return model_config;
    }

    /** Individual stage tests initialize the fields normally owned by sampleTokens(). */
    static PPExecutionResult makeInitializedResult(const PPSamplingPlan& plan) {
        PPExecutionResult result;
        const auto stream_count = plan.request_ids.size(0);
        result.request_ids = plan.request_ids.to(torch::kCPU).contiguous();
        result.request_errors.assign(stream_count, ErrorInfo::OkStatus());
        result.prompt_logits.resize(stream_count);
        return result;
    }

    static EngineInitParams makeMtpParams(size_t num_draft_tokens) {
        EngineInitParams params;
        params.model_id = 0;
        params.model_config_ = makeModelConfig();
        params.model_config_.num_layers = 2;
        params.parallelism_config.pp_size = 2;
        params.parallelism_config.pp_stage_layer_counts = {1, 1};
        params.sp_config.type = SP_TYPE_MTP;
        params.sp_config.gen_num_per_cycle = num_draft_tokens;
        params.py_model = py::none();
        params.py_sp_model = py::none();
        return params;
    }

    static std::unique_ptr<ProposeModelEngineInitParams> makeMtpProposeParams(const EngineInitParams& params,
                                                                              size_t model_count = 1) {
        auto mtp_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        for (size_t i = 0; i < model_count; ++i) {
            auto draft                  = std::make_unique<EngineInitParams>();
            draft->model_id             = i + 1;
            draft->model_config_        = makeModelConfig();
            draft->parallelism_config   = params.parallelism_config;
            draft->sp_config            = params.sp_config;
            draft->py_model             = py::none();
            draft->py_sp_model          = py::none();
            mtp_params->emplace_back(std::move(draft));
        }
        return std::make_unique<ProposeModelEngineInitParams>(
            params.sp_config.type, params.sp_config.gen_num_per_cycle, std::move(mtp_params));
    }

    static GenerateStreamPtr makeStream(const ResourceContext& resource_context,
                                        const ModelConfig&     model_config,
                                        int64_t                request_id,
                                        std::vector<int32_t>   input_ids,
                                        int32_t                num_return_sequences) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->request_id                            = request_id;
        input->input_ids                             = intTensor(std::move(input_ids));
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->num_return_sequences = num_return_sequences;

        RuntimeConfig runtime_config;
        auto          stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }
};

TEST_F(PPBatchStreamProcessorTest, StageWithProposeParamsBuildsOnlyFirstConfiguredMtpModel) {
    auto params                            = makeMtpParams(2);
    params.parallelism_config.pp_rank      = 1;
    params.parallelism_config.world_rank   = 1;
    params.parallelism_config.world_size   = 2;
    auto propose_params                    = makeMtpProposeParams(params, 2);
    std::vector<size_t> constructed_models = {};
    struct FactoryReset {
        ~FactoryReset() {
            PPExecutor::test_model_factory = nullptr;
        }
    } factory_reset;
    PPExecutor::test_model_factory = [&constructed_models](const GptModelInitParams& init_params) {
        constructed_models.push_back(init_params.model_id);
        return std::make_unique<RecordingDraftModel>();
    };

    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose_params.get());

    EXPECT_EQ(constructed_models, (std::vector<size_t>{0, 1}));
}

TEST_F(PPBatchStreamProcessorTest, NonOwnerStageKeepsSpEnabledWithoutBuildingDraftModel) {
    auto params                            = makeMtpParams(2);
    params.parallelism_config.pp_rank      = 0;
    params.parallelism_config.world_rank   = 0;
    params.parallelism_config.world_size   = 2;
    std::vector<size_t> constructed_models = {};
    struct FactoryReset {
        ~FactoryReset() {
            PPExecutor::test_model_factory = nullptr;
        }
    } factory_reset;
    PPExecutor::test_model_factory = [&constructed_models](const GptModelInitParams& init_params) {
        constructed_models.push_back(init_params.model_id);
        return std::make_unique<RecordingDraftModel>();
    };

    PPExecutor executor(params, nullptr, false);

    EXPECT_EQ(constructed_models, (std::vector<size_t>{0}));
    EXPECT_TRUE(executor.sp_enabled_);
    EXPECT_TRUE(executor.batch_stream_processor_->sp_enabled_);
    EXPECT_EQ(executor.draft_model_, nullptr);
}

TEST_F(PPBatchStreamProcessorTest, ProposeDraftTokensRunsAllForwardsAndPreservesMtpHiddenStates) {
    for (int64_t count : {1, 3, 4}) {
        SCOPED_TRACE(count);
        auto params = makeMtpParams(count);
        PPExecutor executor(params, nullptr, true);
        executor.position_id_len_factor_ = 2;
        auto model = std::make_unique<RecordingDraftModel>();
        auto* recorded_model = model.get();
        executor.draft_model_ = std::move(model);
        executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>();

        GptModelInputs initial_input;
        initial_input.combo_tokens = intTensor({1, 2, 3, 4, 5});
        initial_input.input_lengths = intTensor({2, 3});
        initial_input.prefix_lengths = intTensor({3, 5});
        initial_input.sequence_lengths = intTensor({});
        initial_input.lm_output_indexes = intTensor({1, 4});
        initial_input.combo_position_ids = intTensor({3, 13, 4, 14, 5, 15, 6, 16, 7, 17});
        initial_input.request_id = torch::tensor({101, 202}, torch::kInt64);
        initial_input.request_pd_separation = torch::ones({2}, torch::kBool);
        initial_input.cache_keys = torch::ones({2, 3}, torch::kInt64);

        auto proposed = executor.proposeDraftTokens(initial_input, count);
        ASSERT_EQ(proposed.sizes().vec(), (std::vector<int64_t>{2, count}));
        EXPECT_TRUE(proposed.device().is_cpu());
        EXPECT_EQ(proposed.scalar_type(), torch::kInt32);
        ASSERT_EQ(recorded_model->inputs.size(), count);
        for (int64_t step = 0; step < count; ++step) {
            EXPECT_EQ(tensorToVector<int32_t>(proposed.select(1, step)),
                      (std::vector<int32_t>{static_cast<int32_t>(10 + step * 2), static_cast<int32_t>(11 + step * 2)}));
            const auto& input = recorded_model->inputs[step];
            EXPECT_FALSE(input.is_target_verify);
            if (step == 0) {
                EXPECT_TRUE(torch::equal(input.combo_tokens, initial_input.combo_tokens));
                continue;
            }
            EXPECT_TRUE(input.combo_position_ids.is_pinned());
            EXPECT_EQ(input.combo_tokens.numel(), 2);
            EXPECT_EQ(input.prefix_lengths.numel(), 0);
            EXPECT_FALSE(input.request_id.defined());
            EXPECT_FALSE(input.request_pd_separation.defined());
            EXPECT_FALSE(input.cache_keys.defined());
            EXPECT_EQ(tensorToVector<int32_t>(input.lm_output_indexes), (std::vector<int32_t>{0, 1}));
            EXPECT_TRUE(torch::equal(input.combo_tokens.cpu(), proposed.select(1, step - 1)));
            EXPECT_TRUE(torch::equal(input.sequence_lengths.cpu(), intTensor({5, 8}) + step - 1));
            EXPECT_EQ(tensorToVector<int32_t>(input.input_lengths), (std::vector<int32_t>{1, 1}));
            EXPECT_TRUE(torch::equal(input.combo_position_ids.cpu(), intTensor({5, 15, 8, 18}) + step - 1));
            auto expected_hidden = step == 1 ? torch::tensor({{4.f, 5.f, 6.f, 7.f}, {16.f, 17.f, 18.f, 19.f}}) :
                                              torch::arange(8, torch::kFloat32).reshape({2, 4}) + (step - 1) * 100;
            EXPECT_TRUE(torch::equal(input.last_hidden_states.cpu(), expected_hidden));
        }
        EXPECT_EQ(tensorToVector<int32_t>(initial_input.input_lengths), (std::vector<int32_t>{2, 3}));
        EXPECT_EQ(tensorToVector<int32_t>(initial_input.prefix_lengths), (std::vector<int32_t>{3, 5}));
    }
}

TEST_F(PPBatchStreamProcessorTest, DSparkDecodeCommitsThenProposesFromAcceptedRoundState) {
    constexpr int32_t gamma                  = 3;
    constexpr int32_t mask_id                = 99;
    auto              params                 = makeMtpParams(gamma);
    params.sp_config.type                    = SP_TYPE_DSPARK;
    params.sp_config.sp_dspark_mask_token_id = mask_id;
    PPExecutor executor(params, nullptr, true);

    auto target_model = std::make_unique<RecordingDraftModel>();
    target_model->hidden_buffer =
        torch::arange(32, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({8, 4});
    auto target_features = target_model->hidden_buffer.clone();
    executor.setModel(std::move(target_model));

    auto  draft_model     = std::make_unique<RecordingDSparkModel>(gamma);
    auto* recorded_draft  = draft_model.get();
    executor.draft_model_ = std::move(draft_model);

    PPExecutionPlan plan;
    plan.is_decode                            = true;
    plan.model_input.combo_tokens             = intTensor({1, 2, 3, 4, 5, 6, 7, 8});
    plan.model_input.input_lengths            = intTensor({4, 4});
    plan.model_input.prefix_lengths           = intTensor({5, 9});
    plan.model_input.sequence_lengths         = intTensor({});
    plan.model_input.lm_output_indexes        = torch::arange(8, torch::kInt32);
    plan.model_input.is_target_verify         = true;
    plan.model_input.request_id               = torch::tensor({101, 202}, torch::kInt64);
    plan.model_input.request_pd_separation    = torch::ones({2}, torch::kBool);
    plan.model_input.cache_keys               = torch::tensor({11, 12, 21, 22}, torch::kInt64).reshape({2, 2});

    GptModelOutputs   target_output;
    PPExecutionResult result;
    result.new_token_ids = intTensor({20, 21, 22, 23, 30, 31, 32, 33}).reshape({2, 4});
    result.accept_len    = intTensor({2, 4});
    result.request_errors.resize(2);

    executor.draftSampleAndPropose(plan, target_output, result);

    ASSERT_EQ(recorded_draft->inputs.size(), 2u);
    const auto& commit_input = recorded_draft->inputs[0];
    EXPECT_EQ(commit_input.dspark_call_phase, DSparkCallPhase::COMMIT);
    EXPECT_FALSE(commit_input.is_target_verify);
    EXPECT_TRUE(torch::equal(commit_input.combo_tokens, plan.model_input.combo_tokens));
    EXPECT_TRUE(torch::equal(commit_input.input_lengths, plan.model_input.input_lengths));
    EXPECT_TRUE(torch::equal(commit_input.prefix_lengths, plan.model_input.prefix_lengths));
    EXPECT_TRUE(torch::equal(commit_input.lm_output_indexes, plan.model_input.lm_output_indexes));
    EXPECT_TRUE(torch::equal(commit_input.last_hidden_states, target_features));
    EXPECT_TRUE(torch::equal(commit_input.request_id, plan.model_input.request_id));
    EXPECT_TRUE(torch::equal(commit_input.request_pd_separation, plan.model_input.request_pd_separation));
    EXPECT_TRUE(torch::equal(commit_input.cache_keys, plan.model_input.cache_keys));

    const auto& propose_input = recorded_draft->inputs[1];
    EXPECT_EQ(propose_input.dspark_call_phase, DSparkCallPhase::PROPOSE);
    EXPECT_FALSE(propose_input.is_target_verify);
    EXPECT_FALSE(propose_input.last_hidden_states.defined());
    EXPECT_TRUE(propose_input.sequence_lengths.is_cuda());
    EXPECT_EQ(propose_input.sequence_lengths.numel(), 0);
    EXPECT_EQ(tensorToVector<int32_t>(propose_input.combo_tokens),
              (std::vector<int32_t>{21, mask_id, mask_id, 33, mask_id, mask_id}));
    EXPECT_EQ(tensorToVector<int32_t>(propose_input.input_lengths), (std::vector<int32_t>{gamma, gamma}));
    EXPECT_EQ(tensorToVector<int32_t>(propose_input.prefix_lengths), (std::vector<int32_t>{7, 13}));
    EXPECT_EQ(tensorToVector<int32_t>(propose_input.lm_output_indexes), (std::vector<int32_t>{0, gamma}));
    EXPECT_FALSE(propose_input.request_id.defined());
    EXPECT_FALSE(propose_input.request_pd_separation.defined());
    EXPECT_FALSE(propose_input.cache_keys.defined());
    EXPECT_TRUE(result.propose_token_ids.device().is_cpu());
    EXPECT_EQ(result.propose_token_ids.scalar_type(), torch::kInt32);
    EXPECT_EQ(tensorToVector<int32_t>(result.propose_token_ids), (std::vector<int32_t>{100, 101, 102, 103, 104, 105}));
}

TEST_F(PPBatchStreamProcessorTest, DSparkPrefillCommitsThenProposesFromSampledTokens) {
    constexpr int32_t gamma                  = 3;
    constexpr int32_t mask_id                = 99;
    auto              params                 = makeMtpParams(gamma);
    params.sp_config.type                    = SP_TYPE_DSPARK;
    params.sp_config.sp_dspark_mask_token_id = mask_id;
    PPExecutor executor(params, nullptr, true);

    auto target_model = std::make_unique<RecordingDraftModel>();
    target_model->hidden_buffer =
        torch::arange(20, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({5, 4});
    executor.setModel(std::move(target_model));

    auto  draft_model     = std::make_unique<RecordingDSparkModel>(gamma);
    auto* recorded_draft  = draft_model.get();
    executor.draft_model_ = std::move(draft_model);

    PPExecutionPlan plan;
    plan.model_input.combo_tokens             = intTensor({1, 2, 3, 4, 5});
    plan.model_input.input_lengths            = intTensor({2, 3});
    plan.model_input.prefix_lengths           = intTensor({4, 7});
    plan.model_input.sequence_lengths         = intTensor({});
    plan.model_input.lm_output_indexes        = intTensor({1, 4});
    plan.model_input.request_id               = torch::tensor({101, 202}, torch::kInt64);
    plan.model_input.request_pd_separation    = torch::ones({2}, torch::kBool);
    plan.model_input.cache_keys               = torch::tensor({11, 12, 21, 22}, torch::kInt64).reshape({2, 2});

    GptModelOutputs   target_output;
    PPExecutionResult result;
    result.new_token_ids = intTensor({40, 50}).reshape({2, 1});
    result.accept_len    = intTensor({1, 1});
    result.request_errors.resize(2);

    executor.draftSampleAndPropose(plan, target_output, result);

    ASSERT_EQ(recorded_draft->inputs.size(), 2u);
    EXPECT_EQ(recorded_draft->inputs[0].dspark_call_phase, DSparkCallPhase::COMMIT);
    const auto& propose_input = recorded_draft->inputs[1];
    EXPECT_EQ(propose_input.dspark_call_phase, DSparkCallPhase::PROPOSE);
    EXPECT_EQ(tensorToVector<int32_t>(propose_input.combo_tokens),
              (std::vector<int32_t>{40, mask_id, mask_id, 50, mask_id, mask_id}));
    EXPECT_EQ(tensorToVector<int32_t>(propose_input.prefix_lengths), (std::vector<int32_t>{6, 10}));
    EXPECT_FALSE(propose_input.request_id.defined());
    EXPECT_FALSE(propose_input.request_pd_separation.defined());
    EXPECT_FALSE(propose_input.cache_keys.defined());
    EXPECT_EQ(tensorToVector<int32_t>(result.propose_token_ids), (std::vector<int32_t>{100, 101, 102, 103, 104, 105}));
}

TEST_F(PPBatchStreamProcessorTest, RequestErrorsStillDraftAndExecutionExceptionsPropagate) {
    for (const std::string error_stage : {"initialize", "process", "update", "execute"}) {
        SCOPED_TRACE(error_stage);
        auto params = makeMtpParams(3);
        PPExecutor first_stage(params, nullptr, true);
        ResourceContext resource_context;
        auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
        std::list<GenerateStreamPtr> streams_to_prepare{stream};
        first_stage.prepareStreams(streams_to_prepare);
        auto plan = first_stage.buildPlan(StreamGroups({stream}), {});
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        ASSERT_FALSE(plan->is_decode);
        if (error_stage == "initialize") {
            plan->sampling_plan.logits_processor_configs[0].grammar_type = "invalid";
        }

        params.pd_sep_config.role_type = RoleType::DECODE;
        params.parallelism_config.pp_rank = 1;
        params.parallelism_config.world_rank = 1;
        params.parallelism_config.world_size = 2;
        auto propose_params = makeMtpProposeParams(params);
        PPExecutor last_stage(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose_params.get());
        auto target_model = std::make_unique<RecordingDraftModel>();
        auto* recorded_target = target_model.get();
        recorded_target->invalid_logits = error_stage == "execute";
        last_stage.setModel(std::move(target_model));
        auto draft_model = std::make_unique<RecordingDraftModel>();
        auto* recorded_draft = draft_model.get();
        last_stage.draft_model_ = std::move(draft_model);
        auto processor = std::make_shared<RecordingLogitsProcessor>();
        if (error_stage == "process" || error_stage == "update") {
            if (error_stage == "update") {
                processor->on_update = [recorded_draft]() { EXPECT_EQ(recorded_draft->inputs.size(), 3); };
                processor->update_error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "injected processor update failure");
            } else {
                processor->process_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected processor failure");
            }
            auto& state = last_stage.sampling_states_[101];
            state.logits_processors = {processor};
            state.cum_log_probs = torch::zeros({1}, torch::kFloat32);
        }
        auto transport = std::make_unique<InMemoryPPTransport>();
        auto* recorded_transport = transport.get();
        for (const auto& object : {pp_serialization::serializePlan(plan.value(), false),
                                   pp_serialization::serializeTensorsMetadata(PPIntermediateTensors{})}) {
            transport->received_tensors.push_back(torch::tensor({object.numel()}, torch::kInt64));
            transport->received_tensors.push_back(object);
        }
        last_stage.transport_ = std::move(transport);

        if (error_stage == "execute") {
            EXPECT_THROW(static_cast<void>(last_stage.process(ScheduleOutput{})), std::exception);
            EXPECT_EQ(recorded_target->inputs.size(), 1);
            EXPECT_TRUE(recorded_draft->inputs.empty());
            EXPECT_TRUE(recorded_transport->sent_tensors.empty());
            continue;
        }

        const auto status = last_stage.process(ScheduleOutput{});
        EXPECT_TRUE(status.ok()) << status.ToString();
        EXPECT_EQ(recorded_target->inputs.size(), 1);
        EXPECT_EQ(recorded_draft->inputs.size(), 3);
        ASSERT_EQ(recorded_transport->sent_tensors.size(), 2);
        const auto result = pp_serialization::deserializeExecutionResult(recorded_transport->sent_tensors[1]);
        ASSERT_EQ(result.request_errors.size(), 1);
        const auto& error = result.request_errors[0];
        ASSERT_TRUE(error.hasError());
        EXPECT_EQ(error.code(),
                  error_stage == "initialize" ? ErrorCode::INVALID_PARAMS :
                  error_stage == "process" ? ErrorCode::EXECUTION_EXCEPTION : ErrorCode::UNKNOWN_ERROR);
        EXPECT_EQ(result.accept_len[0].item<int32_t>(), 1);
        if (error_stage == "initialize" || error_stage == "process") {
            EXPECT_TRUE(torch::equal(result.new_token_ids, torch::zeros_like(result.new_token_ids)));
            EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[0].combo_tokens),
                      (std::vector<int32_t>{2, 0}));
        }
        ASSERT_TRUE(result.propose_token_ids.defined());
        EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{1, 3}));
        const auto proposals_before_dispatch = stream->getSPOutputBuffer()->tokens.clone();
        stream->setPPInflight();
        ASSERT_TRUE(first_stage.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_TRUE(stream->hasError());
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
        EXPECT_TRUE(torch::equal(stream->getSPOutputBuffer()->tokens, proposals_before_dispatch));
        EXPECT_EQ(processor->committed_tokens.size(), error_stage == "update" ? 1 : 0);
        EXPECT_EQ(last_stage.sampling_states_.count(101), 1);
        PPExecutionPlan cleanup;
        cleanup.finished_request_ids = {101};
        cleanup.model_input.skip_run = true;
        const auto cleanup_payload = pp_serialization::serializePlan(cleanup, false);
        recorded_transport->received_tensors.push_back(torch::tensor({cleanup_payload.numel()}, torch::kInt64));
        recorded_transport->received_tensors.push_back(cleanup_payload);
        ASSERT_TRUE(last_stage.process(ScheduleOutput{}).ok());
        EXPECT_EQ(recorded_draft->inputs.size(), 3);
        EXPECT_TRUE(last_stage.sampling_states_.empty());
    }
}

TEST_F(PPBatchStreamProcessorTest, SamplingAndDraftProposalLeaveHistoryForExplicitCommit) {
    for (bool mtp_enabled : {false, true}) {
        SCOPED_TRACE(mtp_enabled);
        auto params = makeMtpParams(3);
        params.sp_config.type = mtp_enabled ? SP_TYPE_MTP : SP_TYPE_NONE;
        PPExecutor executor(params, nullptr, true);
        executor.sampler_ = std::make_unique<Sampler>(SamplerInitParams{2, false});
        ResourceContext resource_context;
        const int64_t batch_size = mtp_enabled ? 1 : 2;
        auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, batch_size);
        stream->generateConfig()->do_sample = false;
        auto plan = executor.buildPlan(StreamGroups({stream}), {});
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        plan->output_config.return_cum_log_probs = true;

        auto processor = std::make_shared<RecordingLogitsProcessor>();
        auto& state = executor.sampling_states_[101];
        state.logits_processors.push_back(processor);
        state.cum_log_probs = torch::zeros({batch_size}, torch::kFloat32);
        GptModelOutputs output;
        output.logits = torch::zeros({batch_size, 64}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        for (int64_t row = 0; row < batch_size; ++row) {
            output.logits[row][20 + row] = 100;
        }
        PPExecutionResult result;
        executor.sampleTokens(plan.value(), output, result);
        EXPECT_TRUE(torch::equal(result.new_token_ids.flatten(), torch::arange(20, 20 + batch_size, torch::kInt32)));
        EXPECT_EQ(result.new_token_ids.sizes().vec(), (std::vector<int64_t>{batch_size, 1}));
        EXPECT_EQ(result.accept_len.defined(), mtp_enabled);
        EXPECT_FALSE(result.propose_token_ids.defined());
        EXPECT_TRUE(processor->committed_tokens.empty());
        if (mtp_enabled) {
            EXPECT_EQ(tensorToVector<int32_t>(result.accept_len), (std::vector<int32_t>{1}));
            auto target_model = std::make_unique<RecordingDraftModel>();
            target_model->hidden_buffer = torch::arange(8, output.logits.options()).reshape({2, 4});
            executor.setModel(std::move(target_model));
            auto draft_model = std::make_unique<RecordingDraftModel>();
            auto* recorded_draft = draft_model.get();
            executor.draft_model_ = std::move(draft_model);
            executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>();
            executor.draftSampleAndPropose(plan.value(), output, result);
            EXPECT_EQ(tensorToVector<int32_t>(result.propose_token_ids), (std::vector<int32_t>{10, 12, 14}));
            ASSERT_EQ(recorded_draft->inputs.size(), 3);
            EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[0].combo_tokens), (std::vector<int32_t>{2, 20}));
            EXPECT_TRUE(processor->committed_tokens.empty());
            EXPECT_EQ(tensorToVector<int32_t>(plan->model_input.combo_tokens), (std::vector<int32_t>{1, 2}));
        }
        executor.advanceSamplingStates(plan->sampling_plan, result);
        ASSERT_EQ(processor->committed_tokens.size(), 1);
        EXPECT_TRUE(torch::equal(processor->committed_tokens[0], result.new_token_ids));
        EXPECT_TRUE(torch::equal(state.cum_log_probs, result.cum_log_probs));
    }
}

TEST_F(PPBatchStreamProcessorTest, PdPrefillPublishesOnlyD1ForMtpAndEagle) {
    for (int64_t count : {1, 3, 4}) {
        for (const auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE}) {
            for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
                SCOPED_TRACE("K=" + std::to_string(count) + ", type=" + std::to_string(type)
                             + ", role=" + std::to_string(role));
                auto params = makeMtpParams(count);
                params.sp_config.type = type;
                params.pd_sep_config.role_type = role;
                auto cache_config = test::makeSimpleMhaCacheConfig(1, 16, 4, DataType::TYPE_FP16);
                const auto draft_config = cache_config;
                cache_config.block_size_bytes += draft_config.block_size_bytes;
                cache_config.mtp_sub_configs.push_back(cache_config.mergeMTPModule(draft_config, 0, 1));
                cache_config.finalizeBlockNums(16, RuntimeConfig{});
                auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
                ASSERT_TRUE(cache_manager->init());
                PPExecutor executor(params, cache_manager, true);
                ResourceContext resource_context;
                resource_context.cache_manager = cache_manager;
                resource_context.role_type = role;
                auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
                stream->generateConfig()->max_new_tokens = 20;
                stream->fakeInitKVBlock(4);
                std::list<GenerateStreamPtr> streams{stream};
                executor.prepareStreams(streams);
                auto plan_status = executor.buildPlan(StreamGroups(streams), {});
                ASSERT_TRUE(plan_status.ok()) << plan_status.status().ToString();
                auto plan =
                    pp_serialization::deserializePlan(pp_serialization::serializePlan(plan_status.value(), false));
                ASSERT_FALSE(plan.is_decode);
                EXPECT_EQ(plan.model_input.pd_separation, role == RoleType::PREFILL);

                auto target_model = std::make_unique<RecordingDraftModel>();
                target_model->hidden_buffer =
                    torch::arange(8, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA)).reshape({2, 4});
                executor.setModel(std::move(target_model));
                auto draft_model = std::make_unique<RecordingDraftModel>();
                auto* recorded_draft = draft_model.get();
                executor.draft_model_ = std::move(draft_model);
                const int32_t vocab_offset = type == SP_TYPE_EAGLE ? 17 : 0;
                torch::Tensor d2t_map;
                if (vocab_offset) {
                    d2t_map =
                        (torch::arange(64, torch::TensorOptions(torch::kLong).device(torch::kCUDA)) + vocab_offset)
                            .remainder(64);
                }
                executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>(d2t_map);
                auto result = makeInitializedResult(plan.sampling_plan);
                result.new_token_ids = intTensor({20}).reshape({1, 1});
                result.accept_len = intTensor({1});
                executor.draftSampleAndPropose(plan, GptModelOutputs{}, result);

                const int64_t draft_count = role == RoleType::PREFILL ? 1 : count;
                ASSERT_EQ(recorded_draft->inputs.size(), draft_count);
                EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[0].combo_tokens),
                          (std::vector<int32_t>{2, 20}));
                result =
                    pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
                EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{1, draft_count}));
                EXPECT_TRUE(torch::equal(result.propose_token_ids.flatten(),
                                         torch::arange(10, 10 + 2 * draft_count, 2, torch::kInt32) + vocab_offset));

                ASSERT_TRUE(
                    executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups(streams), result).ok());
                ASSERT_FALSE(stream->hasError());
                EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 20}));
                EXPECT_EQ(stream->getSPOutputBuffer()->propose_step, count);
                EXPECT_EQ(stream->getSPOutputBuffer()->tokens.numel(), count + 1);
                std::vector<int> expected_proposals{20};
                for (int64_t step = 0; step < draft_count; ++step) {
                    expected_proposals.push_back(10 + 2 * step + vocab_offset);
                }
                EXPECT_EQ(stream->getProposeToken(), expected_proposals);
            }
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, DecodeSamplingCommitsOnlyAcceptedTokensAndSkipsFailedRequests) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    executor.sampler_ = std::make_unique<Sampler>(SamplerInitParams{8, false});
    executor.speculative_sampler_ = std::make_unique<speculative::SpeculativeSampler>(torch::Tensor(), 3);
    ResourceContext resource_context;
    auto first = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
    auto second = makeStream(resource_context, params.model_config_, 202, {3, 4}, 0);
    for (const auto& stream : {first, second}) {
        stream->generateConfig()->do_sample = true;
        stream->generateConfig()->top_k = 2;
        stream->generateConfig()->top_p = 1.0f;
    }
    PPExecutionPlan plan;
    plan.is_decode = true;
    plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({first, second}));
    plan.sampling_plan.spec_do_sample.fill_(false);
    plan.model_input.combo_tokens = intTensor({2, 11, 12, 13, 4, 21, 22, 23});
    GptModelOutputs output;
    // Use probability sampling; the all-top-k=1 path always reports success.
    output.logits = torch::full({8, 64}, -std::numeric_limits<float>::infinity(),
                               torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    const std::vector<int64_t> target_tokens{11, 12, 13, 14, 29, 22, 23, 24};
    for (int64_t row = 0; row < 8; ++row) {
        output.logits[row][target_tokens[row]] = 100;
    }
    output.logits[7].fill_(-std::numeric_limits<float>::infinity());  // Unused suffix after row 4 rejects.
    PPExecutionResult result;
    executor.sampleTokens(plan, output, result);
    EXPECT_EQ(result.new_token_ids.sizes().vec(), (std::vector<int64_t>{2, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(result.accept_len), (std::vector<int32_t>{4, 1}));
    ASSERT_EQ(result.request_errors.size(), 2);
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_FALSE(result.propose_token_ids.defined());

    auto first_processor = std::make_shared<RecordingLogitsProcessor>();
    auto second_processor = std::make_shared<RecordingLogitsProcessor>();
    executor.sampling_states_.at(101).logits_processors = {first_processor};
    executor.sampling_states_.at(202).logits_processors = {second_processor};
    executor.advanceSamplingStates(plan.sampling_plan, result);
    ASSERT_EQ(first_processor->committed_tokens.size(), 1);
    ASSERT_EQ(second_processor->committed_tokens.size(), 1);
    EXPECT_EQ(tensorToVector<int32_t>(first_processor->committed_tokens[0]), (std::vector<int32_t>{11, 12, 13, 14}));
    EXPECT_EQ(tensorToVector<int32_t>(second_processor->committed_tokens[0]), (std::vector<int32_t>{29}));

    result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "processor failed");
    result.request_errors[1] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
    executor.advanceSamplingStates(plan.sampling_plan, result);
    EXPECT_EQ(first_processor->committed_tokens.size(), 1);
    EXPECT_EQ(second_processor->committed_tokens.size(), 1);
}

TEST_F(PPBatchStreamProcessorTest, DecodeSamplingFailureBeyondLengthCapDoesNotFailRequest) {
    for (int32_t max_new_tokens : {3, 4}) {
        SCOPED_TRACE(max_new_tokens);
        auto params = makeMtpParams(3);
        PPExecutor executor(params, nullptr, true);
        executor.sampler_ = std::make_unique<Sampler>(SamplerInitParams{4, false});
        executor.speculative_sampler_ = std::make_unique<speculative::SpeculativeSampler>(torch::Tensor(), 3);
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
        stream->generateConfig()->do_sample = true;
        stream->generateConfig()->top_k = 2;
        stream->generateConfig()->top_p = 1.0f;
        stream->generateConfig()->max_new_tokens = max_new_tokens;
        PPExecutionPlan plan;
        plan.is_decode = true;
        plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({stream}));
        plan.sampling_plan.spec_do_sample.fill_(false);
        plan.model_input.combo_tokens = intTensor({2, 11, 12, 13});
        GptModelOutputs output;
        // One valid token per prefix row keeps probability sampling deterministic.
        output.logits = torch::full({4, 64}, -std::numeric_limits<float>::infinity(),
                                   torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        for (int64_t row = 0; row < 3; ++row) {
            output.logits[row][11 + row] = 100;
        }
        // All three proposals match. Only the bonus row fails sampling.
        output.logits[3].fill_(-std::numeric_limits<float>::infinity());
        PPExecutionResult result;
        executor.sampleTokens(plan, output, result);
        ASSERT_EQ(result.request_errors.size(), 1);
        EXPECT_EQ(result.accept_len.item<int32_t>(), 4);
        EXPECT_EQ(result.request_errors[0].code(),
                  max_new_tokens == 3 ? ErrorCode::NONE_ERROR : ErrorCode::UNKNOWN_ERROR);
        executor.clipMtpAcceptedLengths(plan.sampling_plan, result);
        EXPECT_EQ(result.accept_len.item<int32_t>(), max_new_tokens);
    }
}

TEST_F(PPBatchStreamProcessorTest, DecodeRequestErrorsKeepInitializationAndVerifyFailuresBeforeSamplerFailure) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    executor.sampler_ = std::make_unique<Sampler>(SamplerInitParams{12, false});
    executor.speculative_sampler_ = std::make_unique<speculative::SpeculativeSampler>(torch::Tensor(), 3);
    executor.spec_logits_verify_runner_ = std::make_unique<SpecLogitsVerifyRunner>();
    auto first = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
    auto second = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 0);
    auto third = makeStream(ResourceContext{}, params.model_config_, 303, {5, 6}, 0);
    PPExecutionPlan plan;
    plan.is_decode = true;
    plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({first, second, third}));
    plan.sampling_plan.do_sample.fill_(true);
    plan.sampling_plan.top_k.fill_(2);
    plan.sampling_plan.top_p.fill_(1.0f);
    plan.sampling_plan.spec_do_sample.fill_(false);
    plan.sampling_plan.logits_processor_configs[0].grammar_type = "invalid";
    plan.model_input.combo_tokens = intTensor({2, 11, 12, 13, 4, 21, 22, 23, 6, 31, 32, 33});
    auto processor = std::make_shared<RecordingLogitsProcessor>();
    const ErrorInfo verify_error(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, "injected verify failure");
    processor->verify_error = verify_error;
    auto& state = executor.sampling_states_[202];
    state.logits_processors = {processor};
    state.cum_log_probs = torch::zeros({1}, torch::kFloat32);
    GptModelOutputs output;
    output.logits = torch::full({12, 64}, -std::numeric_limits<float>::infinity(),
                                torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    PPExecutionResult result;
    executor.sampleTokens(plan, output, result);
    ASSERT_EQ(result.request_errors.size(), 3);
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(result.request_errors[1].code(), verify_error.code());
    EXPECT_EQ(result.request_errors[1].ToString(), verify_error.ToString());
    EXPECT_EQ(result.request_errors[2].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(result.request_errors[2].ToString(), "sampler generate token id failed");
    executor.advanceSamplingStates(plan.sampling_plan, result);
    EXPECT_TRUE(processor->committed_tokens.empty());
    EXPECT_EQ(result.request_errors[1].ToString(), verify_error.ToString());
}

TEST_F(PPBatchStreamProcessorTest, CompactDraftInputAdvancesAcrossTwoRounds) {
    for (int32_t count : {1, 3, 4}) {
        SCOPED_TRACE("K=" + std::to_string(count));
        auto params = makeMtpParams(count);
        PPExecutor executor(params, nullptr, true);
        executor.position_id_len_factor_ = 2;
        const int32_t width = count + 1;
        auto target = std::make_unique<RecordingDraftModel>();
        const auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
        target->hidden_buffer = torch::arange(3 * width * 4, options).reshape({3 * width, 4});
        auto target_hidden = target->hidden_buffer.clone();
        executor.setModel(std::move(target));
        executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>();
        auto prefixes = intTensor({5, 9, 12});
        for (int round = 0; round < 2; ++round) {
            const auto lengths = round == 0 ? intTensor({1, std::min(3, width), width}) :
                                              intTensor({width, 1, 2});
            auto draft = std::make_unique<RecordingDraftModel>();
            auto* recorded = draft.get();
            executor.draft_model_ = std::move(draft);
            GptModelInputs verify;
            verify.is_target_verify = true;
            verify.input_lengths = torch::full({3}, width, torch::kInt32);
            verify.prefix_lengths = prefixes.clone();
            verify.sequence_lengths = intTensor({});
            verify.combo_tokens = torch::arange(3 * width, torch::kInt32);
            verify.lm_output_indexes = torch::arange(3 * width, torch::kInt32);
            auto positions = prefixes.unsqueeze(1) + torch::arange(width, torch::kInt32).unsqueeze(0);
            verify.combo_position_ids = torch::stack({positions, positions + 100}, -1).flatten();
            auto accepted = (verify.combo_tokens + 20).reshape({3, width});
            auto input = executor.prepareDraftInputForDecode(verify, GptModelOutputs{}, accepted, lengths);
            const auto last_rows = lengths.cumsum(0).to(torch::kInt32) - 1;
            EXPECT_TRUE(torch::equal(input.input_lengths.cpu(), lengths));
            EXPECT_TRUE(torch::equal(input.lm_output_indexes.cpu(), last_rows));
            auto expected_last_hidden = target_hidden.index_select(
                0, (torch::arange(3, torch::kInt32) * width + lengths - 1).to(torch::kCUDA, torch::kLong));
            EXPECT_TRUE(torch::equal(input.last_hidden_states.index_select(
                                         0, last_rows.to(input.last_hidden_states.device(), torch::kLong)),
                                     expected_last_hidden));
            auto proposals = executor.proposeDraftTokens(input, count);
            ASSERT_EQ(recorded->inputs.size(), count);
            EXPECT_EQ(proposals.sizes().vec(), (std::vector<int64_t>{3, count}));
            const auto next_positions = prefixes + lengths;
            if (count == 3 && round == 0) {
                EXPECT_EQ(tensorToVector<int32_t>(next_positions), (std::vector<int32_t>{6, 12, 16}));
            }
            for (int32_t step = 1; step < count; ++step) {
                const auto& next = recorded->inputs[step];
                EXPECT_TRUE(torch::equal(next.sequence_lengths.cpu(), next_positions + step - 1));
                EXPECT_EQ(tensorToVector<int32_t>(next.input_lengths), (std::vector<int32_t>{1, 1, 1}));
                EXPECT_TRUE(torch::equal(next.combo_position_ids.cpu(),
                                         torch::stack({next_positions, next_positions + 100}, -1).flatten()
                                             + step - 1));
                EXPECT_EQ(next.prefix_lengths.numel(), 0);
            }
            EXPECT_TRUE(verify.is_target_verify);
            EXPECT_TRUE(torch::equal(verify.input_lengths, torch::full({3}, width, torch::kInt32)));
            EXPECT_TRUE(torch::equal(verify.prefix_lengths, prefixes));
            prefixes = next_positions;
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, VerifySamplingAppliesPenaltiesAndNgramsToEachCandidatePrefix) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    ResourceContext context;
    auto first = makeStream(context, params.model_config_, 101, {1, 2, 1}, 0);
    auto second = makeStream(context, params.model_config_, 202, {7, 8}, 0);
    first->generateConfig()->repetition_penalty = 1.7;
    first->generateConfig()->presence_penalty = 0.4;
    first->generateConfig()->frequency_penalty = 0.3;
    first->generateConfig()->no_repeat_ngram_size = 2;
    for (const auto& stream : {first, second}) {
        stream->generateConfig()->do_sample = true;
        stream->generateConfig()->top_k = 1;
    }
    const auto plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({first, second}));
    const auto candidates = intTensor({1, 2, 9, 10, 8, 8, 11, 12});
    auto init_result = makeInitializedResult(plan);
    executor.batch_stream_processor_->initSamplingStates(plan, executor.sampling_states_, init_result);
    ASSERT_EQ(init_result.request_errors.size(), 2);
    EXPECT_FALSE(init_result.request_errors[0].hasError());
    EXPECT_FALSE(init_result.request_errors[1].hasError());
    auto logits = torch::zeros({8, 64}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    logits.select(1, 2).fill_(10);
    logits.select(1, 9).fill_(9);
    logits.select(1, 10).fill_(8);
    logits[0][2] = 100;
    auto inputs = executor.batch_stream_processor_->gatherSamplerInputs(
        plan, PPOutputConfig{}, logits, executor.sampling_states_, true, 3, candidates);
    EXPECT_EQ(tensorToVector<int32_t>(inputs.sequence_lengths),
              (std::vector<int32_t>{3, 4, 5, 6, 2, 3, 4, 5}));
    const std::vector<std::vector<int32_t>> expected{
        {1, 2, 1}, {1, 2, 1, 2}, {1, 2, 1, 2, 9}, {1, 2, 1, 2, 9, 10},
        {7, 8}, {7, 8, 8}, {7, 8, 8, 11}, {7, 8, 8, 11, 12}};
    for (int64_t row = 0; row < 8; ++row) {
        EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[row].narrow(0, 0, expected[row].size())), expected[row]);
        if (row < 4) {
            EXPECT_EQ(inputs.no_repeat_ngram_size[row].item<int32_t>(), 2);
            EXPECT_FLOAT_EQ(inputs.repetition_penalty[row].item<float>(), 1.7);
            EXPECT_FLOAT_EQ(inputs.presence_penalty[row].item<float>(), 0.4);
            EXPECT_FLOAT_EQ(inputs.frequency_penalty[row].item<float>(), 0.3);
        }
    }
    /** Row 0 bans the repeated bigram; later rows penalize only candidates already in their prefix. */
    Sampler sampler(SamplerInitParams{8, false});
    const auto sampled = sampler.forward(inputs);
    EXPECT_EQ(tensorToVector<int32_t>(sampled.token_ids.select(1, sampled.token_ids.size(1) - 1)),
              (std::vector<int32_t>{9, 9, 10, 2, 2, 2, 2, 2}));
}

TEST_F(PPBatchStreamProcessorTest, FirstTailProcessorReplaysPdAnchorOnlyOnce) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {3}, 0);
    stream->update({intTensor({0}).reshape({1, 1}), 1, {}, {}, {}, {}, {}, {}, {}, {}});
    auto plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({stream}));
    plan.logits_processor_configs[0].grammar_type = "regex";
    plan.logits_processor_configs[0].grammar_value = "abc";
    auto backend = XGrammarBackend::create(
        xgrammar_impl::serializeTokenizerInfo({"a", "b", "c", "<eos>"},
                                               R"({"vocab_size":4,"stop_token_ids":[3],"vocab_type":"RAW","add_prefix_space":false})"),
        GrammarConfig{});
    ASSERT_NE(backend, nullptr);
    auto previous_backend = LogitsProcessorFactory::grammarBackend();
    LogitsProcessorFactory::grammarBackend() = backend;
    auto init_result = makeInitializedResult(plan);
    executor.batch_stream_processor_->initSamplingStates(plan, executor.sampling_states_, init_result);
    executor.batch_stream_processor_->gatherSamplerInputs(
        plan, PPOutputConfig{}, torch::zeros({1, 4}), executor.sampling_states_);
    const auto first_processors = executor.sampling_states_.at(101).logits_processors;
    auto repeated_result = makeInitializedResult(plan);
    executor.batch_stream_processor_->initSamplingStates(plan, executor.sampling_states_, repeated_result);
    executor.batch_stream_processor_->gatherSamplerInputs(
        plan, PPOutputConfig{}, torch::zeros({1, 4}), executor.sampling_states_);
    LogitsProcessorFactory::grammarBackend() = previous_backend;
    ASSERT_EQ(init_result.request_errors.size(), 1);
    EXPECT_FALSE(init_result.request_errors[0].hasError());
    ASSERT_EQ(repeated_result.request_errors.size(), 1);
    EXPECT_FALSE(repeated_result.request_errors[0].hasError());
    const auto& processors = executor.sampling_states_.at(101).logits_processors;
    ASSERT_EQ(processors.size(), 1);
    EXPECT_EQ(processors, first_processors);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(1));
}

TEST_F(PPBatchStreamProcessorTest, RegistrationUsesSequenceOffsetAfterExistingMultiSequenceRequest) {
    auto params = makeMtpParams(3);
    params.sp_config.type = SP_TYPE_NONE;
    params.model_config_.special_tokens.eos_token_id = 3;
    PPExecutor executor(params, nullptr, true);
    auto existing = makeStream(ResourceContext{}, params.model_config_, 101, {3, 3}, 2);
    existing->setIsContextStream(false);
    existing->generateConfig()->random_seed = 123;
    const auto existing_plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({existing}));
    auto existing_result = makeInitializedResult(existing_plan);
    executor.batch_stream_processor_->initSamplingStates(existing_plan, executor.sampling_states_, existing_result);
    ASSERT_EQ(existing_result.request_errors.size(), 1);
    ASSERT_FALSE(existing_result.request_errors[0].hasError());
    auto& existing_state = executor.sampling_states_.at(101);
    const auto generator = existing_state.generator;
    auto processor = std::make_shared<RecordingLogitsProcessor>();
    existing_state.logits_processors = {processor};
    existing_state.cum_log_probs.fill_(-4.0);

    auto added = makeStream(ResourceContext{}, params.model_config_, 202, {3}, 0);
    added->update({intTensor({0}).reshape({1, 1}), 1, {}, {}, {}, {}, {}, {}, {}, {}});
    added->setIsContextStream(false);
    auto plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({existing, added}));
    plan.random_seeds = {999, 456};
    plan.logits_processor_configs[1].grammar_type = "regex";
    plan.logits_processor_configs[1].grammar_value = "abc";
    auto backend = XGrammarBackend::create(
        xgrammar_impl::serializeTokenizerInfo({"a", "b", "c", "<eos>"},
                                               R"({"vocab_size":4,"stop_token_ids":[3],"vocab_type":"RAW","add_prefix_space":false})"),
        GrammarConfig{});
    ASSERT_NE(backend, nullptr);
    auto previous_backend = LogitsProcessorFactory::grammarBackend();
    LogitsProcessorFactory::grammarBackend() = backend;
    auto init_result = makeInitializedResult(plan);
    executor.batch_stream_processor_->initSamplingStates(plan, executor.sampling_states_, init_result);
    LogitsProcessorFactory::grammarBackend() = previous_backend;

    ASSERT_EQ(init_result.request_errors.size(), 2);
    EXPECT_FALSE(init_result.request_errors[0].hasError());
    EXPECT_FALSE(init_result.request_errors[1].hasError());
    EXPECT_EQ(existing_state.generator, generator);
    EXPECT_EQ(existing_state.generator.current_seed(), 123);
    ASSERT_EQ(existing_state.logits_processors.size(), 1);
    EXPECT_EQ(existing_state.logits_processors[0], processor);
    EXPECT_TRUE(processor->committed_tokens.empty());
    const auto& added_state = executor.sampling_states_.at(202);
    ASSERT_EQ(added_state.logits_processors.size(), 1);
    EXPECT_EQ(added_state.logits_processors[0]->committedOutputLen(), std::optional<int64_t>(1));
    ASSERT_TRUE(added_state.generator.defined());
    EXPECT_EQ(added_state.generator.current_seed(), 456);

    PPOutputConfig output_config;
    output_config.return_cum_log_probs = true;
    const auto inputs = executor.batch_stream_processor_->gatherSamplerInputs(
        plan, output_config, torch::zeros({3, 4}), executor.sampling_states_);
    EXPECT_EQ(inputs.generator[0], generator);
    EXPECT_EQ(inputs.generator[1], generator);
    EXPECT_EQ(inputs.generator[2], added_state.generator);
    EXPECT_EQ(tensorToVector<float>(inputs.cum_log_probs), (std::vector<float>{-4.0, -4.0, 0.0}));
}

TEST_F(PPBatchStreamProcessorTest, MaxTokensCapsTailProcessorAndHeadCommitAcrossRounds) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    ResourceContext context;
    context.cache_manager = std::make_shared<KVCacheManager>(
        test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(context.cache_manager->init());
    auto stream = makeStream(context, params.model_config_, 101, {1, 2}, 0);
    stream->fakeInitKVBlock(4);
    std::list<GenerateStreamPtr> streams_to_prepare{stream};
    executor.prepareStreams(streams_to_prepare);
    stream->setIsContextStream(false);
    stream->generateConfig()->max_new_tokens = 3;
    auto processor = std::make_shared<RecordingLogitsProcessor>();
    executor.sampling_states_[101].logits_processors = {processor};
    for (int32_t round = 0; round < 2; ++round) {
        SCOPED_TRACE(round);
        PPExecutionPlan plan;
        plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({stream}));
        plan = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
        EXPECT_EQ(tensorToVector<int32_t>(plan.sampling_plan.max_tokens), (std::vector<int32_t>{5}));
        EXPECT_EQ(tensorToVector<int32_t>(plan.sampling_plan.sequence_lengths), (std::vector<int32_t>{2 + round}));

        PPExecutionResult result;
        result.request_ids = plan.sampling_plan.request_ids;
        result.new_token_ids = intTensor({10 + round, 11 + round, 12 + round, 13 + round}).reshape({1, 4});
        result.accept_len = intTensor({round == 0 ? 1 : 4});
        result.propose_token_ids = intTensor({20, 21, 22}).reshape({1, 3});
        result.request_errors.resize(1);
        result.prompt_logits.resize(1);
        executor.clipMtpAcceptedLengths(plan.sampling_plan, result);
        EXPECT_EQ(processor->committed_tokens.size(), round);
        executor.advanceSamplingStates(plan.sampling_plan, result);
        EXPECT_EQ(result.accept_len.item<int32_t>(), round == 0 ? 1 : 2);
        ASSERT_EQ(processor->committed_tokens.size(), round + 1);
        EXPECT_EQ(tensorToVector<int32_t>(processor->committed_tokens.back()),
                  round == 0 ? (std::vector<int32_t>{10}) : (std::vector<int32_t>{11, 12}));
        stream->setPPInflight();
        ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_EQ(stream->hasEvent(StreamEvents::GenerateDone), round == 1);
        EXPECT_FALSE(stream->isPPInflight());
    }
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 10, 11, 12}));
}

TEST_F(PPBatchStreamProcessorTest, TailReplayFailureClearsProcessorsAndPreservesSamplingState) {
    auto params = makeMtpParams(3);
    params.model_config_.special_tokens.eos_token_id = 3;
    PPExecutor executor(params, nullptr, true);
    auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {3}, 0);
    stream->update({intTensor({1}).reshape({1, 1}), 1, {}, {}, {}, {}, {}, {}, {}, {}});
    auto plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({stream}));
    // The head accepted this history; only the tail's reconstructed grammar rejects it.
    plan.logits_processor_configs[0].grammar_type = "regex";
    plan.logits_processor_configs[0].grammar_value = "abc";
    plan.random_seeds[0] = 123;
    auto backend = XGrammarBackend::create(
        xgrammar_impl::serializeTokenizerInfo({"a", "b", "c", "<eos>"},
                                               R"({"vocab_size":4,"stop_token_ids":[3],"vocab_type":"RAW","add_prefix_space":false})"),
        GrammarConfig{});
    ASSERT_NE(backend, nullptr);
    auto previous_backend = LogitsProcessorFactory::grammarBackend();
    LogitsProcessorFactory::grammarBackend() = backend;
    auto init_result = makeInitializedResult(plan);
    executor.batch_stream_processor_->initSamplingStates(plan, executor.sampling_states_, init_result);
    auto inputs = executor.batch_stream_processor_->gatherSamplerInputs(
        plan, PPOutputConfig{}, torch::zeros({1, 4}), executor.sampling_states_);
    LogitsProcessorFactory::grammarBackend() = previous_backend;
    auto& state = executor.sampling_states_.at(101);
    ASSERT_EQ(init_result.request_errors.size(), 1);
    ASSERT_TRUE(init_result.request_errors[0].hasError());
    EXPECT_EQ(init_result.request_errors[0].code(), ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN);
    EXPECT_NE(init_result.request_errors[0].ToString().find("parser rejected token"), std::string::npos);
    EXPECT_TRUE(state.logits_processors.empty());
    ASSERT_TRUE(state.generator.defined());
    EXPECT_EQ(state.generator.current_seed(), 123);
    EXPECT_EQ(inputs.generator[0], state.generator);
    ASSERT_NE(inputs.logits_processor_states_ptr, nullptr);
    EXPECT_TRUE(inputs.logits_processor_states_ptr->invocations_.empty());
    EXPECT_FALSE(inputs.logits_processor_states_ptr->batchProcess(inputs)[0].has_value());


}

TEST_F(PPBatchStreamProcessorTest, TailInitializationFailurePreservesHealthyRequestsThroughDraftAndCommit) {
    for (const std::string mode : {"normal", "mtp_prefill", "mtp_decode"}) {
        for (int64_t failed_idx = 0; failed_idx < 3; ++failed_idx) {
            SCOPED_TRACE(mode);
            SCOPED_TRACE(failed_idx);
            auto params = makeMtpParams(3);
            params.model_config_.special_tokens.eos_token_id = 63;
            if (mode == "normal") {
                params.sp_config.type = SP_TYPE_NONE;
            }
            PPExecutor executor(params, nullptr, true);
            PPExecutor reference(params, nullptr, true);
            for (auto* current : {&executor, &reference}) {
                current->sampler_ = std::make_unique<Sampler>(SamplerInitParams{12, false});
                current->speculative_sampler_ =
                    std::make_unique<speculative::SpeculativeSampler>(torch::Tensor(), 3);
                current->spec_logits_verify_runner_ = std::make_unique<SpecLogitsVerifyRunner>();
            }
            ResourceContext context;
            context.cache_manager = std::make_shared<KVCacheManager>(
                test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
            ASSERT_TRUE(context.cache_manager->init());
            std::vector<GenerateStreamPtr> streams{
                makeStream(context, params.model_config_, 101, {1, 2}, 0),
                makeStream(context, params.model_config_, 202, {3, 4}, 0),
                makeStream(context, params.model_config_, 303, {5, 6}, 0)};
            for (const auto& stream : streams) {
                stream->fakeInitKVBlock(4);
            }
            std::list<GenerateStreamPtr> streams_to_prepare(streams.begin(), streams.end());
            executor.prepareStreams(streams_to_prepare);
            if (mode == "mtp_decode") {
                for (const auto& stream : streams) {
                    stream->setIsContextStream(false);
                }
            }
            const StreamGroups groups(streams_to_prepare);
            PPExecutionPlan plan;
            plan.is_decode = mode == "mtp_decode";
            plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(groups);
            plan.sampling_plan.random_seeds = {123, 456, 789};
            plan.sampling_plan.top_k.fill_(4);
            plan.sampling_plan.do_sample.fill_(true);
            if (mode != "normal") {
                plan.sampling_plan.spec_do_sample.fill_(true);
            }
            plan.model_input.combo_tokens = plan.is_decode ? intTensor({2, 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3}) :
                                                             intTensor({1, 2, 3, 4, 5, 6});
            plan.model_input.input_lengths = torch::full({3}, plan.is_decode ? 4 : 2, torch::kInt32);
            plan.model_input.prefix_lengths = torch::full({3}, plan.is_decode ? 1 : 0, torch::kInt32);
            plan.model_input.sequence_lengths = intTensor({});
            plan.model_input.lm_output_indexes = plan.is_decode ? torch::arange(12, torch::kInt32) : intTensor({1, 3, 5});
            plan.model_input.is_target_verify = plan.is_decode;
            const auto reference_plan = plan;
            plan.sampling_plan.logits_processor_configs[failed_idx].grammar_type = "invalid";
            const auto sampling_rows = plan.is_decode ? 12 : 3;
            const auto logits = torch::arange(64, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA))
                                    .mul_(-0.05).repeat({sampling_rows, 1});
            auto reference_init_result = makeInitializedResult(reference_plan.sampling_plan);
            reference.batch_stream_processor_->initSamplingStates(
                reference_plan.sampling_plan, reference.sampling_states_, reference_init_result);
            ASSERT_EQ(reference_init_result.request_errors.size(), 3);
            std::vector<at::Generator> generators(3);
            std::vector<torch::Tensor> initial_rng_states;
            std::vector<std::shared_ptr<RecordingLogitsProcessor>> processors(3);
            for (int64_t idx = 0; idx < 3; ++idx) {
                EXPECT_FALSE(reference_init_result.request_errors[idx].hasError());
                initial_rng_states.push_back(
                    reference.sampling_states_.at(streams[idx]->streamId()).generator.get_state());
                if (idx != failed_idx) {
                    auto existing_plan =
                        executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({streams[idx]}));
                    existing_plan.random_seeds[0] = plan.sampling_plan.random_seeds[idx];
                    auto init_result = makeInitializedResult(existing_plan);
                    executor.batch_stream_processor_->initSamplingStates(
                        existing_plan, executor.sampling_states_, init_result);
                    ASSERT_EQ(init_result.request_errors.size(), 1);
                    ASSERT_FALSE(init_result.request_errors[0].hasError());
                    auto& state = executor.sampling_states_.at(streams[idx]->streamId());
                    generators[idx] = state.generator;
                    processors[idx] = std::make_shared<RecordingLogitsProcessor>();
                    state.logits_processors = {processors[idx]};
                }
            }
            // The failed request must first be initialized by sampleTokens, alongside existing healthy states.
            EXPECT_EQ(executor.sampling_states_.count(streams[failed_idx]->streamId()), 0);
            GptModelOutputs output;
            output.logits = logits.clone();
            PPExecutionResult result;
            executor.sampleTokens(plan, output, result);
            output.logits = logits.clone();
            PPExecutionResult expected;
            reference.sampleTokens(reference_plan, output, expected);
            ASSERT_EQ(result.request_errors.size(), 3);
            ASSERT_TRUE(result.request_errors[failed_idx].hasError());
            EXPECT_EQ(result.request_errors[failed_idx].code(), ErrorCode::INVALID_PARAMS);
            const auto init_error = result.request_errors[failed_idx];
            std::vector<torch::Tensor> sampled_rng_states;
            for (int64_t idx = 0; idx < 3; ++idx) {
                auto& state = executor.sampling_states_.at(streams[idx]->streamId());
                ASSERT_TRUE(state.generator.defined());
                EXPECT_EQ(state.generator.current_seed(), plan.sampling_plan.random_seeds[idx].value());
                EXPECT_EQ(result.request_errors[idx].hasError(), idx == failed_idx);
                sampled_rng_states.push_back(state.generator.get_state());
                if (idx != failed_idx) {
                    EXPECT_FALSE(torch::equal(sampled_rng_states.back(), initial_rng_states[idx]));
                    EXPECT_TRUE(torch::equal(sampled_rng_states.back(),
                                             reference.sampling_states_.at(streams[idx]->streamId()).generator.get_state()));
                    EXPECT_EQ(processors[idx]->verify_calls, plan.is_decode ? 1 : 0);
                    EXPECT_EQ(processors[idx]->process_intervals.size(), plan.is_decode ? 0 : 1);
                    if (!plan.is_decode) {
                        EXPECT_EQ(processors[idx]->process_intervals[0], (std::pair<size_t, size_t>{idx, idx + 1}));
                    }
                } else {
                    generators[idx] = state.generator;
                    EXPECT_TRUE(state.logits_processors.empty());
                }
            }
            if (mode != "normal") {
                auto target_model = std::make_unique<RecordingDraftModel>();
                target_model->hidden_buffer = torch::zeros({plan.model_input.combo_tokens.numel(), 4}, logits.options());
                executor.setModel(std::move(target_model));
                executor.draft_model_ = std::make_unique<RecordingDraftModel>();
                executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>();
                executor.draftSampleAndPropose(plan, output, result);
                EXPECT_EQ(result.accept_len[failed_idx].item<int32_t>(), 1);
                EXPECT_TRUE(torch::equal(result.new_token_ids[failed_idx],
                                         torch::zeros_like(result.new_token_ids[failed_idx])));
            }
            for (int64_t idx = 0; idx < 3; ++idx) {
                if (idx != failed_idx) {
                    const auto accepted = plan.is_decode ? expected.accept_len[idx].item<int32_t>() : 1;
                    EXPECT_TRUE(torch::equal(result.new_token_ids[idx].narrow(0, 0, accepted),
                                             expected.new_token_ids[idx].narrow(0, 0, accepted)));
                    if (mode != "normal") {
                        EXPECT_EQ(result.accept_len[idx].item<int32_t>(), accepted);
                    }
                }
            }
            auto inputs = executor.batch_stream_processor_->gatherSamplerInputs(
                plan.sampling_plan, PPOutputConfig{}, logits, executor.sampling_states_,
                plan.is_decode, plan.is_decode ? 3 : 0, plan.model_input.combo_tokens);
            for (int64_t idx = 0; idx < 3; ++idx) {
                const auto& state = executor.sampling_states_.at(streams[idx]->streamId());
                EXPECT_EQ(state.generator, generators[idx]);
                EXPECT_TRUE(torch::equal(state.generator.get_state(), sampled_rng_states[idx]));
                const auto width = plan.is_decode ? 4 : 1;
                for (int64_t offset = 0; offset < width; ++offset) {
                    EXPECT_EQ(inputs.generator[idx * width + offset], state.generator);
                    EXPECT_EQ(inputs.token_ids[idx * width + offset][0].item<int32_t>(), 1 + idx * 2);
                    EXPECT_EQ(inputs.sequence_lengths[idx * width + offset].item<int32_t>(), 2 + offset);
                }
            }

            result.cum_log_probs = torch::full({3}, -5.0, torch::kFloat32);
            executor.advanceSamplingStates(plan.sampling_plan, result);
            for (int64_t idx = 0; idx < 3; ++idx) {
                EXPECT_EQ(executor.sampling_states_.at(streams[idx]->streamId()).cum_log_probs.item<float>(),
                          idx == failed_idx ? 0.0f : -5.0f);
                EXPECT_EQ(result.request_errors[idx].hasError(), idx == failed_idx);
            }
            auto received = pp_serialization::deserializeExecutionResult(
                pp_serialization::serializeExecutionResult(result));
            for (const auto& stream : streams) {
                stream->setPPInflight();
            }
            ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(groups, received).ok());
            for (int64_t idx = 0; idx < 3; ++idx) {
                EXPECT_FALSE(streams[idx]->isPPInflight());
                EXPECT_EQ(streams[idx]->hasError(), idx == failed_idx);
                const auto committed = idx == failed_idx ? 0 :
                                       mode == "normal" ? 1 : result.accept_len[idx].item<int32_t>();
                EXPECT_EQ(streams[idx]->seqLength(), 2 + committed);
            }
            EXPECT_EQ(streams[failed_idx]->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(streams[failed_idx]->statusInfo().ToString(), init_error.ToString());
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, RequestErrorsPreserveStagePriorityAcrossSequencesAndStateUpdate) {
    auto params = makeMtpParams(3);
    params.sp_config.type = SP_TYPE_NONE;
    PPExecutor executor(params, nullptr, true);
    const std::vector<GenerateStreamPtr> streams{
        makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 2),
        makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 2),
        makeStream(ResourceContext{}, params.model_config_, 303, {5, 6}, 1),
        makeStream(ResourceContext{}, params.model_config_, 404, {7, 8}, 1),
        makeStream(ResourceContext{}, params.model_config_, 505, {9, 10}, 1)};
    const StreamGroups groups(std::list<GenerateStreamPtr>(streams.begin(), streams.end()));
    PPExecutionPlan plan;
    plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(groups);
    plan.output_config.return_cum_log_probs = true;
    const ErrorInfo init_error(ErrorCode::INVALID_PARAMS, "initialization failed");
    const ErrorInfo process_error(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN, "second sequence processor failed");
    const ErrorInfo update_error(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL, "state update failed");
    auto result = makeInitializedResult(plan.sampling_plan);
    result.request_errors[0] = init_error;
    std::vector<std::shared_ptr<RecordingLogitsProcessor>> processors;
    for (size_t i = 0; i < streams.size(); ++i) {
        auto processor = std::make_shared<RecordingLogitsProcessor>();
        if (i != 3) {
            processor->update_error = update_error;
        }
        processors.push_back(processor);
        auto& state = executor.sampling_states_[streams[i]->streamId()];
        state.logits_processors = {processor};
        state.cum_log_probs = torch::zeros({streams[i]->currentBatchSize()}, torch::kFloat32);
    }
    SamplerOutput sampled;
    sampled.token_ids = intTensor({11, 12, 13, 14, 15, 16, 17}).reshape({7, 1});
    sampled.cum_log_probs = torch::full({7}, -5.0, torch::kFloat32);
    // A processor error on the second sequence takes precedence over a sampler
    // failure on the first sequence of the same request.
    sampled.success = torch::tensor({false, true, false, true, false, true, true}, torch::kBool);
    sampled.processor_errors.resize(7);
    sampled.processor_errors[1] = process_error;
    sampled.processor_errors[3] = process_error;
    executor.batch_stream_processor_->fillExecutionResult(plan, GptModelOutputs{}, sampled, result);
    ASSERT_EQ(result.request_errors.size(), streams.size());
    EXPECT_EQ(result.request_errors[0].code(), init_error.code());
    EXPECT_EQ(result.request_errors[1].code(), process_error.code());
    EXPECT_EQ(result.request_errors[2].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_TRUE(result.request_errors[3].ok());
    EXPECT_TRUE(result.request_errors[4].ok());

    executor.advanceSamplingStates(plan.sampling_plan, result);
    const std::vector<ErrorInfo> expected{
        init_error,
        process_error,
        ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed"),
        ErrorInfo::OkStatus(),
        update_error};
    auto received = pp_serialization::deserializeExecutionResult(
        pp_serialization::serializeExecutionResult(result));
    ASSERT_EQ(received.request_errors.size(), expected.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        EXPECT_EQ(received.request_errors[i].code(), expected[i].code());
        EXPECT_EQ(received.request_errors[i].ToString(), expected[i].ToString());
        EXPECT_EQ(processors[i]->committed_tokens.size(), i >= 3 ? 1 : 0);
        const auto& probs = executor.sampling_states_.at(streams[i]->streamId()).cum_log_probs;
        EXPECT_TRUE(torch::equal(probs, torch::full_like(probs, i == 3 ? -5.0 : 0.0)));
    }
}

TEST_F(PPBatchStreamProcessorTest, OrdinaryPpErrorsUseRequestCoordinatesWithMultipleSequences) {
    for (const std::string stage : {"initialize", "process", "update"}) {
        for (int64_t failed_idx : {0, 1}) {
            SCOPED_TRACE(stage);
            SCOPED_TRACE(failed_idx);
            auto params = makeMtpParams(3);
            params.sp_config.type = SP_TYPE_NONE;
            PPExecutor executor(params, nullptr, true);
            executor.sampler_ = std::make_unique<Sampler>(SamplerInitParams{3, false});
            const std::vector<GenerateStreamPtr> streams{
                makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 2),
                makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 1)};
            const StreamGroups groups(std::list<GenerateStreamPtr>{streams[0], streams[1]});
            PPExecutionPlan plan;
            plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(groups);
            plan.output_config = executor.batch_stream_processor_->gatherOutputConfig(groups);
            const ErrorInfo processor_error(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN, "processor failed");
            const auto healthy_idx = 1 - failed_idx;
            auto healthy_processor = std::make_shared<RecordingLogitsProcessor>();
            auto failed_processor = std::make_shared<RecordingLogitsProcessor>();
            for (int64_t idx = 0; idx < 2; ++idx) {
                if (stage == "initialize" && idx == failed_idx) {
                    plan.sampling_plan.logits_processor_configs[idx].grammar_type = "invalid";
                    continue;
                }
                auto& state = executor.sampling_states_[streams[idx]->streamId()];
                state.logits_processors = {idx == failed_idx ? failed_processor : healthy_processor};
                state.cum_log_probs = torch::zeros({streams[idx]->currentBatchSize()}, torch::kFloat32);
            }
            if (stage == "process") {
                failed_processor->process_error = processor_error;
            } else if (stage == "update") {
                failed_processor->update_error = processor_error;
            }
            GptModelOutputs output;
            output.logits = torch::zeros({3, 64}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
            output.logits.select(1, 10).fill_(100);
            PPExecutionResult result;
            executor.sampleTokens(plan, output, result);
            executor.advanceSamplingStates(plan.sampling_plan, result);
            EXPECT_FALSE(result.accept_len.defined());
            ASSERT_EQ(result.request_errors.size(), 2);
            EXPECT_EQ(result.request_errors[failed_idx].code(),
                      stage == "initialize" ? ErrorCode::INVALID_PARAMS : processor_error.code());
            EXPECT_TRUE(result.request_errors[healthy_idx].ok());
            EXPECT_EQ(healthy_processor->committed_tokens.size(), 1);
            EXPECT_EQ(failed_processor->committed_tokens.size(), stage == "update" ? 1 : 0);

            auto received = pp_serialization::deserializeExecutionResult(
                pp_serialization::serializeExecutionResult(result));
            for (const auto& stream : streams) {
                stream->setPPInflight();
            }
            ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(groups, received).ok());
            EXPECT_EQ(streams[failed_idx]->statusInfo().code(),
                      stage == "initialize" ? ErrorCode::INVALID_PARAMS : processor_error.code());
            EXPECT_FALSE(streams[healthy_idx]->hasError());
            for (int64_t idx = 0; idx < 2; ++idx) {
                EXPECT_FALSE(streams[idx]->isPPInflight());
                auto expected = idx == 0 ? std::vector<int>{1, 2} : std::vector<int>{3, 4};
                if (idx == healthy_idx) {
                    expected.push_back(10);
                }
                for (int row = 0; row < streams[idx]->currentBatchSize(); ++row) {
                    EXPECT_EQ(streams[idx]->completeTokenIdsVec(row), expected);
                }
            }
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, MtpAdmissionRejectsBeforePlanAndSchedulerReclaimsRequest) {
    auto params = makeMtpParams(3);
    params.pd_sep_config.role_type = RoleType::PDFUSION;
    params.runtime_config.max_generate_batch_size = 8;
    params.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 32;
    auto cache_manager = std::make_shared<KVCacheManager>(
        test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(cache_manager->init());
    const auto free_blocks = cache_manager->freeBlocksNum();
    PPExecutor executor(params, cache_manager, false);
    PPScheduler scheduler(params.runtime_config, params.model_config_, params.pd_sep_config,
                          params.parallelism_config, params.model_specific_config, cache_manager);
    ResourceContext context;
    context.cache_manager = cache_manager;
    auto stream = makeStream(context, params.model_config_, 101, {1, 2}, 0);
    stream->generate_status_->status = StreamState::WAITING;
    auto processor = std::make_shared<RecordingLogitsProcessor>();
    processor->mtp_supported = false;
    stream->sampling_state_.logits_processors = {processor};
    const auto expected_error = LogitsProcessorFactory::validateMtpCompatibility(stream->getAllLogitsProcessorPtr());
    ASSERT_TRUE(expected_error.has_value());
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.ok());
    ASSERT_EQ(scheduled->streams.size(), 1);
    ASSERT_TRUE(stream->isPPInflight());
    EXPECT_LT(cache_manager->freeBlocksNum(), free_blocks);

    auto transport = std::make_unique<InMemoryPPTransport>();
    auto* recorded_transport = transport.get();
    executor.transport_ = std::move(transport);
    // All requests are rejected, so process must send an empty plan without invoking a model.
    ASSERT_TRUE(executor.process(scheduled.value()).ok());
    EXPECT_FALSE(stream->isPPInflight());
    EXPECT_EQ(stream->statusInfo().code(), expected_error->code());
    EXPECT_EQ(stream->statusInfo().ToString(), expected_error->ToString());
    ASSERT_EQ(recorded_transport->sent_tensors.size(), 2);
    const auto plan = pp_serialization::deserializePlan(recorded_transport->sent_tensors[1]);
    EXPECT_TRUE(plan.model_input.skip_run);

    auto retired = scheduler.schedule();
    ASSERT_TRUE(retired.ok());
    EXPECT_TRUE(retired->streams.empty());
    EXPECT_EQ(retired->finished_request_ids, (std::vector<int64_t>{101}));
    EXPECT_TRUE(stream->isFinished());
    EXPECT_TRUE(scheduler.empty());
    EXPECT_EQ(cache_manager->freeBlocksNum(), free_blocks);
}

TEST_F(PPBatchStreamProcessorTest, RequestErrorsDoNotRequireSuccessfulTokenPayloads) {
    for (bool mtp_enabled : {false, true}) {
        SCOPED_TRACE(mtp_enabled);
        auto params = makeMtpParams(3);
        if (!mtp_enabled) {
            params.sp_config.type = SP_TYPE_NONE;
        }
        PPExecutor executor(params, nullptr, true);
        const ErrorInfo error(ErrorCode::INVALID_PARAMS, "initialization failed");
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
        stream->generateConfig()->return_hidden_states = true;
        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        if (mtp_enabled) {
            result.accept_len = intTensor({0});
        }
        result.prompt_logits.resize(1);
        result.request_errors = {error};
        result = pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
        ASSERT_EQ(result.request_errors.size(), 1);
        EXPECT_EQ(result.request_errors[0].code(), error.code());
        EXPECT_EQ(result.request_errors[0].ToString(), error.ToString());
        EXPECT_FALSE(result.new_token_ids.defined());
        EXPECT_FALSE(result.propose_token_ids.defined());
        stream->setPPInflight();
        ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_EQ(stream->statusInfo().code(), error.code());
        EXPECT_EQ(stream->statusInfo().ToString(), error.ToString());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
    }
}

TEST_F(PPBatchStreamProcessorTest, RequestErrorsAndCancellationReachSchedulerAndTailCleanup) {
    for (bool mtp_enabled : {false, true}) {
        SCOPED_TRACE(mtp_enabled);
        for (const std::string stage : {"result", "cancel", "timeout"}) {
            SCOPED_TRACE(stage);
            auto params = makeMtpParams(3);
            if (!mtp_enabled) {
                params.sp_config.type = SP_TYPE_NONE;
            }
            params.pd_sep_config.role_type = RoleType::PDFUSION;
            params.runtime_config.max_generate_batch_size = 8;
            params.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 32;
            auto cache_manager = std::make_shared<KVCacheManager>(
                test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
            ASSERT_TRUE(cache_manager->init());
            const auto free_blocks = cache_manager->freeBlocksNum();
            PPExecutor head(params, cache_manager, true);
            auto tail_params = params;
            tail_params.parallelism_config.pp_rank = 1;
            tail_params.parallelism_config.world_rank = 1;
            tail_params.parallelism_config.world_size = 2;
            PPExecutor tail(tail_params, cache_manager, false);
            PPScheduler scheduler(params.runtime_config, params.model_config_, params.pd_sep_config,
                                  params.parallelism_config, params.model_specific_config, cache_manager);
            ResourceContext context;
            context.cache_manager = cache_manager;
            auto failed = makeStream(context, params.model_config_, 101, {1, 2}, 0);
            auto healthy = makeStream(context, params.model_config_, 202, {3, 4}, 0);
            for (const auto& stream : {failed, healthy}) {
                stream->generate_status_->status = StreamState::WAITING;
                ASSERT_TRUE(scheduler.enqueue(stream).ok());
            }
            auto scheduled = scheduler.schedule();
            ASSERT_TRUE(scheduled.ok());
            ASSERT_EQ(scheduled->streams.size(), 2);
            head.prepareStreams(scheduled->streams);
            ASSERT_EQ(scheduled->streams.size(), 2);
            EXPECT_TRUE(failed->isPPInflight());
            EXPECT_TRUE(healthy->isPPInflight());

            PPExecutionResult result;
            result.request_ids = torch::tensor({101, 202}, torch::kInt64);
            result.new_token_ids = intTensor({10, 11}).reshape({2, 1});
            if (mtp_enabled) {
                result.accept_len = intTensor({1, 1});
                result.propose_token_ids = intTensor({20, 21, 22, 30, 31, 32}).reshape({2, 3});
            }
            result.prompt_logits.resize(2);
            result.request_errors.resize(2);
            ErrorInfo expected_error(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN, "request processor failed");
            if (stage == "result") {
                result.request_errors[0] = expected_error;
            } else {
                expected_error = ErrorInfo(stage == "cancel" ? ErrorCode::CANCELLED : ErrorCode::GENERATE_TIMEOUT,
                                          "request ended while in flight");
                failed->reportError(expected_error.code(), expected_error.ToString());
                result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "late processor error");
            }
            // A pending cancellation/timeout must not release KV before the result returns.
            const auto allocated_free_blocks = cache_manager->freeBlocksNum();
            auto in_flight = scheduler.schedule();
            ASSERT_TRUE(in_flight.ok());
            EXPECT_TRUE(in_flight->streams.empty());
            EXPECT_TRUE(in_flight->finished_request_ids.empty());
            EXPECT_EQ(cache_manager->freeBlocksNum(), allocated_free_blocks);

            result = pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
            ASSERT_TRUE(head.batch_stream_processor_->dispatchExecutionResult(StreamGroups(scheduled->streams), result).ok());
            EXPECT_FALSE(failed->isPPInflight());
            EXPECT_FALSE(healthy->isPPInflight());
            EXPECT_EQ(failed->statusInfo().code(), expected_error.code());
            EXPECT_EQ(failed->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
            EXPECT_FALSE(healthy->hasError());
            EXPECT_EQ(healthy->completeTokenIdsVec(0), (std::vector<int>{3, 4, 11}));
            auto output = failed->nextOutput();
            ASSERT_FALSE(output.ok());
            EXPECT_EQ(output.status().code(), expected_error.code());
            EXPECT_EQ(output.status().ToString(), expected_error.ToString());

            auto retired = scheduler.schedule();
            ASSERT_TRUE(retired.ok());
            EXPECT_EQ(retired->finished_request_ids, (std::vector<int64_t>{101}));
            ASSERT_EQ(retired->streams.size(), 1);
            EXPECT_EQ(retired->streams.front(), healthy);
            EXPECT_TRUE(failed->isFinished());
            EXPECT_EQ(cache_manager->freeBlocksNum(), free_blocks - healthy->curBlocksNum());

            tail.sampling_states_[101] = SamplingState{};
            tail.sampling_states_[202] = SamplingState{};
            auto cleanup = head.buildPlan(StreamGroups{}, retired->finished_request_ids);
            ASSERT_TRUE(cleanup.ok());
            const auto payload = pp_serialization::serializePlan(cleanup.value(), false);
            auto transport = std::make_unique<InMemoryPPTransport>();
            transport->received_tensors = {torch::tensor({payload.numel()}, torch::kInt64), payload};
            tail.transport_ = std::move(transport);
            ASSERT_TRUE(tail.process(ScheduleOutput{}).ok());
            EXPECT_EQ(tail.sampling_states_.count(101), 0);
            EXPECT_EQ(tail.sampling_states_.count(202), 1);

            healthy->reportError(ErrorCode::CANCELLED, "test cleanup");
            healthy->clearPPInflight();
            ASSERT_TRUE(scheduler.schedule().ok());
            EXPECT_TRUE(scheduler.empty());
            EXPECT_EQ(cache_manager->freeBlocksNum(), free_blocks);
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, EosAndStopKeepSharedStreamTerminationSemantics) {
    for (bool use_stop_words : {false, true}) {
        auto params = makeMtpParams(3);
        params.model_config_.special_tokens.eos_token_id = 10;
        PPExecutor executor(params, nullptr, true);
        ResourceContext context;
        context.cache_manager = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
        ASSERT_TRUE(context.cache_manager->init());
        auto pp = makeStream(context, params.model_config_, 101, {1, 2}, 0);
        auto reference = makeStream(context, params.model_config_, 202, {1, 2}, 0);
        std::list<GenerateStreamPtr> streams_to_prepare{pp, reference};
        executor.prepareStreams(streams_to_prepare);
        for (const auto& stream : {pp, reference}) {
            stream->fakeInitKVBlock(4);
            stream->setIsContextStream(false);
            stream->generateConfig()->max_new_tokens = 16;
            stream->generateConfig()->ignore_eos = use_stop_words;
            if (use_stop_words) {
                stream->generateConfig()->stop_words_list = {{9, 10}};
            }
        }
        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.new_token_ids = intTensor({9, 10, 11, 12}).reshape({1, 4});
        result.accept_len = intTensor({4});
        result.propose_token_ids = intTensor({20, 21, 22}).reshape({1, 3});
        result.request_errors.resize(1);
        result.prompt_logits.resize(1);
        pp->setPPInflight();
        ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({pp}), result).ok());
        reference->specUpdate({result.new_token_ids, 4, result.propose_token_ids[0], {}, {}, {}, true, false, {}});
        EXPECT_EQ(pp->completeTokenIdsVec(0), reference->completeTokenIdsVec(0));
        EXPECT_TRUE(pp->hasEvent(StreamEvents::GenerateDone));
        EXPECT_TRUE(reference->hasEvent(StreamEvents::GenerateDone));
        EXPECT_FALSE(pp->isPPInflight());
    }
}

TEST_F(PPBatchStreamProcessorTest, MtpPreparePreservesLoadedProposalsAndExpandsFakeTokens) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    ResourceContext resource_context;
    auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
    stream->setIsContextStream(false);
    auto sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer->propose_step = 3;
    sp_output_buffer->tokens = intTensor({10, 11, 12, 13}).reshape({1, 4});
    sp_output_buffer->hidden_states = torch::ones({1, 2});
    stream->setSPOutputBuffer(sp_output_buffer);

    auto fake_stream = makeStream(resource_context, params.model_config_, 102, {1, 2}, 0);
    fake_stream->setIsContextStream(false);
    fake_stream->setIsFakeStream(true);
    auto fake_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    fake_output_buffer->tokens = torch::zeros({1, 2}, torch::kInt32);
    fake_stream->setSPOutputBuffer(fake_output_buffer);

    std::list<GenerateStreamPtr> streams_to_prepare{stream, fake_stream};
    executor.prepareStreams(streams_to_prepare);
    executor.prepareStreams(streams_to_prepare);
    EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
    EXPECT_EQ(tensorToVector<int32_t>(sp_output_buffer->tokens), (std::vector<int32_t>{10, 11, 12, 13}));
    EXPECT_TRUE(torch::equal(sp_output_buffer->hidden_states, torch::ones({1, 2})));
    EXPECT_EQ(fake_stream->getSPOutputBuffer(), fake_output_buffer);
    EXPECT_EQ(fake_output_buffer->propose_step, 3);
    EXPECT_EQ(fake_output_buffer->tokens.sizes().vec(), (std::vector<int64_t>{1, 4}));
}

TEST_F(PPBatchStreamProcessorTest, MtpPrepareRejectsMissingOrIncompleteDecodeProposals) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    ResourceContext resource_context;
    auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
    stream->setIsContextStream(false);
    std::list<GenerateStreamPtr> streams_to_prepare{stream};
    EXPECT_THROW(executor.prepareStreams(streams_to_prepare), std::runtime_error);
    EXPECT_EQ(stream->getSPOutputBuffer(), nullptr);

    auto sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer->tokens = intTensor({10, 11}).reshape({1, 2});
    stream->setSPOutputBuffer(sp_output_buffer);
    EXPECT_THROW(executor.prepareStreams(streams_to_prepare), std::runtime_error);
    EXPECT_EQ(tensorToVector<int32_t>(sp_output_buffer->tokens), (std::vector<int32_t>{10, 11}));
}

TEST_F(PPBatchStreamProcessorTest, MtpPrefillFinishesAtReservedMaxLength) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    ResourceContext resource_context;
    const int input_length = params.model_config_.max_seq_len - 4;
    auto stream = makeStream(resource_context, params.model_config_, 101, std::vector<int32_t>(input_length, 1), 0);
    stream->generateConfig()->max_new_tokens = 32;
    stream->setReserveStep(4);
    std::list<GenerateStreamPtr> streams_to_prepare{stream};
    executor.prepareStreams(streams_to_prepare);
    const auto sp_output_buffer = stream->getSPOutputBuffer();
    EXPECT_EQ(stream->maxTokenNum(), input_length + 1);

    PPExecutionResult result;
    result.request_ids = torch::tensor({101}, torch::kInt64);
    result.new_token_ids = intTensor({10}).reshape({1, 1});
    result.accept_len = intTensor({1});
    result.propose_token_ids = intTensor({11, 12, 13}).reshape({1, 3});
    result.prompt_logits.resize(1);
    result.request_errors.resize(1);
    stream->setPPInflight();

    ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(stream->seqLength(), input_length + 1);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_FALSE(stream->isPPInflight());
    EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
}

TEST_F(PPBatchStreamProcessorTest, MtpCancelledOrFinishedUpdatePreservesProposalBuffer) {
    for (bool cancelled : {false, true}) {
        SCOPED_TRACE(cancelled);
        auto params = makeMtpParams(3);
        PPExecutor executor(params, nullptr, true);
        ResourceContext resource_context;
        auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
        std::list<GenerateStreamPtr> streams_to_prepare{stream};
        executor.prepareStreams(streams_to_prepare);
        const auto sp_output_buffer = stream->getSPOutputBuffer();
        sp_output_buffer->tokens.copy_(intTensor({10, 11, 12, 13}).reshape({1, 4}));
        stream->setProposeToken({10, 11, 12, 13});

        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.new_token_ids = intTensor({20}).reshape({1, 1});
        result.accept_len = intTensor({1});
        result.propose_token_ids = intTensor({21, 22, 23}).reshape({1, 3});
        result.prompt_logits.resize(1);
        result.request_errors.resize(1);
        stream->setPPInflight();
        if (cancelled) {
            stream->reportError(ErrorCode::CANCELLED, "cancelled");
        } else {
            stream->generate_status_->status = StreamState::FINISHED;
        }

        ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_EQ(stream->hasError(), cancelled);
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
        EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
        EXPECT_EQ(tensorToVector<int32_t>(sp_output_buffer->tokens), (std::vector<int32_t>{10, 11, 12, 13}));
        EXPECT_EQ(stream->getProposeToken(), (std::vector<int>{10, 11, 12, 13}));
    }
}

TEST_F(PPBatchStreamProcessorTest, TailExecutionFeedsNextHeadPlanAcrossRounds) {
    for (const auto& [count, type] : std::vector<std::pair<int32_t, SpeculativeType>>{
             {1, SP_TYPE_MTP}, {3, SP_TYPE_MTP}, {4, SP_TYPE_MTP}, {3, SP_TYPE_EAGLE}}) {
        for (bool pd_bootstrap : {false, true}) {
            SCOPED_TRACE("K=" + std::to_string(count) + ", type=" + std::to_string(type)
                         + ", PD=" + std::to_string(pd_bootstrap));
            auto params = makeMtpParams(count);
            params.sp_config.type = type;
            params.pd_sep_config.role_type = pd_bootstrap ? RoleType::DECODE : RoleType::PDFUSION;
            auto cache = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
            ASSERT_TRUE(cache->init());
            PPExecutor head(params, cache, true);
            auto tail_params = params;
            tail_params.parallelism_config.pp_rank = 1;
            tail_params.parallelism_config.world_rank = 1;
            tail_params.parallelism_config.world_size = 2;
            auto propose_params = makeMtpProposeParams(tail_params);
            PPExecutor tail(tail_params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose_params.get());
            auto target = std::make_unique<RecordingDraftModel>();
            auto* target_model = target.get();
            tail.setModel(std::move(target));
            auto draft = std::make_unique<RecordingDraftModel>();
            auto* draft_model = draft.get();
            tail.draft_model_ = std::move(draft);
            const int32_t vocab_offset = type == SP_TYPE_EAGLE ? 17 : 0;
            if (vocab_offset) {
                const auto d2t_map =
                    (torch::arange(64, torch::TensorOptions(torch::kLong).device(torch::kCUDA)) + vocab_offset)
                        .remainder(64);
                tail.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>(d2t_map);
                tail.speculative_sampler_ = std::make_unique<speculative::SpeculativeSampler>(d2t_map, count);
            }
            auto transport = std::make_unique<InMemoryPPTransport>();
            auto* wire = transport.get();
            tail.transport_ = std::move(transport);
            ResourceContext resources;
            resources.cache_manager = cache;
            resources.role_type = params.pd_sep_config.role_type;
            std::vector<std::vector<int32_t>> history{{1, 2}, {3, 4, 5}};
            std::vector<GenerateStreamPtr> streams;
            for (int row = 0; row < 2; ++row) {
                auto stream = makeStream(resources, params.model_config_, 101 + row, history[row], 0);
                stream->generateConfig()->do_sample = false;
                stream->generateConfig()->max_new_tokens = 20;
                stream->fakeInitKVBlock(4);
                streams.push_back(stream);
            }
            std::list<GenerateStreamPtr> batch(streams.begin(), streams.end());
            head.prepareStreams(batch);
            if (pd_bootstrap) {
                /** Incoming D state: one padded handoff and one request with a full steady-state proposal. */
                for (int row = 0; row < 2; ++row) {
                    const int32_t anchor = 20 + row * 10;
                    if (row == 1) {
                        streams[row]->update({intTensor({29}).reshape({1, 1}), 1});
                        history[row].push_back(29);
                    }
                    streams[row]->update({intTensor({anchor}).reshape({1, 1}), 1});
                    streams[row]->initSpeculativeHandoffPositions();
                    history[row].push_back(anchor);
                    std::vector<int32_t> proposals(count + 1, 0);
                    proposals[0] = anchor;
                    proposals[1] = 10 + row + vocab_offset;
                    if (row == 1) {
                        for (int32_t step = 1; step < count; ++step) {
                            proposals[step + 1] = 11 + 2 * step + vocab_offset;
                        }
                    }
                    streams[row]->getSPOutputBuffer()->tokens.copy_(intTensor(proposals).reshape({1, count + 1}));
                    streams[row]->setProposeToken(proposals);
                    streams[row]->setContainProposeToken(true);
                }
                const auto steady_plan = head.batch_stream_processor_->gatherSamplingPlan(StreamGroups({streams[1]}));
                auto init_result = makeInitializedResult(steady_plan);
                tail.batch_stream_processor_->initSamplingStates(steady_plan, tail.sampling_states_, init_result);
                ASSERT_FALSE(init_result.request_errors[0].hasError());
                EXPECT_EQ(tail.sampling_states_.count(101), 0);
            }

            for (int round = pd_bootstrap ? 1 : 0; round < 3; ++round) {
                SCOPED_TRACE(round);
                head.prepareStreams(batch);
                auto plan_status = head.buildPlan(StreamGroups(batch), {});
                ASSERT_TRUE(plan_status.ok()) << plan_status.status().ToString();
                const auto& plan = plan_status.value();
                EXPECT_EQ(plan.is_decode, round != 0);
                EXPECT_EQ(plan.model_input.is_target_verify, round != 0);
                if (round != 0) {
                    EXPECT_TRUE(torch::equal(plan.model_input.lm_output_indexes.cpu(),
                                             torch::arange(2 * (count + 1), torch::kInt32)));
                }
                std::vector<int32_t> lengths{1, 1};
                if (round == 1) {
                    lengths = {pd_bootstrap ? 2 : 1, count + 1};
                } else if (round == 2) {
                    lengths = {count + 1, std::min(3, count + 1)};
                }
                std::vector<std::vector<int32_t>> committed(2);
                target_model->next_tokens.clear();
                for (int row = 0; row < 2; ++row) {
                    if (round == 0) {
                        committed[row] = {20 + row * 10};
                        target_model->next_tokens.push_back(committed[row][0]);
                    } else {
                        const auto previous = streams[row]->getProposeToken();
                        const auto verify_tokens = plan.model_input.combo_tokens.narrow(0, row * (count + 1), count + 1);
                        EXPECT_EQ(tensorToVector<int32_t>(verify_tokens), previous);
                        EXPECT_EQ(plan.model_input.prefix_lengths[row].item<int32_t>(), history[row].size() - 1);
                        EXPECT_EQ(plan.model_input.input_lengths[row].item<int32_t>(), count + 1);
                        committed[row].assign(previous.begin() + 1, previous.begin() + lengths[row]);
                        committed[row].push_back(40 + round * 2 + row);
                        auto target_tokens = std::vector<int32_t>(previous.begin() + 1, previous.end());
                        target_tokens.push_back(committed[row].back());
                        target_tokens[lengths[row] - 1] = committed[row].back();
                        target_model->next_tokens.insert(
                            target_model->next_tokens.end(), target_tokens.begin(), target_tokens.end());
                    }
                }
                for (const auto& object : {pp_serialization::serializePlan(plan, false),
                                           pp_serialization::serializeTensorsMetadata(PPIntermediateTensors{})}) {
                    wire->received_tensors.push_back(torch::tensor({object.numel()}, torch::kInt64));
                    wire->received_tensors.push_back(object);
                }
                const auto sends_before = wire->sent_tensors.size();
                const auto draft_before = draft_model->inputs.size();
                ASSERT_TRUE(tail.process(ScheduleOutput{}).ok());
                ASSERT_EQ(wire->sent_tensors.size(), sends_before + 2);
                const auto result = pp_serialization::deserializeExecutionResult(wire->sent_tensors.back());
                EXPECT_EQ(tensorToVector<int32_t>(result.accept_len), lengths);
                ASSERT_EQ(draft_model->inputs.size(), draft_before + count);
                EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{2, count}));
                if (round != 0) {
                    const auto& update = draft_model->inputs[draft_before];
                    auto expected_tokens = committed[0];
                    expected_tokens.insert(expected_tokens.end(), committed[1].begin(), committed[1].end());
                    EXPECT_EQ(tensorToVector<int32_t>(update.combo_tokens), expected_tokens);
                    EXPECT_EQ(tensorToVector<int32_t>(update.input_lengths), lengths);
                    std::vector<int32_t> hidden_rows;
                    for (int row = 0; row < 2; ++row) {
                        for (int32_t offset = 0; offset < lengths[row]; ++offset) {
                            hidden_rows.push_back(row * (count + 1) + offset);
                        }
                    }
                    const auto expected_hidden =
                        target_model->hidden_buffer.index_select(0, intTensor(hidden_rows).to(torch::kCUDA, torch::kLong));
                    EXPECT_TRUE(torch::equal(update.last_hidden_states, expected_hidden));
                    for (int32_t step = 1; step < count; ++step) {
                        EXPECT_TRUE(torch::equal(
                            draft_model->inputs[draft_before + step].sequence_lengths.cpu(),
                            plan.model_input.prefix_lengths.cpu() + intTensor(lengths) + step - 1));
                    }
                }
                for (int row = 0; row < 2; ++row) {
                    ASSERT_FALSE(result.request_errors[row].hasError()) << result.request_errors[row].ToString();
                    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids[row].narrow(0, 0, lengths[row])), committed[row]);
                    streams[row]->setPPInflight();
                }
                ASSERT_TRUE(head.batch_stream_processor_->dispatchExecutionResult(StreamGroups(batch), result).ok());
                for (int row = 0; row < 2; ++row) {
                    history[row].insert(history[row].end(), committed[row].begin(), committed[row].end());
                    EXPECT_EQ(streams[row]->completeTokenIdsVec(0), history[row]);
                    EXPECT_FALSE(streams[row]->isPPInflight());
                    std::vector<int32_t> expected_proposals{committed[row].back()};
                    for (int32_t step = 0; step < count; ++step) {
                        expected_proposals.push_back(10 + 2 * (draft_before + step) + row + vocab_offset);
                    }
                    EXPECT_EQ(streams[row]->getProposeToken(), expected_proposals);
                }
            }
            auto cleanup = head.buildPlan(StreamGroups{}, {101, 102});
            ASSERT_TRUE(cleanup.ok());
            const auto payload = pp_serialization::serializePlan(cleanup.value(), false);
            wire->received_tensors.push_back(torch::tensor({payload.numel()}, torch::kInt64));
            wire->received_tensors.push_back(payload);
            const auto calls_before = draft_model->inputs.size();
            ASSERT_TRUE(tail.process(ScheduleOutput{}).ok());
            EXPECT_TRUE(tail.sampling_states_.empty());
            EXPECT_EQ(draft_model->inputs.size(), calls_before);
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, NonPositiveSpeculativeWidthIsRejectedAtStartup) {
    auto params = makeMtpParams(3);
    for (const int64_t width : {-1, 0}) {
        SCOPED_TRACE(width);
        params.sp_config.gen_num_per_cycle = width;
        EXPECT_THROW((PPExecutor(params, nullptr, true)), std::runtime_error);
    }
}

TEST_F(PPBatchStreamProcessorTest, MtpPlanAcceptsDefaultSingleSequenceAndRejectsBeamSearch) {
    ResourceContext resource_context;
    EngineInitParams params;
    params.model_id                                 = 0;
    params.model_config_                            = makeModelConfig();
    params.model_config_.num_layers                 = 2;
    params.parallelism_config.pp_size               = 2;
    params.parallelism_config.pp_stage_layer_counts = {1, 1};
    params.sp_config.type                           = SP_TYPE_MTP;
    params.sp_config.gen_num_per_cycle              = 1;
    params.py_model                                 = py::none();
    PPExecutor executor(params, nullptr, /*warm_up=*/true);

    for (int32_t num_return_sequences : {0, 1}) {
        auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, num_return_sequences);
        EXPECT_EQ(stream->currentBatchSize(), 1);
        EXPECT_EQ(stream->nextBatchSize(), 1);
        std::list<GenerateStreamPtr> streams_to_prepare{stream};
        executor.prepareStreams(streams_to_prepare);
        auto plan = executor.buildPlan(StreamGroups({stream}), {});
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        EXPECT_FALSE(plan->is_decode);
        EXPECT_FALSE(plan->model_input.skip_run);
        EXPECT_EQ(plan->sampling_plan.num_return_sequences, std::vector<int32_t>({num_return_sequences}));
        EXPECT_EQ(plan->sampling_plan.token_ids.size(0), 1);
    }

    auto beam_stream = makeStream(resource_context, params.model_config_, 103, {1, 2}, 0);
    beam_stream->generateConfig()->num_beams = 2;
    EXPECT_THROW((void)executor.buildPlan(StreamGroups({beam_stream}), {}), std::runtime_error);
}

TEST_F(PPBatchStreamProcessorTest, DecodePhaseDoesNotDependOnSpeculativeMode) {
    auto params           = makeMtpParams(3);
    params.sp_config.type = SP_TYPE_NONE;
    auto cache_manager =
        std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(cache_manager->init());
    PPExecutor executor(params, cache_manager, true);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
    stream->fakeInitKVBlock(4);
    stream->setIsContextStream(false);

    auto plan = executor.buildPlan(StreamGroups({stream}), {});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_TRUE(plan->is_decode);
    EXPECT_FALSE(plan->model_input.is_target_verify);
}

TEST_F(PPBatchStreamProcessorTest, PlanSerializationPreservesCacheBlocksToZero) {
    PPExecutionPlan plan;
    plan.model_input.kv_cache_blocks_to_zero = intTensor({3, 5, 8});

    const auto round_trip = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));

    EXPECT_TRUE(torch::equal(round_trip.model_input.kv_cache_blocks_to_zero,
                             plan.model_input.kv_cache_blocks_to_zero));
}

TEST_F(PPBatchStreamProcessorTest, MtpDispatchCommitsAcceptedLinearState) {
    struct Case {
        int input_len;
        int accept_len;
        int max_new_tokens;
        std::vector<int32_t> expected_linear_blocks;
        int draft_count = 1;
    };
    const std::vector<Case> cases = {
        {2, 2, 8, {2, 1, 3, 4}},
        {2, 1, 8, {1, 2, 3, 4}},
        {2, 2, 1, {1, 2, 3, 4}},
        {4, 2, 8, {1, 2, 3, 4}},
        {2, 3, 8, {3, 2, 1, 4}, 3},
        {2, 4, 8, {3, 4, 1, 2}, 3},
        {2, 4, 1, {1, 2, 3, 4}, 3},
        {4, 4, 8, {1, 4, 3, 2}, 3},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(::testing::Message() << "input_len=" << c.input_len << " accept_len=" << c.accept_len
                                         << " max_new_tokens=" << c.max_new_tokens);
        auto cache_config = test::makeSimpleHybridMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1);
        ResourceContext resource_context;
        resource_context.cache_manager = std::make_shared<KVCacheManager>(cache_config);
        const auto model_config = makeModelConfig();
        auto stream = makeStream(resource_context, model_config, 101, std::vector<int32_t>(c.input_len, 1), 0);
        stream->generateConfig()->max_new_tokens = c.max_new_tokens;
        auto sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer->propose_step = c.draft_count;
        sp_output_buffer->tokens = torch::zeros({1, c.draft_count + 1}, torch::kInt32);
        stream->setSPOutputBuffer(sp_output_buffer);
        stream->setIsContextStream(false);
        stream->fakeInitKVBlock(4);
        auto& resource = stream->kvCacheMutable().cacheResource(0);
        resource.mutableBlockIds("linear").assign(BlockIndicesType{1, 2, 3, 4});
        resource.mutableBlockIds("full1").assign(BlockIndicesType{5, 6, 7, 8});

        PPBatchStreamProcessor processor(
            model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache_config, true, /*mtp_enabled=*/true);
        PPExecutionResult result;
        result.request_ids    = torch::tensor({101}, torch::kInt64);
        result.new_token_ids  = torch::arange(10, 11 + c.draft_count, torch::kInt32).reshape({1, c.draft_count + 1});
        result.accept_len     = intTensor({c.accept_len});
        result.propose_token_ids = torch::arange(20, 20 + c.draft_count, torch::kInt32).reshape({1, c.draft_count});
        result.prompt_logits.resize(1);
        result.request_errors.resize(1);
        ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->seqLength(), c.input_len + std::min(c.accept_len, c.max_new_tokens));
        EXPECT_EQ(resource.blocks("linear"), c.expected_linear_blocks);
        EXPECT_EQ(resource.kernelBlocks("linear"), c.expected_linear_blocks);
        EXPECT_EQ(resource.blocks("full1"), BlockIndicesType({5, 6, 7, 8}));
    }
}

TEST_F(PPBatchStreamProcessorTest, MixedReturnSequencesPlanRoundTripAndDispatch) {
    ResourceContext resource_context;
    const auto      model_config = makeModelConfig();

    auto multi_stream                           = makeStream(resource_context, model_config, 101, {1, 2}, 2);
    auto single_stream                          = makeStream(resource_context, model_config, 202, {3, 4, 5}, 1);
    auto multi_config                           = multi_stream->generateConfig();
    multi_config->do_sample                     = true;
    multi_config->top_k                         = 7;
    multi_config->combo_token_size              = 2;
    multi_config->banned_combo_token_ids        = {{20, 21}};
    multi_config->end_think_token_ids           = {30, 31};
    multi_config->enable_cross_sequence_ban     = true;
    multi_config->cross_seq_diverge_start_combo = 3;
    single_stream->generateConfig()->do_sample  = true;
    single_stream->generateConfig()->top_k      = 3;

    std::list<GenerateStreamPtr> streams{multi_stream, single_stream};
    StreamGroups                 stream_groups(streams);
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    PPBatchStreamProcessor processor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    PPExecutionPlan plan;
    TensorHolder    holder;
    auto            model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    plan.model_input   = std::move(model_input.value());
    plan.sampling_plan = processor.gatherSamplingPlan(stream_groups);
    plan.output_config = processor.gatherOutputConfig(stream_groups);

    const auto& sampling_plan = plan.sampling_plan;
    EXPECT_EQ(sampling_plan.num_return_sequences, (std::vector<int32_t>{2, 1}));
    EXPECT_EQ(tensorToVector<int64_t>(sampling_plan.request_ids), (std::vector<int64_t>{101, 202}));
    EXPECT_EQ(sampling_plan.token_ids.sizes().vec(), (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.token_ids[0].narrow(0, 0, 2)), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.token_ids[1].narrow(0, 0, 2)), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.token_ids[2].narrow(0, 0, 3)), (std::vector<int32_t>{3, 4, 5}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.top_k), (std::vector<int32_t>{7, 7, 3}));
    ASSERT_EQ(sampling_plan.logits_processor_configs.size(), 2);
    const auto& processor_config = sampling_plan.logits_processor_configs[0];
    EXPECT_EQ(processor_config.combo_token_size, 2);
    EXPECT_EQ(processor_config.banned_combo_token_ids, (std::vector<std::vector<int>>{{20, 21}}));
    EXPECT_EQ(processor_config.end_think_token_ids, (std::vector<int>{30, 31}));
    EXPECT_TRUE(processor_config.enable_cross_sequence_ban);
    EXPECT_EQ(processor_config.cross_seq_diverge_start_combo, 3);

    auto round_trip_plan = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    EXPECT_EQ(round_trip_plan.sampling_plan.num_return_sequences, (std::vector<int32_t>{2, 1}));
    EXPECT_EQ(tensorToVector<int64_t>(round_trip_plan.sampling_plan.request_ids), (std::vector<int64_t>{101, 202}));
    ASSERT_EQ(round_trip_plan.sampling_plan.logits_processor_configs.size(), 2);
    EXPECT_TRUE(round_trip_plan.sampling_plan.logits_processor_configs[0].enable_cross_sequence_ban);
    EXPECT_EQ(round_trip_plan.sampling_plan.logits_processor_configs[0].cross_seq_diverge_start_combo, 3);

    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    sampler_output.token_ids = torch::tensor({10, 11, 12}, torch::kInt32).reshape({3, 1});
    sampler_output.success   = torch::tensor({true, true, true}, torch::kBool);
    auto result = makeInitializedResult(plan.sampling_plan);
    processor.fillExecutionResult(plan, model_output, sampler_output, result);
    EXPECT_EQ(result.request_errors.size(), 2);

    auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    ASSERT_TRUE(processor.dispatchExecutionResult(stream_groups, round_trip_result).ok());
    EXPECT_EQ(multi_stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 10}));
    EXPECT_EQ(multi_stream->completeTokenIdsVec(1), (std::vector<int>{1, 2, 11}));
    EXPECT_EQ(single_stream->completeTokenIdsVec(0), (std::vector<int>{3, 4, 5, 12}));
}

TEST_F(PPBatchStreamProcessorTest, PromptLogitsPlanResultRoundTripAndDispatch) {
    ResourceContext resource_context;
    const auto      model_config = makeModelConfig();

    auto prompt_logits_stream                        = makeStream(resource_context, model_config, 301, {1, 2, 3}, 1);
    auto regular_stream                              = makeStream(resource_context, model_config, 302, {4, 5}, 1);
    auto prompt_logits_config                        = prompt_logits_stream->generateConfig();
    prompt_logits_config->max_new_tokens             = 1;
    prompt_logits_config->return_prompt_logits       = true;
    prompt_logits_config->return_hidden_states       = true;
    prompt_logits_config->prompt_logits_top_k        = 2;
    prompt_logits_config->prompt_logits_start        = 1;
    prompt_logits_config->prompt_logits_end          = 3;
    prompt_logits_config->return_target_logprob      = true;
    regular_stream->generateConfig()->max_new_tokens = 1;
    regular_stream->generateConfig()->return_hidden_states = true;

    std::list<GenerateStreamPtr> streams{prompt_logits_stream, regular_stream};
    StreamGroups                 stream_groups(streams);
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    PPBatchStreamProcessor processor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    PPExecutionPlan plan;
    TensorHolder    holder;
    auto            model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    plan.model_input   = std::move(model_input.value());
    plan.sampling_plan = processor.gatherSamplingPlan(stream_groups);
    plan.output_config = processor.gatherOutputConfig(stream_groups);

    auto round_trip_plan = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    EXPECT_TRUE(round_trip_plan.model_input.need_all_logits);
    ASSERT_EQ(round_trip_plan.output_config.prompt_logits_requests.size(), 2);
    const auto& prompt_logits_request = round_trip_plan.output_config.prompt_logits_requests[0];
    EXPECT_TRUE(prompt_logits_request.enabled);
    EXPECT_EQ(prompt_logits_request.top_k, 2);
    EXPECT_EQ(prompt_logits_request.start, 1);
    EXPECT_EQ(prompt_logits_request.end, 3);
    EXPECT_TRUE(prompt_logits_request.return_target_logprob);
    EXPECT_FALSE(round_trip_plan.output_config.prompt_logits_requests[1].enabled);

    GptModelOutputs model_output;
    const auto      token_count = round_trip_plan.model_input.combo_tokens.numel();
    model_output.hidden_states  = torch::tensor({5.0f, 6.0f, 9.0f, 10.0f}).reshape({2, 2});
    model_output.all_logits     = torch::arange(token_count * model_config.vocab_size, torch::kFloat32)
                                  .reshape({token_count, static_cast<int64_t>(model_config.vocab_size)});

    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({10, 11}, torch::kInt32).reshape({2, 1});
    sampler_output.success   = torch::tensor({true, true}, torch::kBool);
    auto result = makeInitializedResult(round_trip_plan.sampling_plan);
    processor.fillExecutionResult(round_trip_plan, model_output, sampler_output, result);

    auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    ASSERT_EQ(round_trip_result.prompt_logits.size(), 2);
    ASSERT_TRUE(round_trip_result.prompt_logits[0].has_value());
    EXPECT_FALSE(round_trip_result.prompt_logits[1].has_value());
    EXPECT_EQ(round_trip_result.prompt_logits[0]->topk_logprobs.sizes().vec(), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(round_trip_result.prompt_logits[0]->target_logprobs.numel(), 1);

    ASSERT_TRUE(processor.dispatchExecutionResult(stream_groups, round_trip_result).ok());
    auto prompt_logits_output = prompt_logits_stream->nextOutput(1000);
    auto regular_output       = regular_stream->nextOutput(1000);
    ASSERT_TRUE(prompt_logits_output.ok());
    ASSERT_TRUE(regular_output.ok());
    ASSERT_EQ(prompt_logits_output.value().generate_outputs.size(), 1);
    ASSERT_EQ(regular_output.value().generate_outputs.size(), 1);
    EXPECT_TRUE(prompt_logits_output.value().generate_outputs[0].prompt_logits.has_value());
    EXPECT_FALSE(regular_output.value().generate_outputs[0].prompt_logits.has_value());
    ASSERT_TRUE(prompt_logits_output.value().generate_outputs[0].hidden_states.has_value());
    ASSERT_TRUE(regular_output.value().generate_outputs[0].hidden_states.has_value());
    EXPECT_EQ(tensorToVector<float>(*prompt_logits_output.value().generate_outputs[0].hidden_states),
              (std::vector<float>{5.0f, 6.0f}));
    EXPECT_EQ(tensorToVector<float>(*regular_output.value().generate_outputs[0].hidden_states),
              (std::vector<float>{9.0f, 10.0f}));
}

// Regression for the official PP request-error refactor + local fastgen merge.
// The stream has already advanced to the FINAL chunk when an older result arrives.
TEST_F(PPBatchStreamProcessorTest, DelayedChunkResultUsesDispatchSnapshot) {
    ResourceContext resources;
    const auto model_config = makeModelConfig();
    auto stream = makeStream(resources, model_config, 901, {1, 2, 3, 4, 5, 6, 7, 8}, 1);
    stream->enable_fast_gen_ = true;
    stream->resetChunkLen(8, 8);
    ASSERT_FALSE(stream->isChunkStream());
    stream->ppChunkDispatched();
    PPBatchStreamProcessor processor(model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, CacheConfig{}, true);
    PPExecutionResult result;
    result.request_ids = torch::tensor({901}, torch::kInt64);
    result.new_token_ids = intTensor({12}).reshape({1, 1});
    result.prompt_logits.resize(1);
    result.request_errors.resize(1);
    const std::vector<PPStreamRoundSnapshot> intermediate{{1, 4, true}};
    ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result, intermediate).ok());
    EXPECT_EQ(stream->seqLength(), 8);
    EXPECT_TRUE(stream->isContextStream());
    EXPECT_EQ(stream->ppOutstandingResults(), 0);
    const std::vector<PPStreamRoundSnapshot> final_round{{1, 4, false}};
    ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result, final_round).ok());
    EXPECT_EQ(stream->seqLength(), 9);
    EXPECT_FALSE(stream->isContextStream());
}

TEST_F(PPBatchStreamProcessorTest, SnapshotCardinalityRejectsBeforeDispatch) {
    ResourceContext resources;
    const auto model_config = makeModelConfig();
    auto stream = makeStream(resources, model_config, 902, {1, 2}, 1);
    PPBatchStreamProcessor processor(model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, CacheConfig{}, true);
    PPExecutionResult result;
    EXPECT_THROW(processor.dispatchExecutionResult(StreamGroups({stream}), result, {}), std::runtime_error);
    EXPECT_EQ(stream->seqLength(), 2);
    EXPECT_TRUE(stream->isContextStream());
}

}  // namespace rtp_llm
