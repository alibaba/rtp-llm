#include <cstdint>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"

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
        if (!hidden_buffer.defined()) {
            hidden_buffer = torch::empty({rows, 4}, options);
        }
        hidden_buffer.narrow(0, 0, rows).copy_(torch::arange(rows * 4, options).reshape({rows, 4}) + step * 100);
        if (input.last_hidden_states.defined()) {
            EXPECT_TRUE(torch::equal(input.last_hidden_states, inputs.back().last_hidden_states));
        }
        GptModelOutputs output;
        output.all_hidden_states = hidden_buffer.narrow(0, 0, rows).narrow(1, 0, 2);
        output.logits = torch::zeros({input.input_lengths.numel(), 64}, options);
        for (int64_t row = 0; row < input.input_lengths.numel(); ++row) {
            output.logits[row][10 + step * 2 + row] = 100;
        }
        return output;
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t rows) override {
        return hidden_buffer.narrow(0, 0, rows);
    }

    std::vector<GptModelInputs> inputs;
    torch::Tensor hidden_buffer;
};

class RecordingLogitsProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }

    void updateMultiSeqStatus(const std::vector<int>&) override {}

    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t) override {
        committed_tokens.push_back(new_tokens.clone());
        return std::nullopt;
    }

    std::vector<torch::Tensor> committed_tokens;
};

class CompletedPPTicket: public PPCommTicket {
public:
    using PPCommTicket::PPCommTicket;
    void wait() override {}
};

class InMemoryPPTransport: public PPTransport {
public:
    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override {
        sent_tensors.push_back(tensor.clone());
        return std::make_unique<CompletedPPTicket>(tensor);
    }

    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override {
        tensor.copy_(received_tensors.at(receive_index++));
        return std::make_unique<CompletedPPTicket>(tensor);
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
        return params;
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

TEST_F(PPBatchStreamProcessorTest, ProposeDraftTokensRunsAllForwardsAndPreservesMtpHiddenStates) {
    for (int64_t count : {1, 2, 4}) {
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
            EXPECT_TRUE(torch::equal(input.sequence_lengths.cpu(), intTensor({6, 9}) + step - 1));
            EXPECT_TRUE(torch::equal(input.combo_position_ids.cpu(), intTensor({5, 15, 8, 18}) + step - 1));
            auto expected_hidden = step == 1 ? torch::tensor({{4.f, 5.f, 6.f, 7.f}, {16.f, 17.f, 18.f, 19.f}}) :
                                              torch::arange(8, torch::kFloat32).reshape({2, 4}) + (step - 1) * 100;
            EXPECT_TRUE(torch::equal(input.last_hidden_states.cpu(), expected_hidden));
        }
        EXPECT_EQ(tensorToVector<int32_t>(initial_input.input_lengths), (std::vector<int32_t>{2, 3}));
        EXPECT_EQ(tensorToVector<int32_t>(initial_input.prefix_lengths), (std::vector<int32_t>{3, 5}));
    }
}

TEST_F(PPBatchStreamProcessorTest, SamplingFailurePreservesPlanPhaseAndReturnsBeforeDraftForward) {
    auto params = makeMtpParams(3);
    PPExecutor first_stage(params, nullptr, true);
    ResourceContext resource_context;
    auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
    auto plan = first_stage.buildPlan(StreamGroups({stream}), {});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_FALSE(plan->is_decode);
    plan->sampling_plan.logits_processor_configs[0].grammar_type = "invalid";

    params.pd_sep_config.role_type = RoleType::DECODE;
    params.parallelism_config.pp_rank = 1;
    params.parallelism_config.world_rank = 1;
    params.parallelism_config.world_size = 2;
    PPExecutor last_stage(params, nullptr, false);
    auto target_model = std::make_unique<RecordingDraftModel>();
    auto* recorded_target = target_model.get();
    last_stage.setModel(std::move(target_model));
    auto draft_model = std::make_unique<RecordingDraftModel>();
    auto* recorded_draft = draft_model.get();
    last_stage.draft_model_ = std::move(draft_model);
    auto transport = std::make_unique<InMemoryPPTransport>();
    auto* recorded_transport = transport.get();
    for (const auto& object : {pp_serialization::serializePlan(plan.value(), false),
                               pp_serialization::serializeTensorsMetadata(PPIntermediateTensors{})}) {
        transport->received_tensors.push_back(torch::tensor({object.numel()}, torch::kInt64));
        transport->received_tensors.push_back(object);
    }
    last_stage.transport_ = std::move(transport);

    const auto status = last_stage.process(ScheduleOutput{});
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "unsupported grammar type: invalid");
    EXPECT_EQ(recorded_target->inputs.size(), 1);
    EXPECT_TRUE(recorded_draft->inputs.empty());
    EXPECT_TRUE(recorded_transport->sent_tensors.empty());
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
        auto result = executor.sampleTokens(plan.value(), output);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        EXPECT_TRUE(torch::equal(result->new_token_ids.flatten(), torch::arange(20, 20 + batch_size, torch::kInt32)));
        EXPECT_EQ(result->new_token_ids.sizes().vec(), (std::vector<int64_t>{batch_size, 1}));
        EXPECT_EQ(result->accept_len.defined(), mtp_enabled);
        EXPECT_FALSE(result->propose_token_ids.defined());
        EXPECT_TRUE(processor->committed_tokens.empty());
        if (mtp_enabled) {
            EXPECT_EQ(tensorToVector<int32_t>(result->accept_len), (std::vector<int32_t>{1}));
            auto target_model = std::make_unique<RecordingDraftModel>();
            target_model->hidden_buffer = torch::arange(8, output.logits.options()).reshape({2, 4});
            executor.setModel(std::move(target_model));
            auto draft_model = std::make_unique<RecordingDraftModel>();
            auto* recorded_draft = draft_model.get();
            executor.draft_model_ = std::move(draft_model);
            executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>();
            executor.draftSampleAndPropose(plan.value(), output, result.value());
            EXPECT_EQ(tensorToVector<int32_t>(result->propose_token_ids), (std::vector<int32_t>{10, 12, 14}));
            ASSERT_EQ(recorded_draft->inputs.size(), 3);
            EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[0].combo_tokens), (std::vector<int32_t>{2, 20}));
            EXPECT_TRUE(processor->committed_tokens.empty());
            EXPECT_EQ(tensorToVector<int32_t>(plan->model_input.combo_tokens), (std::vector<int32_t>{1, 2}));
        }
        executor.advanceSamplingStates(plan->sampling_plan, result.value());
        ASSERT_EQ(processor->committed_tokens.size(), 1);
        EXPECT_TRUE(torch::equal(processor->committed_tokens[0], result->new_token_ids));
        EXPECT_TRUE(torch::equal(state.cum_log_probs, result->cum_log_probs));
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
    first->generateConfig()->top_k = 1;
    second->generateConfig()->top_k = 1;
    PPExecutionPlan plan;
    plan.is_decode = true;
    plan.sampling_plan = executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({first, second}));
    plan.model_input.combo_tokens = intTensor({2, 11, 12, 13, 4, 21, 22, 23});
    GptModelOutputs output;
    output.logits = torch::zeros({8, 64}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    const std::vector<int64_t> target_tokens{11, 12, 13, 14, 29, 22, 23, 24};
    for (int64_t row = 0; row < 8; ++row) {
        output.logits[row][target_tokens[row]] = 100;
    }
    auto result = executor.sampleTokens(plan, output);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->new_token_ids.sizes().vec(), (std::vector<int64_t>{2, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(result->accept_len), (std::vector<int32_t>{4, 1}));
    EXPECT_FALSE(result->propose_token_ids.defined());

    auto first_processor = std::make_shared<RecordingLogitsProcessor>();
    auto second_processor = std::make_shared<RecordingLogitsProcessor>();
    executor.sampling_states_.at(101).logits_processors = {first_processor};
    executor.sampling_states_.at(202).logits_processors = {second_processor};
    executor.advanceSamplingStates(plan.sampling_plan, result.value());
    ASSERT_EQ(first_processor->committed_tokens.size(), 1);
    ASSERT_EQ(second_processor->committed_tokens.size(), 1);
    EXPECT_EQ(tensorToVector<int32_t>(first_processor->committed_tokens[0]), (std::vector<int32_t>{11, 12, 13, 14}));
    EXPECT_EQ(tensorToVector<int32_t>(second_processor->committed_tokens[0]), (std::vector<int32_t>{29}));

    result->processor_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "processor failed");
    result->sample_success[1] = false;
    executor.advanceSamplingStates(plan.sampling_plan, result.value());
    EXPECT_EQ(first_processor->committed_tokens.size(), 1);
    EXPECT_EQ(second_processor->committed_tokens.size(), 1);
}

TEST_F(PPBatchStreamProcessorTest, DraftDecodeInputUsesOnlyAcceptedRowsWithoutMutatingVerifyInput) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, true);
    executor.position_id_len_factor_ = 1;
    auto target = std::make_unique<RecordingDraftModel>();
    target->hidden_buffer = torch::arange(32, torch::kFloat32).reshape({8, 4});
    executor.setModel(std::move(target));
    GptModelInputs verify_input;
    verify_input.is_target_verify = true;
    verify_input.combo_tokens = intTensor({1, 2, 3, 4, 5, 6, 7, 8});
    verify_input.input_lengths = intTensor({4, 4});
    verify_input.prefix_lengths = intTensor({5, 9});
    verify_input.sequence_lengths = intTensor({});
    verify_input.lm_output_indexes = torch::arange(8, torch::kInt32);
    verify_input.combo_position_ids = intTensor({5, 6, 7, 8, 9, 10, 11, 12});
    GptModelOutputs target_output;
    target_output.all_hidden_states = torch::zeros({8, 2});
    auto accepted_tokens = intTensor({10, 0, 0, 0, 20, 21, 22, 0}).reshape({2, 4});
    auto draft_input = executor.prepareDraftInputForDecode(
        verify_input, target_output, accepted_tokens, intTensor({1, 3}));
    EXPECT_FALSE(draft_input.is_target_verify);
    EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_tokens), (std::vector<int32_t>{10, 20, 21, 22}));
    EXPECT_EQ(tensorToVector<int32_t>(draft_input.input_lengths), (std::vector<int32_t>{1, 3}));
    EXPECT_EQ(tensorToVector<int32_t>(draft_input.lm_output_indexes), (std::vector<int32_t>{0, 3}));
    EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_position_ids), (std::vector<int32_t>{5, 9, 10, 11}));
    auto expected_hidden = torch::arange(32, torch::kFloat32).reshape({8, 4}).index_select(
        0, torch::tensor({0, 4, 5, 6}, torch::kLong));
    EXPECT_TRUE(torch::equal(draft_input.last_hidden_states, expected_hidden));
    EXPECT_TRUE(verify_input.is_target_verify);
    EXPECT_EQ(tensorToVector<int32_t>(verify_input.input_lengths), (std::vector<int32_t>{4, 4}));
    EXPECT_EQ(verify_input.combo_tokens.numel(), 8);
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

    executor.prepareStreams({stream, fake_stream});
    executor.prepareStreams({stream, fake_stream});
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
    EXPECT_THROW(executor.prepareStreams({stream}), std::runtime_error);
    EXPECT_EQ(stream->getSPOutputBuffer(), nullptr);

    auto sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer->tokens = intTensor({10, 11}).reshape({1, 2});
    stream->setSPOutputBuffer(sp_output_buffer);
    EXPECT_THROW(executor.prepareStreams({stream}), std::runtime_error);
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
    executor.prepareStreams({stream});
    const auto sp_output_buffer = stream->getSPOutputBuffer();
    EXPECT_EQ(stream->maxTokenNum(), input_length + 1);

    PPExecutionResult result;
    result.request_ids = torch::tensor({101}, torch::kInt64);
    result.new_token_ids = intTensor({10}).reshape({1, 1});
    result.accept_len = intTensor({1});
    result.propose_token_ids = intTensor({11, 12, 13}).reshape({1, 3});
    result.sample_success = torch::ones({1}, torch::kBool);
    result.prompt_logits.resize(1);
    result.processor_errors.resize(1);
    stream->setPPInflight();

    ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(stream->seqLength(), input_length + 1);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_FALSE(stream->isPPInflight());
    EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
}

TEST_F(PPBatchStreamProcessorTest, MtpFailedOrCancelledUpdatePreservesProposalBuffer) {
    for (bool cancelled : {false, true}) {
        SCOPED_TRACE(cancelled);
        auto params = makeMtpParams(3);
        PPExecutor executor(params, nullptr, true);
        ResourceContext resource_context;
        auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
        executor.prepareStreams({stream});
        const auto sp_output_buffer = stream->getSPOutputBuffer();
        sp_output_buffer->tokens.copy_(intTensor({10, 11, 12, 13}).reshape({1, 4}));
        stream->setProposeToken({10, 11, 12, 13});

        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.new_token_ids = intTensor({20}).reshape({1, 1});
        result.accept_len = intTensor({cancelled ? 1 : 0});
        result.propose_token_ids = intTensor({21, 22, 23}).reshape({1, 3});
        result.sample_success = torch::full({1}, cancelled, torch::kBool);
        result.prompt_logits.resize(1);
        result.processor_errors.resize(1);
        stream->setPPInflight();
        if (cancelled) {
            stream->reportError(ErrorCode::CANCELLED, "cancelled");
        }

        ASSERT_TRUE(executor.batch_stream_processor_->dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_TRUE(stream->hasError());
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
        EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
        EXPECT_EQ(tensorToVector<int32_t>(sp_output_buffer->tokens), (std::vector<int32_t>{10, 11, 12, 13}));
        EXPECT_EQ(stream->getProposeToken(), (std::vector<int>{10, 11, 12, 13}));
    }
}

TEST_F(PPBatchStreamProcessorTest, MtpDispatchAndVerifyPlanCarryAllDraftTokens) {
    for (int64_t count : {1, 3}) {
        auto params = makeMtpParams(count);
        auto cache_manager = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
        ASSERT_TRUE(cache_manager->init());
        PPExecutor executor(params, cache_manager, true);
        auto& processor = *executor.batch_stream_processor_;
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        auto stream = makeStream(resource_context, params.model_config_, 101, {1, 2}, 0);
        stream->generateConfig()->max_new_tokens = 20;
        stream->fakeInitKVBlock(4);
        executor.prepareStreams({stream});
        const auto sp_output_buffer = stream->getSPOutputBuffer();
        ASSERT_NE(sp_output_buffer, nullptr);
        auto recording_processor = std::make_shared<RecordingLogitsProcessor>();
        stream->sampling_state_.logits_processors.push_back(recording_processor);
        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.new_token_ids = intTensor({10}).reshape({1, 1});
        result.accept_len = intTensor({1});
        result.propose_token_ids = torch::arange(11, 11 + count, torch::kInt32).reshape({1, count});
        result.sample_success = torch::ones({1}, torch::kBool);
        result.prompt_logits.resize(1);
        result.processor_errors.resize(1);

        if (count > 1) {
            auto incomplete_result = result;
            incomplete_result.propose_token_ids = result.propose_token_ids.narrow(1, 0, count - 1);
            EXPECT_THROW((void)processor.dispatchExecutionResult(StreamGroups({stream}), incomplete_result),
                         std::runtime_error);
            EXPECT_EQ(stream->seqLength(), 2);
        }

        result = pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
        ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result).ok());
        ASSERT_FALSE(stream->hasError());
        EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
        EXPECT_EQ(stream->seqLength(), 3);
        EXPECT_EQ(stream->getSPOutputBuffer()->propose_step, count);
        EXPECT_EQ(stream->getProposeToken(), tensorToVector<int32_t>(torch::arange(10, 11 + count, torch::kInt32)));

        for (int64_t accepted : {int64_t{1}, count + 1}) {
            executor.prepareStreams({stream});
            auto& tokens = stream->getSPOutputBuffer()->tokens;
            tokens = tokens.reshape({1, count + 1});
            auto plan = executor.buildPlan(StreamGroups({stream}), {});
            ASSERT_TRUE(plan.ok()) << plan.status().ToString();
            EXPECT_TRUE(plan->is_decode);
            EXPECT_TRUE(plan->model_input.is_target_verify);
            EXPECT_TRUE(torch::equal(plan->model_input.combo_tokens.cpu(), tokens.flatten()));
            EXPECT_EQ(tensorToVector<int32_t>(plan->model_input.input_lengths),
                      (std::vector<int32_t>{static_cast<int32_t>(count + 1)}));
            EXPECT_EQ(tensorToVector<int32_t>(plan->model_input.prefix_lengths),
                      (std::vector<int32_t>{stream->seqLength() - 1}));

            const auto previous_length = stream->seqLength();
            result.new_token_ids = torch::arange(20, 21 + count, torch::kInt32).reshape({1, count + 1});
            result.accept_len = intTensor({static_cast<int32_t>(accepted)});
            result.propose_token_ids = torch::arange(30, 30 + count, torch::kInt32).reshape({1, count});
            ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result).ok());
            ASSERT_FALSE(stream->hasError());
            EXPECT_EQ(stream->getSPOutputBuffer(), sp_output_buffer);
            EXPECT_EQ(stream->seqLength(), previous_length + accepted);
            EXPECT_EQ(stream->getSPOutputBuffer()->tokens.flatten()[0].item<int32_t>(), 19 + accepted);
            EXPECT_TRUE(torch::equal(stream->getSPOutputBuffer()->tokens.flatten().narrow(0, 1, count),
                                     result.propose_token_ids.flatten()));
        }
        EXPECT_TRUE(recording_processor->committed_tokens.empty());
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
        executor.prepareStreams({stream});
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
        result.sample_success = torch::ones({1}, torch::kBool);
        result.prompt_logits.resize(1);
        result.processor_errors.resize(1);
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
    auto result              = processor.makeExecutionResult(plan, model_output, sampler_output);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->processor_errors.size(), 3);

    auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result.value()));
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
    auto result              = processor.makeExecutionResult(round_trip_plan, model_output, sampler_output);
    ASSERT_TRUE(result.ok());

    auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result.value()));
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

}  // namespace rtp_llm
