/**
 * PPBatchStreamProcessor interface tests, ordered by the existing six groups:
 *   1. Plans: sampling configuration, request/sequence coordinates and output flags.
 *   2. Sampling states: request registration, existing state and PD history replay.
 *   3. Sampler inputs: row expansion, token histories, parameters and processor routing.
 *   4. SP model inputs: target verification inputs and draft position IDs.
 *   5. Result assembly: sampled tokens, optional outputs and request-local errors.
 *   6. Dispatch: stream updates, token budgets, linear KV mappings and inflight cleanup.
 *
 * Add cases beside the interface they exercise; keep ordinary, speculative and PD
 * variants in that group. Supply legal stream/plan data and model/sampler outputs
 * directly, then check the interface's outputs and stream changes.
 * Stage orchestration, model/draft execution and communication waits belong in
 * PPExecutorTest; serialization belongs in PPSerializationTest. Real multi-stage
 * communication and overlap belong to smoke tests.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <torch/extension.h>
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#undef protected
#undef private

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

/** Record processor routing without testing the sampler or executor's execution loop. */
class RecordingLogitsProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t start, size_t end) override {
        intervals.emplace_back(start, end);
        return std::nullopt;
    }

    void updateMultiSeqStatus(const std::vector<int>&) override {}

    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        ++updates;
        return std::nullopt;
    }

    std::vector<std::pair<size_t, size_t>> intervals;
    int                                  updates = 0;
};

}

class PPBatchStreamProcessorTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        previous_backend_ = LogitsProcessorFactory::grammarBackend();
        LogitsProcessorFactory::grammarBackend().reset();
        model_config_.max_seq_len                 = 64;
        model_config_.vocab_size                  = 64;
        model_config_.input_vocab_size            = 64;
        model_config_.num_layers                  = 1;
        model_config_.hidden_size                 = 4;
        model_config_.attn_config.head_num        = 2;
        model_config_.attn_config.kv_head_num     = 2;
        model_config_.attn_config.size_per_head   = 2;
        model_config_.special_tokens.eos_token_id = 63;
    }

    void TearDown() override {
        LogitsProcessorFactory::grammarBackend() = std::move(previous_backend_);
        DeviceTestBase::TearDown();
    }

    static GenerateConfig config(int sequences = 1) {
        GenerateConfig result;
        result.num_return_sequences = sequences;
        result.max_new_tokens       = 8;
        result.is_streaming         = true;
        return result;
    }

    GenerateStreamPtr makeStream(int64_t request_id, std::vector<int32_t> tokens, GenerateConfig cfg = config()) {
        auto input              = std::make_shared<GenerateInput>();
        input->request_id      = request_id;
        input->input_ids       = intTensor(std::move(tokens));
        input->generate_config = std::make_shared<GenerateConfig>(std::move(cfg));
        auto stream = std::make_shared<NormalGenerateStream>(
            input, model_config_, RuntimeConfig{}, resources_, nullptr);
        EXPECT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }

    PPBatchStreamProcessor makeProcessor(SpeculativeType type = SP_TYPE_NONE) const {
        return PPBatchStreamProcessor(
            model_config_, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache_config_, true, type);
    }

    /** Result metadata is initialized by PPExecutor::sampleTokens before either processor interface. */
    static PPExecutionResult makeResult(const std::vector<int64_t>& ids) {
        PPExecutionResult result;
        result.request_ids = torch::tensor(ids, torch::kInt64);
        result.request_errors.assign(ids.size(), ErrorInfo::OkStatus());
        result.prompt_logits.resize(ids.size());
        return result;
    }

    static PPExecutionResult makeResult(const PPSamplingPlan& plan) {
        return makeResult(tensorToVector<int64_t>(plan.request_ids));
    }

    static void commitTokens(const GenerateStreamPtr& stream, const torch::Tensor& tokens) {
        stream->updateFromPP({tokens, static_cast<int>(tokens.size(1))});
        ASSERT_FALSE(stream->hasError());
        while (stream->hasOutput()) {
            ASSERT_TRUE(stream->nextOutput(1).ok());
        }
    }

    static void setProposals(const GenerateStreamPtr& stream, std::vector<int32_t> tokens) {
        auto buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->propose_step = tokens.size() - 1;
        buffer->tokens       = intTensor(tokens).reshape({1, static_cast<int64_t>(tokens.size())});
        stream->setSPOutputBuffer(buffer);
        stream->setProposeToken(std::vector<int>(tokens.begin(), tokens.end()));
    }

    void initCache() {
        model_config_.attn_config.tokens_per_block = 4;
        cache_config_ = test::makeSimpleMhaCacheConfig(1, 16, 4, DataType::TYPE_FP16, 2, 2);
        resources_.cache_manager = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(resources_.cache_manager->init());
    }

    void initGrammar() {
        model_config_.vocab_size                 = 4;
        model_config_.input_vocab_size           = 4;
        model_config_.special_tokens.eos_token_id = 3;
        LogitsProcessorFactory::grammarBackend() = XGrammarBackend::create(
            xgrammar_impl::serializeTokenizerInfo(
                {"a", "b", "c", "<eos>"},
                R"({"vocab_size":4,"stop_token_ids":[3],"vocab_type":"RAW","add_prefix_space":false})"),
            GrammarConfig{});
        ASSERT_NE(LogitsProcessorFactory::grammarBackend(), nullptr);
    }

    ModelConfig                     model_config_{};
    CacheConfig                     cache_config_;
    ResourceContext                 resources_;
    std::shared_ptr<XGrammarBackend> previous_backend_;
};

/** 1. Plans: request-level configuration and sequence-level rows must retain their distinct coordinates. */
TEST_F(PPBatchStreamProcessorTest, SamplingPlanKeepsRequestAndSequenceOffsets) {
    auto multi_config = config(2);
    multi_config.random_seed               = 123;
    multi_config.top_k                     = 7;
    multi_config.top_p                     = 0.8f;
    multi_config.temperature               = 0.7f;
    multi_config.repetition_penalty        = 1.5f;
    multi_config.presence_penalty          = 0.2f;
    multi_config.frequency_penalty         = 0.3f;
    multi_config.no_repeat_ngram_size       = 2;
    auto multi = makeStream(101, {1, 2}, multi_config);
    commitTokens(multi, intTensor({10, 63}).reshape({2, 1}));
    auto single = makeStream(202, {3, 4, 5, 6}, config(0));

    auto processor = makeProcessor();
    const auto plan = processor.gatherSamplingPlan(StreamGroups({single, multi}));
    EXPECT_EQ(tensorToVector<int64_t>(plan.request_ids), (std::vector<int64_t>{101, 202}));
    EXPECT_EQ(plan.num_return_sequences, (std::vector<int32_t>{2, 0}));
    ASSERT_EQ(plan.random_seeds.size(), 2);
    EXPECT_EQ(plan.random_seeds[0], std::optional<int>(123));
    EXPECT_FALSE(plan.random_seeds[1].has_value());
    EXPECT_EQ(plan.token_ids.sizes().vec(), (std::vector<int64_t>{3, 5}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.token_ids[0].narrow(0, 0, 3)), (std::vector<int32_t>{1, 2, 10}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.token_ids[1].narrow(0, 0, 3)), (std::vector<int32_t>{1, 2, 63}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.token_ids[2].narrow(0, 0, 4)), (std::vector<int32_t>{3, 4, 5, 6}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.input_lengths), (std::vector<int32_t>{2, 2, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.sequence_lengths), (std::vector<int32_t>{3, 3, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.max_tokens), (std::vector<int32_t>{10, 10, 12}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.top_k), (std::vector<int32_t>{7, 7, 0}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.no_repeat_ngram_size), (std::vector<int32_t>{2, 2, 0}));
    EXPECT_EQ(tensorToVector<bool>(plan.finished_mask), (std::vector<bool>{false, true, false}));
    for (int row = 0; row < 2; ++row) {
        EXPECT_FLOAT_EQ(plan.top_p[row].item<float>(), 0.8f);
        EXPECT_FLOAT_EQ(plan.temperature[row].item<float>(), 0.7f);
        EXPECT_FLOAT_EQ(plan.repetition_penalty[row].item<float>(), 1.5f);
        EXPECT_FLOAT_EQ(plan.presence_penalty[row].item<float>(), 0.2f);
        EXPECT_FLOAT_EQ(plan.frequency_penalty[row].item<float>(), 0.3f);
    }
    EXPECT_FALSE(plan.spec_do_sample.defined());
    EXPECT_FALSE(plan.force_sp_accept.defined());
}

TEST_F(PPBatchStreamProcessorTest, SamplingPlanNormalizesGreedyAndCarriesProcessorConfig) {
    auto cfg = config(2);
    cfg.do_sample                     = false;
    cfg.top_k                         = 7;
    cfg.top_p                         = 0.4f;
    cfg.temperature                   = 0.3f;
    cfg.combo_token_size              = 2;
    cfg.banned_combo_token_ids        = {{20, 21}};
    cfg.end_think_token_ids           = {30, 31};
    cfg.enable_cross_sequence_ban     = true;
    cfg.cross_seq_diverge_start_combo = 3;
    auto stream    = makeStream(101, {1, 2}, cfg);
    auto processor = makeProcessor();
    const auto plan = processor.gatherSamplingPlan(StreamGroups({stream}));
    EXPECT_EQ(tensorToVector<bool>(plan.do_sample), (std::vector<bool>{false, false}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.top_k), (std::vector<int32_t>{1, 1}));
    EXPECT_EQ(tensorToVector<float>(plan.top_p), (std::vector<float>{1, 1}));
    EXPECT_EQ(tensorToVector<float>(plan.temperature), (std::vector<float>{1, 1}));
    ASSERT_EQ(plan.logits_processor_configs.size(), 1);
    const auto& copied = plan.logits_processor_configs[0];
    EXPECT_EQ(copied.combo_token_size, 2);
    EXPECT_EQ(copied.banned_combo_token_ids, (std::vector<std::vector<int>>{{20, 21}}));
    EXPECT_EQ(copied.end_think_token_ids, (std::vector<int>{30, 31}));
    EXPECT_TRUE(copied.enable_cross_sequence_ban);
    EXPECT_EQ(copied.cross_seq_diverge_start_combo, 3);
    EXPECT_EQ(stream->generateConfig()->top_k, 7);
}

TEST_F(PPBatchStreamProcessorTest, SamplingPlanCarriesSpeculativeFlagsPerRequest) {
    auto greedy = config();
    greedy.top_k = 1;
    auto stochastic = config();
    stochastic.top_k = 4;
    stochastic.force_sp_accept = true;
    auto first     = makeStream(101, {1, 2}, greedy);
    auto second    = makeStream(202, {3}, stochastic);
    auto processor = makeProcessor(SP_TYPE_MTP);
    const auto plan = processor.gatherSamplingPlan(StreamGroups({first, second}));
    EXPECT_EQ(tensorToVector<bool>(plan.spec_do_sample), (std::vector<bool>{false, true}));
    EXPECT_EQ(tensorToVector<bool>(plan.force_sp_accept), (std::vector<bool>{false, true}));
}

TEST_F(PPBatchStreamProcessorTest, SamplingPlanRejectsUnsupportedBeamSearch) {
    auto cfg = config();
    cfg.num_beams = 2;
    auto stream = makeStream(101, {1, 2}, cfg);
    auto processor = makeProcessor();
    EXPECT_THROW(processor.gatherSamplingPlan(StreamGroups({stream})), std::runtime_error);
}

TEST_F(PPBatchStreamProcessorTest, OutputConfigCombinesFlagsAndKeepsPromptRequestsSeparate) {
    auto first_config = config();
    first_config.return_logits            = true;
    first_config.return_hidden_states     = true;
    first_config.return_all_hidden_states = true;
    first_config.return_prompt_logits     = true;
    first_config.prompt_logits_top_k      = 2;
    first_config.prompt_logits_start      = 1;
    first_config.prompt_logits_end        = 3;
    first_config.return_target_logprob    = false;
    auto second_config = config();
    second_config.return_softmax_probs = true;
    second_config.return_cum_log_probs = true;
    second_config.calculate_loss       = 2;
    second_config.return_all_probs    = ReturnAllProbsMode::DEFAULT;
    auto first  = makeStream(101, {1, 2, 3}, first_config);
    auto second = makeStream(202, {4, 5}, second_config);
    StreamGroups groups({first, second});
    auto processor = makeProcessor();
    const auto output = processor.gatherOutputConfig(groups);
    EXPECT_TRUE(output.return_logits);
    EXPECT_TRUE(output.return_hidden_states);
    EXPECT_TRUE(output.return_all_hidden_states);
    EXPECT_TRUE(output.return_softmax_probs);
    EXPECT_TRUE(output.return_cum_log_probs);
    EXPECT_TRUE(output.calculate_loss);
    EXPECT_EQ(output.return_all_probs, ReturnAllProbsMode::DEFAULT);
    ASSERT_EQ(output.prompt_logits_requests.size(), 2);
    const auto& prompt = output.prompt_logits_requests[0];
    EXPECT_TRUE(prompt.enabled);
    EXPECT_EQ(prompt.top_k, 2);
    EXPECT_EQ(prompt.start, 1);
    EXPECT_EQ(prompt.end, 3);
    EXPECT_FALSE(prompt.return_target_logprob);
    EXPECT_FALSE(output.prompt_logits_requests[1].enabled);
    second->setReturnAllProbs(ReturnAllProbsMode::ORIGINAL);
    EXPECT_EQ(processor.gatherOutputConfig(groups).return_all_probs, ReturnAllProbsMode::ORIGINAL);
    second->setReturnAllProbs(ReturnAllProbsMode::NONE);
    EXPECT_EQ(processor.gatherOutputConfig(groups).return_all_probs, ReturnAllProbsMode::NONE);
    auto speculative = makeProcessor(SP_TYPE_MTP);
    EXPECT_FALSE(speculative.gatherOutputConfig(groups).return_all_hidden_states);
}

/** 2. Tail sampling states: registration, preservation and PD history replay are request-local. */
TEST_F(PPBatchStreamProcessorTest, SamplingStatesPreserveExistingRequestWhenAddingSequences) {
    auto cfg = config(2);
    cfg.random_seed = 123;
    auto existing  = makeStream(101, {1, 2}, cfg);
    auto processor = makeProcessor();
    auto plan      = processor.gatherSamplingPlan(StreamGroups({existing}));
    auto result    = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);
    ASSERT_TRUE(result.request_errors[0].ok());
    auto& state = states.at(101);
    ASSERT_TRUE(state.generator.defined());
    EXPECT_EQ(state.generator.current_seed(), 123);
    EXPECT_EQ(tensorToVector<float>(state.cum_log_probs), (std::vector<float>{0, 0}));
    const auto generator  = state.generator;
    const auto processors = state.logits_processors;
    ASSERT_EQ(processors.size(), 1);
    torch::rand({4}, generator, std::nullopt, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    const auto rng_state = generator.get_state().clone();
    state.cum_log_probs.copy_(torch::tensor({-2.0f, -4.0f}));

    auto added = makeStream(202, {3}, config(0));
    plan       = processor.gatherSamplingPlan(StreamGroups({existing, added}));
    result     = makeResult(plan);
    processor.initSamplingStates(plan, states, result);
    ASSERT_EQ(states.size(), 2);
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(states.at(101).generator, generator);
    EXPECT_TRUE(torch::equal(states.at(101).generator.get_state(), rng_state));
    EXPECT_EQ(states.at(101).logits_processors, processors);
    EXPECT_EQ(tensorToVector<float>(states.at(101).cum_log_probs), (std::vector<float>{-2, -4}));
    EXPECT_FALSE(states.at(202).generator.defined());
    EXPECT_TRUE(states.at(202).logits_processors.empty());
    EXPECT_EQ(tensorToVector<float>(states.at(202).cum_log_probs), (std::vector<float>{0}));
}

TEST_F(PPBatchStreamProcessorTest, SamplingStatesReplayPdAnchorAtSequenceOffsetOnlyOnce) {
    initGrammar();
    auto processor = makeProcessor();
    auto existing  = makeStream(101, {3, 3}, config(2));
    commitTokens(existing, intTensor({0, 1}).reshape({2, 1}));
    auto plan   = processor.gatherSamplingPlan(StreamGroups({existing}));
    auto result = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);

    auto cfg = config();
    cfg.regex = "abc";
    auto added = makeStream(202, {3}, cfg);
    commitTokens(added, intTensor({0}).reshape({1, 1}));
    plan   = processor.gatherSamplingPlan(StreamGroups({existing, added}));
    result = makeResult(plan);
    ASSERT_EQ(plan.logits_processor_configs[1].grammar_type, "regex");
    EXPECT_EQ(plan.logits_processor_configs[1].grammar_value, "abc");
    processor.initSamplingStates(plan, states, result);
    ASSERT_TRUE(result.request_errors[1].ok());
    const auto processors = states.at(202).logits_processors;
    ASSERT_EQ(processors.size(), 1);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(1));
    result = makeResult(plan);
    processor.initSamplingStates(plan, states, result);
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(states.at(202).logits_processors, processors);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(1));
}

TEST_F(PPBatchStreamProcessorTest, SamplingStatesDoNotReplayThePrefillPromptAsGeneratedTokens) {
    initGrammar();
    auto cfg = config();
    cfg.regex = "abc";
    auto stream = makeStream(101, {2, 1}, cfg);
    auto processor = makeProcessor();
    const auto plan = processor.gatherSamplingPlan(StreamGroups({stream}));
    auto result = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);
    EXPECT_TRUE(result.request_errors[0].ok());
    const auto& processors = states.at(101).logits_processors;
    ASSERT_EQ(processors.size(), 1);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(0));
}

TEST_F(PPBatchStreamProcessorTest, SamplingStateInitializationFailureIsRequestLocal) {
    initGrammar();
    auto cfg = config();
    cfg.regex       = "abc";
    cfg.random_seed = 456;
    auto failed    = makeStream(101, {3}, cfg);
    auto healthy   = makeStream(202, {1, 2});
    auto processor = makeProcessor();
    const auto plan = processor.gatherSamplingPlan(StreamGroups({failed, healthy}));
    /** A tail with no grammar backend must return an error while retaining rows for ordinary sampling. */
    LogitsProcessorFactory::grammarBackend().reset();
    auto result = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(result.request_errors[1].ok());
    ASSERT_EQ(states.size(), 2);
    EXPECT_TRUE(states.at(101).logits_processors.empty());
    EXPECT_EQ(states.at(101).generator.current_seed(), 456);
    EXPECT_EQ(tensorToVector<float>(states.at(101).cum_log_probs), (std::vector<float>{0}));
    const auto inputs = processor.gatherSamplerInputs(plan, PPOutputConfig{}, torch::zeros({2, 4}), states);
    EXPECT_EQ(inputs.batch_size, 2);
    EXPECT_EQ(inputs.generator[0], states.at(101).generator);
    EXPECT_FALSE(inputs.generator[1].defined());
}

/** 3. Sampler inputs: row expansion must preserve histories, parameters and processor routing. */
TEST_F(PPBatchStreamProcessorTest, SamplerInputsRouteMultiSequenceStateAndClampPositiveTopK) {
    model_config_.vocab_size                 = 16;
    model_config_.input_vocab_size           = 16;
    model_config_.special_tokens.eos_token_id = 15;
    auto cfg = config(2);
    cfg.top_k                = 99;
    cfg.random_seed          = 123;
    cfg.return_cum_log_probs = true;
    auto multi  = makeStream(101, {1, 2}, cfg);
    auto single = makeStream(202, {3}, config(0));
    commitTokens(multi, intTensor({10, 11}).reshape({2, 1}));
    auto processor  = makeProcessor();
    StreamGroups groups({multi, single});
    const auto plan = processor.gatherSamplingPlan(groups);
    auto result     = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);
    states.at(101).cum_log_probs.copy_(torch::tensor({-2.0f, -4.0f}));
    auto first_processor  = std::make_shared<RecordingLogitsProcessor>();
    auto second_processor = std::make_shared<RecordingLogitsProcessor>();
    states.at(101).logits_processors = {first_processor};
    states.at(202).logits_processors = {second_processor};
    auto inputs = processor.gatherSamplerInputs(
        plan, processor.gatherOutputConfig(groups), torch::zeros({3, 16}), states);
    EXPECT_EQ(inputs.batch_size, 3);
    EXPECT_EQ(inputs.batch_size_out, 3);
    EXPECT_EQ(inputs.step, 3);
    EXPECT_EQ(tensorToVector<int32_t>(inputs.top_k), (std::vector<int32_t>{16, 16, 0}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.sequence_lengths), (std::vector<int32_t>{3, 3, 1}));
    EXPECT_EQ(tensorToVector<int64_t>(inputs.num_beams_in), (std::vector<int64_t>{1, 1, 1}));
    EXPECT_EQ(tensorToVector<int64_t>(inputs.num_beams_out), (std::vector<int64_t>{1, 1, 1}));
    EXPECT_EQ(tensorToVector<float>(inputs.cum_log_probs), (std::vector<float>{-2, -4, 0}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[0].narrow(0, 0, 3)), (std::vector<int32_t>{1, 2, 10}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[1].narrow(0, 0, 3)), (std::vector<int32_t>{1, 2, 11}));
    EXPECT_EQ(inputs.generator[0], states.at(101).generator);
    EXPECT_EQ(inputs.generator[1], states.at(101).generator);
    EXPECT_FALSE(inputs.generator[2].defined());
    inputs.logits_processor_states_ptr->batchProcess(inputs);
    EXPECT_EQ(first_processor->intervals, (std::vector<std::pair<size_t, size_t>>{{0, 2}}));
    EXPECT_EQ(second_processor->intervals, (std::vector<std::pair<size_t, size_t>>{{2, 3}}));
    EXPECT_EQ(first_processor->updates, 0);
    EXPECT_EQ(second_processor->updates, 0);
}

TEST_F(PPBatchStreamProcessorTest, VerifySamplerInputsExpandEachCandidatePrefix) {
    auto cfg = config();
    cfg.top_k                = 1;
    cfg.repetition_penalty   = 1.7f;
    cfg.presence_penalty     = 0.4f;
    cfg.frequency_penalty    = 0.3f;
    cfg.no_repeat_ngram_size = 2;
    auto first  = makeStream(101, {1, 2}, cfg);
    auto second = makeStream(202, {7});
    commitTokens(first, intTensor({1}).reshape({1, 1}));
    commitTokens(second, intTensor({8}).reshape({1, 1}));
    auto processor  = makeProcessor(SP_TYPE_MTP);
    const auto plan = processor.gatherSamplingPlan(StreamGroups({first, second}));
    auto result     = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);
    auto first_processor  = std::make_shared<RecordingLogitsProcessor>();
    auto second_processor = std::make_shared<RecordingLogitsProcessor>();
    states.at(101).logits_processors = {first_processor};
    states.at(202).logits_processors = {second_processor};
    const auto candidates = intTensor({1, 2, 9, 10, 8, 8, 11, 12});
    auto logits = torch::zeros({8, 64});
    auto inputs = processor.gatherSamplerInputs(plan, PPOutputConfig{}, logits, states, true, 3, candidates);
    EXPECT_EQ(inputs.step, 6);
    EXPECT_EQ(inputs.batch_size, 8);
    EXPECT_EQ(tensorToVector<int32_t>(inputs.sequence_lengths), (std::vector<int32_t>{3, 4, 5, 6, 2, 3, 4, 5}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.input_lengths), (std::vector<int32_t>{2, 2, 2, 2, 1, 1, 1, 1}));
    const std::vector<std::vector<int32_t>> histories{
        {1, 2, 1}, {1, 2, 1, 2}, {1, 2, 1, 2, 9}, {1, 2, 1, 2, 9, 10},
        {7, 8}, {7, 8, 8}, {7, 8, 8, 11}, {7, 8, 8, 11, 12}};
    for (int64_t row = 0; row < 8; ++row) {
        SCOPED_TRACE(row);
        EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[row].narrow(0, 0, histories[row].size())), histories[row]);
        EXPECT_EQ(inputs.no_repeat_ngram_size[row].item<int32_t>(), row < 4 ? 2 : 0);
        EXPECT_FLOAT_EQ(inputs.repetition_penalty[row].item<float>(), row < 4 ? 1.7f : 1.0f);
        EXPECT_FLOAT_EQ(inputs.presence_penalty[row].item<float>(), row < 4 ? 0.4f : 0.0f);
        EXPECT_FLOAT_EQ(inputs.frequency_penalty[row].item<float>(), row < 4 ? 0.3f : 0.0f);
    }
    inputs.logits_processor_states_ptr->batchProcess(inputs);
    EXPECT_EQ(first_processor->intervals, (std::vector<std::pair<size_t, size_t>>{{0, 4}}));
    EXPECT_EQ(second_processor->intervals, (std::vector<std::pair<size_t, size_t>>{{4, 8}}));
    inputs.logits.fill_(7);
    EXPECT_EQ(logits.abs().sum().item<float>(), 0);
}

TEST_F(PPBatchStreamProcessorTest, SamplerInputsPreserveRequestedRawLogitsAndProbabilityMode) {
    model_config_.vocab_size                 = 4;
    model_config_.input_vocab_size           = 4;
    model_config_.special_tokens.eos_token_id = 3;
    auto stream    = makeStream(101, {1, 2});
    auto processor = makeProcessor();
    const auto plan = processor.gatherSamplingPlan(StreamGroups({stream}));
    auto result = makeResult(plan);
    SamplingStates states;
    processor.initSamplingStates(plan, states, result);
    for (bool return_logits : {false, true}) {
        PPOutputConfig output;
        output.return_logits        = return_logits;
        output.return_softmax_probs = !return_logits;
        auto logits = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({1, 4});
        auto inputs = processor.gatherSamplerInputs(plan, output, logits, states);
        inputs.logits.zero_();
        EXPECT_EQ(tensorToVector<float>(logits), (std::vector<float>{1, 2, 3, 4}));
        EXPECT_FALSE(inputs.cum_log_probs.defined());
        EXPECT_FALSE(inputs.all_probs.defined());
    }
    for (auto mode : {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::ORIGINAL}) {
        PPOutputConfig output;
        output.return_all_probs = mode;
        auto inputs = processor.gatherSamplerInputs(plan, output, torch::zeros({1, 4}), states);
        ASSERT_TRUE(inputs.all_probs.defined());
        EXPECT_EQ(inputs.all_probs.sizes().vec(), (std::vector<int64_t>{1, 4}));
        EXPECT_EQ(inputs.all_probs.sum().item<float>(), 0);
        EXPECT_EQ(inputs.return_original_all_probs, mode == ReturnAllProbsMode::ORIGINAL);
    }
}

/** 4. SP model inputs: extend the target anchor with proposals without committing request history. */
TEST_F(PPBatchStreamProcessorTest, DraftNextPositionsFollowEachRequestAndPositionWidth) {
    for (bool mrope : {false, true}) {
        SCOPED_TRACE(mrope);
        model_config_.has_positional_encoding = true;
        model_config_.mm_model_config.mm_position_ids_style =
            mrope ? PositionIdsStyle::MROPE : PositionIdsStyle::DEFAULT;
        model_config_.attn_config.rope_config.index_factor = mrope ? 3 : 1;
        auto first  = makeStream(101, {1, 2});
        auto second = makeStream(202, {3, 4, 5});
        auto processor = makeProcessor(SP_TYPE_MTP);
        TensorHolder holder;
        StreamGroups groups({first, second});
        auto gathered = processor.gatherModelInput(groups, holder);
        ASSERT_TRUE(gathered.ok());
        auto input = std::move(gathered.value());
        auto positions = processor.gatherDraftNextPositionIds(groups, input);
        EXPECT_EQ(tensorToVector<int32_t>(positions),
                  mrope ? (std::vector<int32_t>{1, 1, 1, 2, 2, 2}) : (std::vector<int32_t>{1, 2}));
        EXPECT_TRUE(positions.device().is_cpu());
        EXPECT_TRUE(positions.is_pinned());
        auto ordinary = makeProcessor();
        EXPECT_FALSE(ordinary.gatherDraftNextPositionIds(groups, input).defined());
        input.is_target_verify = true;
        EXPECT_FALSE(processor.gatherDraftNextPositionIds(groups, input).defined());
        input.is_target_verify   = false;
        input.combo_position_ids = torch::Tensor();
        EXPECT_FALSE(processor.gatherDraftNextPositionIds(groups, input).defined());
    }
}

TEST_F(PPBatchStreamProcessorTest, VerifyModelInputUsesCurrentAnchorAndPreservesProposalsAndCache) {
    initCache();
    for (bool mrope : {false, true}) {
        for (int draft_count : {1, 3}) {
            SCOPED_TRACE(::testing::Message() << "mrope=" << mrope << ", drafts=" << draft_count);
            model_config_.has_positional_encoding = true;
            model_config_.mm_model_config.mm_position_ids_style =
                mrope ? PositionIdsStyle::MROPE : PositionIdsStyle::DEFAULT;
            model_config_.attn_config.rope_config.index_factor = mrope ? 3 : 1;
            auto first  = makeStream(101, {1, 2});
            auto second = makeStream(202, {3, 4, 5});
            commitTokens(first, intTensor({10}).reshape({1, 1}));
            commitTokens(second, intTensor({20}).reshape({1, 1}));
            for (const auto& stream : {first, second}) {
                stream->generateContextPositionIds();
                stream->fakeInitKVBlock(2);
            }
            first->kvCacheMutable().cacheResource(0).mutableBlockIds("default").assign(BlockIndicesType{1, 2});
            second->kvCacheMutable().cacheResource(0).mutableBlockIds("default").assign(BlockIndicesType{3, 4});
            const std::vector<int32_t> first_proposals = draft_count == 1 ?
                std::vector<int32_t>{10, 11} : std::vector<int32_t>{10, 11, 12, 13};
            const std::vector<int32_t> second_proposals = draft_count == 1 ?
                std::vector<int32_t>{20, 21} : std::vector<int32_t>{20, 21, 22, 23};
            setProposals(first, first_proposals);
            setProposals(second, second_proposals);
            auto processor = makeProcessor(SP_TYPE_MTP);
            StreamGroups groups({first, second});
            TensorHolder holder;
            auto ordinary = processor.gatherModelInput(groups, holder);
            ASSERT_TRUE(ordinary.ok());
            auto verified = processor.gatherTargetVerifyModelInput(groups, draft_count, holder);
            ASSERT_TRUE(verified.ok());
            const auto& input = verified.value();
            std::vector<int32_t> expected_tokens(first_proposals);
            expected_tokens.insert(expected_tokens.end(), second_proposals.begin(), second_proposals.end());
            EXPECT_EQ(tensorToVector<int32_t>(input.combo_tokens), expected_tokens);
            EXPECT_TRUE(input.combo_tokens.is_cuda());
            EXPECT_TRUE(input.is_target_verify);
            EXPECT_EQ(input.sequence_lengths.numel(), 0);
            EXPECT_EQ(tensorToVector<int32_t>(input.prefix_lengths), (std::vector<int32_t>{2, 3}));
            EXPECT_EQ(tensorToVector<int32_t>(input.input_lengths),
                      (std::vector<int32_t>{draft_count + 1, draft_count + 1}));
            EXPECT_TRUE(torch::equal(input.lm_output_indexes.cpu(),
                                    torch::arange(2 * (draft_count + 1), torch::kInt32)));
            std::vector<int32_t> expected_positions;
            for (int start : {2, 3}) {
                for (int step = 0; step <= draft_count; ++step) {
                    expected_positions.insert(expected_positions.end(), mrope ? 3 : 1, start + step);
                }
            }
            EXPECT_EQ(tensorToVector<int32_t>(input.combo_position_ids), expected_positions);
            EXPECT_EQ(tensorToVector<int32_t>(input.kv_cache_block_id), (std::vector<int32_t>{1, 2, 3, 4}));
            EXPECT_TRUE(torch::equal(input.kv_cache_block_id, ordinary->kv_cache_block_id));
            EXPECT_TRUE(torch::equal(input.kv_cache_kernel_block_id, ordinary->kv_cache_kernel_block_id));
            EXPECT_EQ(input.kv_block_stride_bytes, ordinary->kv_block_stride_bytes);
            EXPECT_EQ(input.seq_size_per_block, ordinary->seq_size_per_block);
            EXPECT_EQ(tensorToVector<int32_t>(first->getSPOutputBuffer()->tokens), first_proposals);
            EXPECT_EQ(tensorToVector<int32_t>(second->getSPOutputBuffer()->tokens), second_proposals);
            EXPECT_EQ(first->completeTokenIdsVec(0), (std::vector<int>{1, 2, 10}));
            EXPECT_EQ(second->completeTokenIdsVec(0), (std::vector<int>{3, 4, 5, 20}));
        }
    }
}

/** 5. Result assembly: preserve request errors and return values in their documented coordinates. */
TEST_F(PPBatchStreamProcessorTest, FillResultSelectsLastSampleColumnAndCopiesRequestedOutputs) {
    auto cfg = config(2);
    cfg.return_logits            = true;
    cfg.return_softmax_probs     = true;
    cfg.return_cum_log_probs     = true;
    cfg.return_all_probs         = ReturnAllProbsMode::DEFAULT;
    cfg.return_hidden_states     = true;
    cfg.return_all_hidden_states = true;
    auto multi  = makeStream(101, {1, 2}, cfg);
    auto single = makeStream(202, {3});
    StreamGroups groups({multi, single});
    auto processor = makeProcessor();
    PPExecutionPlan plan;
    plan.sampling_plan = processor.gatherSamplingPlan(groups);
    plan.output_config = processor.gatherOutputConfig(groups);

    GptModelOutputs output;
    auto raw_logits = torch::zeros({3, 64});
    for (int row = 0; row < 3; ++row) {
        raw_logits[row][7 + row] = std::log(3.0f + row);
    }
    output.logits            = raw_logits.to(torch::kCUDA);
    output.hidden_states     = torch::arange(12, torch::kFloat32).reshape({3, 4}).to(torch::kCUDA);
    output.all_hidden_states = torch::arange(20, torch::kFloat32).reshape({5, 4}).to(torch::kCUDA);
    SamplerOutput sampled;
    sampled.token_ids     = intTensor({1, 2, 7, 1, 2, 8, 3, 0, 9}).reshape({3, 3}).to(torch::kCUDA);
    sampled.success       = torch::tensor({true, true, true}, torch::kBool).to(torch::kCUDA);
    sampled.cum_log_probs = torch::tensor({-1.0f, -2.0f, -3.0f}).to(torch::kCUDA);
    sampled.all_probs     = torch::full({3, 64}, 1.0f / 64, output.logits.options());
    auto result = makeResult(plan.sampling_plan);
    processor.fillExecutionResult(plan, output, sampled, result);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{7, 8, 9}));
    EXPECT_TRUE(result.new_token_ids.device().is_cpu());
    EXPECT_TRUE(torch::equal(result.logits, raw_logits));
    for (int row = 0; row < 3; ++row) {
        EXPECT_NEAR(result.softmax_probs[row][0].item<float>(), (3.0f + row) / (66.0f + row), 1e-6);
    }
    EXPECT_EQ(tensorToVector<float>(result.cum_log_probs), (std::vector<float>{-1, -2, -3}));
    EXPECT_TRUE(torch::equal(result.all_probs, sampled.all_probs.cpu()));
    EXPECT_TRUE(torch::equal(result.hidden_states, output.hidden_states.cpu()));
    EXPECT_TRUE(torch::equal(result.all_hidden_states, output.all_hidden_states.cpu()));
    EXPECT_FALSE(result.loss.defined());
    EXPECT_FALSE(result.new_token_lengths.defined());
    ASSERT_EQ(result.request_errors.size(), 2);
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_TRUE(result.request_errors[1].ok());
}

TEST_F(PPBatchStreamProcessorTest, FillResultPreservesEarlierErrorsAndMapsSamplerRowsToRequests) {
    auto first  = makeStream(101, {1});
    auto multi  = makeStream(202, {2}, config(2));
    auto failed = makeStream(303, {3});
    auto healthy = makeStream(404, {4});
    auto processor = makeProcessor();
    PPExecutionPlan plan;
    plan.sampling_plan = processor.gatherSamplingPlan(StreamGroups({first, multi, failed, healthy}));
    auto result = makeResult(plan.sampling_plan);
    const ErrorInfo initialization_error(ErrorCode::INVALID_PARAMS, "tail grammar backend unavailable");
    const ErrorInfo processor_error(ErrorCode::EXECUTION_EXCEPTION, "logits processor failed");
    result.request_errors[0] = initialization_error;
    SamplerOutput sampled;
    sampled.token_ids = intTensor({0, 10, 0, 0, 12}).reshape({5, 1});
    sampled.success  = torch::tensor({false, true, false, false, true}, torch::kBool);
    sampled.processor_errors.resize(5);
    sampled.processor_errors[3] = processor_error;
    processor.fillExecutionResult(plan, GptModelOutputs{}, sampled, result);
    EXPECT_EQ(result.request_errors[0].ToString(), initialization_error.ToString());
    EXPECT_EQ(result.request_errors[1].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(result.request_errors[2].ToString(), processor_error.ToString());
    EXPECT_TRUE(result.request_errors[3].ok());
    EXPECT_EQ(result.new_token_ids.size(0), 5);
    EXPECT_FALSE(result.logits.defined());
    EXPECT_FALSE(result.softmax_probs.defined());
    EXPECT_FALSE(result.cum_log_probs.defined());
    EXPECT_FALSE(result.all_probs.defined());
    EXPECT_FALSE(result.hidden_states.defined());
    EXPECT_FALSE(result.all_hidden_states.defined());
}

TEST_F(PPBatchStreamProcessorTest, FillResultUsesCompactProbabilityBeforeRestoringTokenIds) {
    model_config_.output_vocab_ids = {0, 2, 7, 63};
    auto cfg = config();
    cfg.return_softmax_probs = true;
    auto failed  = makeStream(101, {1, 2}, cfg);
    auto healthy = makeStream(202, {3}, cfg);
    auto processor = makeProcessor();
    StreamGroups groups({failed, healthy});
    PPExecutionPlan plan;
    plan.sampling_plan = processor.gatherSamplingPlan(groups);
    plan.output_config = processor.gatherOutputConfig(groups);
    GptModelOutputs output;
    output.logits = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, std::log(3.0f), 0.0f})
                        .reshape({2, 4}).to(torch::kCUDA);
    SamplerOutput sampled;
    sampled.token_ids = intTensor({0, 2}).reshape({2, 1}).to(torch::kCUDA);
    sampled.success  = torch::tensor({false, true}, torch::kBool).to(torch::kCUDA);
    auto result = makeResult(plan.sampling_plan);
    processor.fillExecutionResult(plan, output, sampled, result);
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{0, 7}));
    EXPECT_NEAR(result.softmax_probs[1][0].item<float>(), 0.5f, 1e-6);
}

TEST_F(PPBatchStreamProcessorTest, FillResultPromptScoresUseFirstSequenceAndSkipDecodeRows) {
    model_config_.vocab_size                 = 4;
    model_config_.input_vocab_size           = 4;
    model_config_.special_tokens.eos_token_id = 3;
    auto decode_config = config();
    decode_config.return_prompt_logits = true;
    auto decode = makeStream(101, {2}, decode_config);
    commitTokens(decode, intTensor({0}).reshape({1, 1}));
    auto cfg = config(2);
    cfg.calculate_loss        = 2;
    cfg.return_prompt_logits  = true;
    cfg.prompt_logits_top_k   = 2;
    cfg.prompt_logits_start   = 1;
    cfg.prompt_logits_end     = 3;
    auto multi = makeStream(202, {0, 1, 2}, cfg);
    auto short_config = config();
    short_config.return_prompt_logits = true;
    short_config.prompt_logits_top_k  = 2;
    auto single = makeStream(303, {1}, short_config);
    auto processor = makeProcessor();
    StreamGroups groups({multi, single, decode});
    TensorHolder holder;
    auto gathered = processor.gatherModelInput(groups, holder);
    ASSERT_TRUE(gathered.ok());
    PPExecutionPlan plan;
    plan.model_input   = std::move(gathered.value());
    plan.sampling_plan = processor.gatherSamplingPlan(groups);
    plan.output_config = processor.gatherOutputConfig(groups);
    ASSERT_EQ(tensorToVector<int32_t>(plan.model_input.lm_output_indexes), (std::vector<int32_t>{0, 3, 6, 7}));
    GptModelOutputs output;
    output.all_logits = torch::zeros({8, 4});
    output.all_logits[1].copy_(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).log());
    output.all_logits[2].copy_(torch::tensor({4.0f, 3.0f, 2.0f, 1.0f}).log());
    output.all_logits[3].copy_(torch::tensor({1.0f, 3.0f, 2.0f, 4.0f}).log());
    output.all_logits[7].copy_(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).log());
    SamplerOutput sampled;
    sampled.token_ids = intTensor({1, 2, 2, 0}).reshape({4, 1});
    sampled.success  = torch::ones({4}, torch::kBool);
    auto result = makeResult(plan.sampling_plan);
    processor.fillExecutionResult(plan, output, sampled, result);
    ASSERT_EQ(result.loss.numel(), 2);
    EXPECT_NEAR(result.loss[0].item<float>(), std::log(5.0), 1e-6);
    EXPECT_NEAR(result.loss[1].item<float>(), std::log(5.0), 1e-6);
    ASSERT_EQ(result.prompt_logits.size(), 3);
    EXPECT_FALSE(result.prompt_logits[0].has_value());
    ASSERT_TRUE(result.prompt_logits[1].has_value());
    const auto& prompt = result.prompt_logits[1].value();
    EXPECT_EQ(prompt.start_pos, 1);
    EXPECT_EQ(prompt.end_pos, 3);
    EXPECT_EQ(tensorToVector<int32_t>(prompt.topk_token_ids), (std::vector<int32_t>{0, 1, 3, 1}));
    const auto logprobs = tensorToVector<float>(prompt.topk_logprobs);
    ASSERT_EQ(logprobs.size(), 4);
    for (int index = 0; index < 4; ++index) {
        EXPECT_NEAR(logprobs[index], std::log(index % 2 == 0 ? 0.4 : 0.3), 1e-6);
    }
    ASSERT_EQ(prompt.target_logprobs.numel(), 1);
    EXPECT_NEAR(prompt.target_logprobs[0].item<float>(), std::log(0.2), 1e-6);
    ASSERT_TRUE(result.prompt_logits[2].has_value());
    EXPECT_EQ(result.prompt_logits[2]->topk_logprobs.size(0), 1);
    EXPECT_FALSE(result.prompt_logits[2]->target_logprobs.defined());
}

/** 6. Dispatch: commit each request's slice, isolate errors, and retire every returned inflight row. */
TEST_F(PPBatchStreamProcessorTest, NormalDispatchSlicesMixedPhasesAndMultiSequenceOutputs) {
    auto decode = makeStream(101, {1, 2});
    commitTokens(decode, intTensor({9}).reshape({1, 1}));
    auto cfg = config(2);
    cfg.return_logits            = true;
    cfg.return_softmax_probs     = true;
    cfg.return_cum_log_probs     = true;
    cfg.return_all_probs         = ReturnAllProbsMode::DEFAULT;
    cfg.return_hidden_states     = true;
    cfg.return_all_hidden_states = true;
    cfg.calculate_loss           = 2;
    cfg.return_prompt_logits     = true;
    auto multi = makeStream(202, {3, 4, 5}, cfg);
    auto single_config = config();
    single_config.return_hidden_states = true;
    auto single = makeStream(303, {6, 7}, single_config);
    StreamGroups groups({multi, single, decode});
    auto processor = makeProcessor();
    TensorHolder holder;
    ASSERT_TRUE(processor.gatherModelInput(groups, holder).ok());

    auto result = makeResult(std::vector<int64_t>{101, 202, 303});
    result.new_token_ids     = intTensor({10, 11, 12, 13}).reshape({4, 1});
    result.logits            = torch::arange(4 * 64, torch::kFloat32).reshape({4, 64});
    result.hidden_states     = torch::arange(100, 116, torch::kFloat32).reshape({4, 4});
    result.all_hidden_states = torch::arange(200, 236, torch::kFloat32).reshape({9, 4});
    result.softmax_probs     = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({4, 1});
    result.cum_log_probs     = torch::tensor({-1.0f, -2.0f, -3.0f, -4.0f});
    result.all_probs         = torch::zeros({4, 64});
    for (int row = 0; row < 4; ++row) {
        result.all_probs[row][10 + row] = 1;
    }
    result.loss = torch::tensor({0.25f, 0.5f, 0.75f});
    result.prompt_logits[1] = PromptLogitsOutput{
        torch::tensor({-0.5f}).reshape({1, 1}), intTensor({4}).reshape({1, 1}), torch::tensor({-0.5f}), 0, 1};
    for (const auto& stream : groups.allStreams()) {
        stream->setPPInflight();
    }
    ASSERT_TRUE(processor.dispatchExecutionResult(groups, result).ok());
    for (const auto& stream : groups.allStreams()) {
        EXPECT_FALSE(stream->hasError());
        EXPECT_FALSE(stream->isPPInflight());
    }
    EXPECT_EQ(decode->completeTokenIdsVec(0), (std::vector<int>{1, 2, 9, 10}));
    EXPECT_EQ(multi->completeTokenIdsVec(0), (std::vector<int>{3, 4, 5, 11}));
    EXPECT_EQ(multi->completeTokenIdsVec(1), (std::vector<int>{3, 4, 5, 12}));
    EXPECT_EQ(single->completeTokenIdsVec(0), (std::vector<int>{6, 7, 13}));
    auto multi_output = multi->nextOutput(1);
    ASSERT_TRUE(multi_output.ok());
    ASSERT_EQ(multi_output.value().generate_outputs.size(), 2);
    for (int sequence = 0; sequence < 2; ++sequence) {
        SCOPED_TRACE(sequence);
        const auto& output = multi_output.value().generate_outputs[sequence];
        ASSERT_TRUE(output.logits.has_value());
        EXPECT_TRUE(torch::equal(output.logits.value(), result.logits.narrow(0, 1 + sequence, 1)));
        ASSERT_TRUE(output.hidden_states.has_value());
        EXPECT_TRUE(torch::equal(output.hidden_states.value(), result.hidden_states.narrow(0, 1 + sequence, 1)));
        ASSERT_TRUE(output.all_hidden_states.has_value());
        EXPECT_TRUE(torch::equal(output.all_hidden_states.value(), result.all_hidden_states.narrow(0, 1, 6)));
        ASSERT_TRUE(output.loss.has_value());
        EXPECT_EQ(tensorToVector<float>(output.loss.value()), (std::vector<float>{0.25f, 0.5f}));
        ASSERT_TRUE(output.prompt_logits.has_value());
        EXPECT_TRUE(torch::equal(output.prompt_logits->topk_token_ids, result.prompt_logits[1]->topk_token_ids));
        ASSERT_TRUE(output.aux_info.softmax_probs.has_value());
        EXPECT_NEAR(output.aux_info.softmax_probs->item<float>(), sequence == 0 ? 0.2f : 0.3f, 1e-6);
        ASSERT_TRUE(output.aux_info.cum_log_probs.has_value());
        EXPECT_FLOAT_EQ(output.aux_info.cum_log_probs->item<float>(), -2.0f - sequence);
        ASSERT_TRUE(output.aux_info.all_probs.has_value());
        EXPECT_TRUE(torch::equal(output.aux_info.all_probs.value(), result.all_probs.narrow(0, 1 + sequence, 1)));
    }
    auto single_output = single->nextOutput(1);
    ASSERT_TRUE(single_output.ok());
    const auto& single_value = single_output.value().generate_outputs.at(0);
    ASSERT_TRUE(single_value.hidden_states.has_value());
    EXPECT_TRUE(torch::equal(single_value.hidden_states.value(), result.hidden_states.narrow(0, 3, 1)));
    EXPECT_FALSE(single_value.prompt_logits.has_value());
    EXPECT_FALSE(single_value.all_hidden_states.has_value());
    auto decode_output = decode->nextOutput(1);
    ASSERT_TRUE(decode_output.ok());
    EXPECT_FALSE(decode_output.value().generate_outputs.at(0).logits.has_value());
}

TEST_F(PPBatchStreamProcessorTest, NormalDispatchSkipsFailedSequencesWithoutShiftingHealthyRows) {
    auto failed  = makeStream(101, {1, 2}, config(2));
    auto healthy = makeStream(202, {3, 4});
    StreamGroups groups({failed, healthy});
    auto result = makeResult(std::vector<int64_t>{101, 202});
    result.new_token_ids = intTensor({0, 0, 12}).reshape({3, 1});
    result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
    for (const auto& stream : groups.allStreams()) {
        stream->setPPInflight();
    }
    auto processor = makeProcessor();
    ASSERT_TRUE(processor.dispatchExecutionResult(groups, result).ok());
    EXPECT_EQ(failed->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
    EXPECT_EQ(failed->completeTokenIdsVec(1), (std::vector<int>{1, 2}));
    EXPECT_EQ(failed->statusInfo().ToString(), result.request_errors[0].ToString());
    EXPECT_FALSE(failed->hasOutput());
    EXPECT_FALSE(failed->isPPInflight());
    EXPECT_FALSE(healthy->hasError());
    EXPECT_EQ(healthy->completeTokenIdsVec(0), (std::vector<int>{3, 4, 12}));
    EXPECT_TRUE(healthy->hasOutput());
    EXPECT_FALSE(healthy->isPPInflight());
}

TEST_F(PPBatchStreamProcessorTest, SpeculativeDispatchCommitsAcceptedPrefixesAndNextProposals) {
    initCache();
    for (bool decode : {false, true}) {
        SCOPED_TRACE(decode);
        auto first  = makeStream(101, {1, 2});
        auto second = makeStream(202, {3, 4});
        for (const auto& stream : {first, second}) {
            if (decode) {
                commitTokens(stream, intTensor({9}).reshape({1, 1}));
            }
            stream->setPPInflight();
        }
        setProposals(first, decode ? std::vector<int32_t>{9, 10, 11, 12} : std::vector<int32_t>{0, 0, 0, 0});
        setProposals(second, decode ? std::vector<int32_t>{9, 20, 21, 23} : std::vector<int32_t>{0, 0, 0, 0});
        if (!decode) {
            first->setProposeToken({});
            second->setProposeToken({});
        }
        const auto first_history  = first->completeTokenIdsVec(0);
        const auto second_history = second->completeTokenIdsVec(0);
        auto result = makeResult(std::vector<int64_t>{101, 202});
        result.new_token_ids = decode ? intTensor({30, -1, -1, -1, 20, 21, 22, -1}).reshape({2, 4}) :
                                        intTensor({30, 20}).reshape({2, 1});
        result.new_token_lengths = intTensor({1, decode ? 3 : 1});
        result.propose_token_ids = intTensor({31, 32, 33, 23, 24, 25}).reshape({2, 3});
        auto processor = makeProcessor(SP_TYPE_MTP);
        ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({first, second}), result).ok());
        auto expected_first = first_history;
        expected_first.push_back(30);
        auto expected_second = second_history;
        if (decode) {
            expected_second.insert(expected_second.end(), {20, 21, 22});
        } else {
            expected_second.push_back(20);
        }
        EXPECT_EQ(first->completeTokenIdsVec(0), expected_first);
        EXPECT_EQ(second->completeTokenIdsVec(0), expected_second);
        EXPECT_EQ(first->getProposeToken(), (std::vector<int>{30, 31, 32, 33}));
        EXPECT_EQ(second->getProposeToken(), (std::vector<int>{decode ? 22 : 20, 23, 24, 25}));
        EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{1, decode ? 3 : 1}));
        for (const auto& stream : {first, second}) {
            EXPECT_FALSE(stream->hasError());
            EXPECT_FALSE(stream->isPPInflight());
            EXPECT_TRUE(stream->hasOutput());
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, SpeculativeDispatchKeepsFailedPlaceholderOutOfRequestHistory) {
    initCache();
    auto failed  = makeStream(101, {1, 2});
    auto healthy = makeStream(202, {3, 4});
    for (const auto& stream : {failed, healthy}) {
        commitTokens(stream, intTensor({9}).reshape({1, 1}));
        setProposals(stream, {9, 10, 11, 12});
        stream->setPPInflight();
    }
    setProposals(healthy, {9, 20, 22, 23});
    auto result = makeResult(std::vector<int64_t>{101, 202});
    /** Failed draft rows use a valid one-token placeholder so the remaining draft batch can run. */
    result.new_token_ids     = intTensor({0, 0, 0, 0, 20, 21, -1, -1}).reshape({2, 4});
    result.new_token_lengths = intTensor({1, 2});
    result.propose_token_ids = intTensor({10, 11, 12, 22, 23, 24}).reshape({2, 3});
    result.request_errors[0] = ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, "verify failed");
    auto processor = makeProcessor(SP_TYPE_MTP);
    ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({failed, healthy}), result).ok());
    EXPECT_EQ(failed->completeTokenIdsVec(0), (std::vector<int>{1, 2, 9}));
    EXPECT_EQ(failed->getProposeToken(), (std::vector<int>{9, 10, 11, 12}));
    EXPECT_EQ(failed->statusInfo().ToString(), result.request_errors[0].ToString());
    EXPECT_FALSE(failed->hasOutput());
    EXPECT_FALSE(failed->isPPInflight());
    EXPECT_EQ(healthy->completeTokenIdsVec(0), (std::vector<int>{3, 4, 9, 20, 21}));
    EXPECT_EQ(healthy->getProposeToken(), (std::vector<int>{21, 22, 23, 24}));
    EXPECT_FALSE(healthy->hasError());
    EXPECT_FALSE(healthy->isPPInflight());
}

TEST_F(PPBatchStreamProcessorTest, CommitOnlyDispatchKeepsProposalsEmpty) {
    auto stream = makeStream(101, {1, 2});
    auto buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    buffer->propose_step = 3;
    buffer->tokens = torch::zeros({1, 4}, torch::kInt32);
    stream->setSPOutputBuffer(buffer);
    stream->setPPInflight();
    auto result = makeResult(std::vector<int64_t>{101});
    result.new_token_ids     = intTensor({20}).reshape({1, 1});
    result.new_token_lengths = intTensor({1});
    auto processor = makeProcessor(SP_TYPE_DSPARK);
    ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 20}));
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_EQ(stream->getSPOutputBuffer()->tokens.flatten()[0].item<int32_t>(), 20);
    EXPECT_FALSE(stream->isPPInflight());
}

TEST_F(PPBatchStreamProcessorTest, SpeculativeDispatchClipsRequestBudgetWithoutChangingReturnedAcceptance) {
    auto cfg = config();
    cfg.max_new_tokens = 2;
    auto stream = makeStream(101, {1, 2}, cfg);
    commitTokens(stream, intTensor({9}).reshape({1, 1}));
    setProposals(stream, {9, 10, 11, 12});
    stream->setPPInflight();
    auto result = makeResult(std::vector<int64_t>{101});
    result.new_token_ids     = intTensor({10, 11, 12, 13}).reshape({1, 4});
    result.new_token_lengths = intTensor({4});
    result.propose_token_ids = intTensor({14, 15, 16}).reshape({1, 3});
    auto processor = makeProcessor(SP_TYPE_MTP);
    ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 9, 10}));
    EXPECT_EQ(result.new_token_lengths.item<int32_t>(), 4);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_FALSE(stream->isPPInflight());
    auto output = stream->nextOutput(1);
    ASSERT_TRUE(output.ok());
    EXPECT_TRUE(output.value().generate_outputs.at(0).finished);
    EXPECT_EQ(tensorToVector<int32_t>(output.value().generate_outputs.at(0).output_ids), (std::vector<int32_t>{10}));
}

/** Budget clipping must select linear-state blocks using the number of tokens actually committed. */
TEST_F(PPBatchStreamProcessorTest, MtpDispatchCommitsAcceptedLinearState) {
    struct Case {
        int                  input_len;
        int                  num_new_tokens;
        int                  max_new_tokens;
        std::vector<int32_t> expected_linear_blocks;
        int                  draft_count = 1;
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
    model_config_.max_seq_len = 32;
    for (const auto& c : cases) {
        SCOPED_TRACE(::testing::Message() << "input_len=" << c.input_len << " num_new_tokens=" << c.num_new_tokens
                                          << " max_new_tokens=" << c.max_new_tokens);
        cache_config_ = test::makeSimpleHybridMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1);
        resources_.cache_manager = std::make_shared<KVCacheManager>(cache_config_);
        GenerateConfig cfg;
        cfg.num_return_sequences = 0;
        cfg.max_new_tokens       = 20;
        cfg.do_sample            = false;
        auto stream = makeStream(101, std::vector<int32_t>(c.input_len, 1), cfg);
        stream->generateConfig()->max_new_tokens = c.max_new_tokens;
        auto sp_output_buffer                    = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer->propose_step           = c.draft_count;
        sp_output_buffer->tokens                 = torch::zeros({1, c.draft_count + 1}, torch::kInt32);
        stream->setSPOutputBuffer(sp_output_buffer);
        stream->setIsContextStream(false);
        stream->fakeInitKVBlock(4);
        auto& resource = stream->kvCacheMutable().cacheResource(0);
        resource.mutableBlockIds("linear").assign(BlockIndicesType{1, 2, 3, 4});
        resource.mutableBlockIds("full1").assign(BlockIndicesType{5, 6, 7, 8});

        auto processor = makeProcessor(SP_TYPE_MTP);
        PPExecutionResult result;
        result.request_ids       = torch::tensor({101}, torch::kInt64);
        result.new_token_ids     = torch::arange(10, 11 + c.draft_count, torch::kInt32).reshape({1, c.draft_count + 1});
        result.new_token_lengths = intTensor({c.num_new_tokens});
        result.propose_token_ids = torch::arange(20, 20 + c.draft_count, torch::kInt32).reshape({1, c.draft_count});
        result.prompt_logits.resize(1);
        result.request_errors.resize(1);
        ASSERT_TRUE(processor.dispatchExecutionResult(StreamGroups({stream}), result).ok());
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->seqLength(), c.input_len + std::min(c.num_new_tokens, c.max_new_tokens));
        EXPECT_EQ(resource.blocks("linear"), c.expected_linear_blocks);
        EXPECT_EQ(resource.kernelBlocks("linear"), c.expected_linear_blocks);
        EXPECT_EQ(resource.blocks("full1"), BlockIndicesType({5, 6, 7, 8}));
    }
}

TEST_F(PPBatchStreamProcessorTest, DispatchConsumesCancelledOrTimedOutRowsAndPreservesTheFirstError) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP}) {
        for (auto code : {ErrorCode::CANCELLED, ErrorCode::GENERATE_TIMEOUT}) {
            for (bool tail_error : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", error=" << static_cast<int>(code)
                                                  << ", tail_error=" << tail_error);
                auto cancelled = makeStream(101, {1, 2});
                auto healthy   = makeStream(202, {3, 4});
                if (type != SP_TYPE_NONE) {
                    commitTokens(cancelled, intTensor({9}).reshape({1, 1}));
                    commitTokens(healthy, intTensor({9}).reshape({1, 1}));
                }
                StreamGroups groups({cancelled, healthy});
                for (const auto& stream : groups.allStreams()) {
                    stream->setPPInflight();
                    if (type != SP_TYPE_NONE) {
                        setProposals(stream, {9, 10, 11, 12});
                    }
                }
                cancelled->reportError(code, "request terminated while inflight");
                const auto original_error = cancelled->statusInfo();
                auto result = makeResult(std::vector<int64_t>{101, 202});
                result.new_token_ids = intTensor({tail_error ? 0 : 10, 20}).reshape({2, 1});
                if (tail_error) {
                    result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "tail sampling failed");
                }
                if (type != SP_TYPE_NONE) {
                    result.new_token_ids = intTensor({tail_error ? 0 : 30, 0, 0, 0, 20, -1, -1, -1}).reshape({2, 4});
                    result.new_token_lengths = intTensor({1, 1});
                    result.propose_token_ids = intTensor({11, 12, 13, 21, 22, 23}).reshape({2, 3});
                }
                auto processor = makeProcessor(type);
                ASSERT_TRUE(processor.dispatchExecutionResult(groups, result).ok());
                EXPECT_EQ(cancelled->statusInfo().ToString(), original_error.ToString());
                EXPECT_EQ(cancelled->completeTokenIdsVec(0), type == SP_TYPE_NONE ?
                              (std::vector<int>{1, 2}) : (std::vector<int>{1, 2, 9}));
                EXPECT_FALSE(cancelled->hasOutput());
                EXPECT_FALSE(cancelled->isPPInflight());
                if (type != SP_TYPE_NONE) {
                    EXPECT_EQ(cancelled->getProposeToken(), (std::vector<int>{9, 10, 11, 12}));
                    EXPECT_EQ(tensorToVector<int32_t>(cancelled->getSPOutputBuffer()->tokens),
                              (std::vector<int32_t>{9, 10, 11, 12}));
                }
                EXPECT_EQ(healthy->completeTokenIdsVec(0), type == SP_TYPE_NONE ?
                              (std::vector<int>{3, 4, 20}) : (std::vector<int>{3, 4, 9, 20}));
                EXPECT_FALSE(healthy->hasError());
                EXPECT_TRUE(healthy->hasOutput());
                EXPECT_FALSE(healthy->isPPInflight());
            }
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, DispatchRetiresInflightWhenEveryRequestWasCancelled) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP}) {
        auto first  = makeStream(101, {1, 2});
        auto second = makeStream(202, {3, 4});
        if (type != SP_TYPE_NONE) {
            commitTokens(first, intTensor({9}).reshape({1, 1}));
            commitTokens(second, intTensor({9}).reshape({1, 1}));
        }
        StreamGroups groups({first, second});
        for (const auto& stream : groups.allStreams()) {
            stream->setPPInflight();
            if (type != SP_TYPE_NONE) {
                setProposals(stream, {9, 10, 11, 12});
            }
            stream->reportError(ErrorCode::CANCELLED, "cancelled while inflight");
        }
        auto result = makeResult(std::vector<int64_t>{101, 202});
        result.new_token_ids = intTensor({10, 20}).reshape({2, 1});
        if (type != SP_TYPE_NONE) {
            result.new_token_ids = intTensor({30, -1, -1, -1, 20, -1, -1, -1}).reshape({2, 4});
            result.new_token_lengths = intTensor({1, 1});
            result.propose_token_ids = intTensor({11, 12, 13, 21, 22, 23}).reshape({2, 3});
        }
        auto processor = makeProcessor(type);
        ASSERT_TRUE(processor.dispatchExecutionResult(groups, result).ok());
        EXPECT_EQ(first->completeTokenIdsVec(0), type == SP_TYPE_NONE ?
                      (std::vector<int>{1, 2}) : (std::vector<int>{1, 2, 9}));
        EXPECT_EQ(second->completeTokenIdsVec(0), type == SP_TYPE_NONE ?
                      (std::vector<int>{3, 4}) : (std::vector<int>{3, 4, 9}));
        for (const auto& stream : groups.allStreams()) {
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
            EXPECT_FALSE(stream->hasOutput());
            EXPECT_FALSE(stream->isPPInflight());
        }
    }
}

}
