#include "autil/EnvUtil.h"
/**
 * PPBatchStreamProcessor interface tests, organized by responsibility:
 *   1. Plans: request/sequence coordinates, history deltas, sampling configuration and output flags.
 *   2. Sampling preparation: request-state lifetime, PD history replay and verification Q.
 *   3. Sampler inputs: history reconstruction, candidate prefixes, parameters and processor routing.
 *   4. SP model inputs: target verification, draft preparation, positions and Engram windows.
 *   5. Result assembly: sampled tokens, optional outputs and request-local errors.
 *   6. Dispatch: stream updates, token budgets, linear KV mappings and inflight cleanup.
 *
 * Add cases beside the interface they exercise; keep ordinary, speculative and PD variants together.
 * Build stream-derived plans through gatherExecutionPlan and reuse history cursors across rounds.
 * Supply model/sampler outputs explicitly, then check interface outputs and stream changes.
 * Direct tensor inputs are appropriate for focused model-input adaptation tests.
 *
 * Request admission, stage orchestration, model execution and communication waits belong in PPExecutorTest.
 * Serialization belongs in PPSerializationTest; real multi-stage communication and overlap belong in smoke tests.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

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

/** Record batch intervals and updates to detect request-state duplication or misrouting. */
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
    int                                    updates = 0;
};

}  // namespace

class PPBatchStreamProcessorTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        previous_backend_ = LogitsProcessorFactory::grammarBackend();
        LogitsProcessorFactory::grammarBackend().reset();
        model_config_.max_seq_len = 64;
        setVocab(64);
        model_config_.num_layers                = 1;
        model_config_.hidden_size               = 4;
        model_config_.attn_config.head_num      = 2;
        model_config_.attn_config.kv_head_num   = 2;
        model_config_.attn_config.size_per_head = 2;
    }

    void TearDown() override {
        LogitsProcessorFactory::grammarBackend() = std::move(previous_backend_);
        DeviceTestBase::TearDown();
    }

    /** vocab_size, input_vocab_size and eos_token_id move together; eos is always vocab_size - 1. */
    void setVocab(int64_t size) {
        model_config_.vocab_size                  = size;
        model_config_.input_vocab_size            = size;
        model_config_.special_tokens.eos_token_id = size - 1;
    }

    static GenerateConfig config(int sequences = 1) {
        GenerateConfig result;
        result.num_return_sequences = sequences;
        result.max_new_tokens       = 8;
        result.is_streaming         = true;
        return result;
    }

    GenerateStreamPtr makeStream(int64_t request_id, std::vector<int32_t> tokens, GenerateConfig cfg = config()) {
        auto input             = std::make_shared<GenerateInput>();
        input->request_id      = request_id;
        input->input_ids       = intTensor(std::move(tokens));
        input->generate_config = std::make_shared<GenerateConfig>(std::move(cfg));
        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config_, RuntimeConfig{}, resources_, nullptr);
        EXPECT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }

    PPBatchStreamProcessor makeProcessor(SpeculativeType type         = SP_TYPE_NONE,
                                         RoleType        role_type    = RoleType::PDFUSION,
                                         int64_t         propose_step = 3) const {
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role_type;
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = type;
        sp_config.gen_num_per_cycle = propose_step;
        return PPBatchStreamProcessor(
            model_config_, pd_sep_config, ProfilingDebugLoggingConfig{}, cache_config_, sp_config, true);
    }

    static PPExecutionPlan makePlan(const PPBatchStreamProcessor&           processor,
                                    const StreamGroups&                     groups,
                                    std::unordered_map<RequestId, int32_t>& history_lengths) {
        TensorHolder holder;
        auto         plan = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
        EXPECT_TRUE(plan.ok()) << plan.status().ToString();
        return std::move(plan.value());
    }

    static PPExecutionPlan makePlan(const PPBatchStreamProcessor& processor, const StreamGroups& groups) {
        std::unordered_map<RequestId, int32_t> history_lengths;
        return makePlan(processor, groups, history_lengths);
    }

    /** Each request needs an output and error slot, including requests that fail during preparation. */
    static PPExecutionResult makeResult(const std::vector<int64_t>& ids) {
        PPExecutionResult result;
        result.request_ids = torch::tensor(ids, torch::kInt64);
        result.request_errors.assign(ids.size(), ErrorInfo::OkStatus());
        result.request_outputs.resize(ids.size());
        return result;
    }

    static PPExecutionResult makeResult(const PPSamplingPlan& plan) {
        return makeResult(tensorToVector<int64_t>(plan.request_ids));
    }

    static PPSamplingOutput makeSamplingOutput(const PPSamplingPlan& plan) {
        PPSamplingOutput output;
        output.request_errors.assign(plan.request_ids.numel(), ErrorInfo::OkStatus());
        return output;
    }

    static void commitTokens(const GenerateStreamPtr& stream, const torch::Tensor& tokens) {
        stream->update({tokens, static_cast<int>(tokens.size(1))}, false);
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
        cache_config_            = test::makeSimpleMhaCacheConfig(1, 16, 4, DataType::TYPE_FP16, 2, 2);
        resources_.cache_manager = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(resources_.cache_manager->init());
    }

    void initGrammar() {
        setVocab(4);
        GrammarConfig grammar_config;
        grammar_config.num_workers               = 1;
        LogitsProcessorFactory::grammarBackend() = XGrammarBackend::create(
            xgrammar_impl::serializeTokenizerInfo(
                {"a", "b", "c", "<eos>"},
                R"({"vocab_size":4,"stop_token_ids":[3],"vocab_type":"RAW","add_prefix_space":false})"),
            grammar_config);
        ASSERT_NE(LogitsProcessorFactory::grammarBackend(), nullptr);
    }

    std::unordered_map<RequestId, torch::Tensor> dspark_q_;
    ModelConfig                      model_config_{};
    CacheConfig                      cache_config_;
    ResourceContext                  resources_;
    std::shared_ptr<XGrammarBackend> previous_backend_;
};

/** Plans map request configuration onto sequence rows. */
TEST_F(PPBatchStreamProcessorTest, ExecutionPlanKeepsRequestAndSequenceOffsets) {
    initCache();
    auto multi_config                 = config(2);
    multi_config.random_seed          = 123;
    multi_config.top_k                = 7;
    multi_config.top_p                = 0.8f;
    multi_config.temperature          = 0.7f;
    multi_config.repetition_penalty   = 1.5f;
    multi_config.presence_penalty     = 0.2f;
    multi_config.frequency_penalty    = 0.3f;
    multi_config.no_repeat_ngram_size = 2;
    multi_config.return_logits        = true;
    auto multi                        = makeStream(101, {1, 2}, multi_config);
    commitTokens(multi, intTensor({10, 63}).reshape({2, 1}));
    auto single = makeStream(202, {3, 4, 5, 6}, config(0));
    for (const auto& stream : {multi, single}) {
        stream->fakeInitKVBlock(1);
    }

    auto                                   processor = makeProcessor();
    const StreamGroups                     groups({single, multi});
    TensorHolder                           holder;
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto gathered = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
    ASSERT_TRUE(gathered.ok());
    EXPECT_TRUE(gathered->is_decode);
    EXPECT_FALSE(gathered->model_input.is_target_verify);
    EXPECT_EQ(tensorToVector<int32_t>(gathered->model_input.combo_tokens), (std::vector<int32_t>{10, 63, 3, 4, 5, 6}));
    EXPECT_EQ(tensorToVector<int32_t>(gathered->model_input.input_lengths), (std::vector<int32_t>{2, 2, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(gathered->model_input.sequence_lengths), (std::vector<int32_t>{2, 2}));
    EXPECT_EQ(tensorToVector<int32_t>(gathered->model_input.prefix_lengths), (std::vector<int32_t>{0}));
    EXPECT_EQ(tensorToVector<int32_t>(gathered->model_input.lm_output_indexes), (std::vector<int32_t>{0, 1, 5}));
    const auto& plan = gathered->sampling_plan;
    EXPECT_EQ(tensorToVector<int64_t>(plan.request_ids), (std::vector<int64_t>{101, 202}));
    EXPECT_EQ(plan.num_return_sequences, (std::vector<int32_t>{2, 0}));
    ASSERT_EQ(plan.random_seeds.size(), 2);
    EXPECT_EQ(plan.random_seeds[0], std::optional<int>(123));
    EXPECT_FALSE(plan.random_seeds[1].has_value());
    EXPECT_EQ(plan.token_ids.sizes().vec(), (std::vector<int64_t>{10}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.token_ids), (std::vector<int32_t>{1, 2, 10, 1, 2, 63, 3, 4, 5, 6}));
    EXPECT_EQ(tensorToVector<int64_t>(plan.token_offsets), (std::vector<int64_t>{0, 3, 6, 10}));
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
    EXPECT_EQ(history_lengths.at(101), 3);
    EXPECT_EQ(history_lengths.at(202), 4);
    ASSERT_EQ(gathered->output_config.requests.size(), 2u);
    EXPECT_TRUE(gathered->output_config.return_logits);
    EXPECT_TRUE(gathered->output_config.requests[0].return_logits);
    EXPECT_FALSE(gathered->output_config.requests[1].return_logits);
    EXPECT_FALSE(gathered->draft_next_position_ids.defined());

    const auto next = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
    ASSERT_TRUE(next.ok());
    EXPECT_EQ(next->sampling_plan.token_ids.numel(), 0);
    EXPECT_EQ(tensorToVector<int64_t>(next->sampling_plan.token_offsets), (std::vector<int64_t>{0, 0, 0, 0}));
}

TEST_F(PPBatchStreamProcessorTest, EmptyExecutionPlanPreservesControlFieldsWithoutAdvancingHistory) {
    auto processor = makeProcessor();
    for (bool shutdown : {false, true}) {
        SCOPED_TRACE(shutdown);
        TensorHolder                           holder;
        std::unordered_map<RequestId, int32_t> history_lengths{{101, 3}};
        const auto plan = processor.gatherExecutionPlan({}, {101}, shutdown, history_lengths, holder);
        ASSERT_TRUE(plan.ok());
        EXPECT_TRUE(plan->model_input.skip_run);
        EXPECT_EQ(plan->model_input.shutdown, shutdown);
        EXPECT_FALSE(plan->is_decode);
        EXPECT_EQ(plan->finished_request_ids, (std::vector<RequestId>{101}));
        EXPECT_FALSE(plan->sampling_plan.request_ids.defined());
        EXPECT_FALSE(plan->sampling_plan.token_ids.defined());
        EXPECT_FALSE(plan->sampling_plan.token_offsets.defined());
        EXPECT_TRUE(plan->output_config.requests.empty());
        EXPECT_FALSE(plan->draft_next_position_ids.defined());
        EXPECT_EQ(history_lengths.at(101), 3);
    }
}

TEST_F(PPBatchStreamProcessorTest, SkippedSamplingKeepsRequestMetadataWithoutAdvancingHistory) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (bool fake : {false, true}) {
            SCOPED_TRACE(testing::Message() << "type=" << type << ", fake=" << fake);
            auto cfg           = config();
            cfg.max_new_tokens = fake ? 8 : 0;
            cfg.random_seed    = 123;
            auto stream        = makeStream(101, {1, 2}, cfg);
            stream->setIsFakeStream(fake);
            auto                                   processor = makeProcessor(type);
            std::unordered_map<RequestId, int32_t> history_lengths{{202, 5}};
            const auto                             plan = makePlan(processor, StreamGroups({stream}), history_lengths);
            EXPECT_FALSE(plan.model_input.skip_run);
            EXPECT_EQ(plan.model_input.is_fake_stream, fake);
            EXPECT_EQ(plan.model_input.skip_lm_head, !fake);
            EXPECT_EQ(tensorToVector<RequestId>(plan.sampling_plan.request_ids), (std::vector<RequestId>{101}));
            EXPECT_EQ(tensorToVector<int32_t>(plan.sampling_plan.sequence_lengths), (std::vector<int32_t>{2}));
            EXPECT_EQ(plan.sampling_plan.random_seeds, (std::vector<std::optional<int>>{123}));
            EXPECT_FALSE(plan.sampling_plan.token_ids.defined());
            EXPECT_FALSE(plan.sampling_plan.token_offsets.defined());
            EXPECT_EQ(plan.sampling_plan.spec_do_sample.defined(), type != SP_TYPE_NONE);
            EXPECT_EQ(plan.output_config.requests.size(), 1u);
            EXPECT_EQ(history_lengths.size(), 1u);
            EXPECT_EQ(history_lengths.at(202), 5);
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, SamplingPlanNormalizesGreedyAndCarriesProcessorConfig) {
    auto cfg                          = config(2);
    cfg.do_sample                     = false;
    cfg.top_k                         = 7;
    cfg.top_p                         = 0.4f;
    cfg.temperature                   = 0.3f;
    cfg.combo_token_size              = 2;
    cfg.banned_combo_token_ids        = {{20, 21}};
    cfg.end_think_token_ids           = {30, 31};
    cfg.enable_cross_sequence_ban     = true;
    cfg.cross_seq_diverge_start_combo = 3;
    auto       stream                 = makeStream(101, {1, 2}, cfg);
    auto       processor              = makeProcessor();
    const auto plan                   = makePlan(processor, StreamGroups({stream})).sampling_plan;
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
    auto greedy                = config();
    greedy.top_k               = 1;
    auto stochastic            = config();
    stochastic.top_k           = 4;
    stochastic.force_sp_accept = true;
    auto disabled              = stochastic;
    disabled.do_sample         = false;
    auto disabled_top1         = greedy;
    disabled_top1.do_sample    = false;
    auto first                 = makeStream(101, {1, 2}, greedy);
    auto second                = makeStream(202, {3}, stochastic);
    auto third                 = makeStream(303, {4}, disabled);
    auto fourth                = makeStream(404, {5}, disabled_top1);
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(static_cast<int>(type));
        auto       processor = makeProcessor(type);
        const auto plan      = makePlan(processor, StreamGroups({first, second, third, fourth})).sampling_plan;
        EXPECT_EQ(tensorToVector<bool>(plan.spec_do_sample), (std::vector<bool>{false, true, false, false}));
        EXPECT_EQ(tensorToVector<bool>(plan.force_sp_accept), (std::vector<bool>{false, true, true, false}));
    }
}

TEST_F(PPBatchStreamProcessorTest, DSparkVerifyNormalizesTemperatureForTargetAndDraft) {
    struct TemperatureCase {
        float temperature;
        bool  do_sample;
        int   top_k;
        float target_temperature;
        float draft_temperature;
    };
    const float                        infinity = std::numeric_limits<float>::infinity();
    const std::vector<TemperatureCase> cases{
        {infinity, true, 4, 1.0f, 1.0f},
        {-infinity, true, 4, 1.0f, 1.0f},
        {std::numeric_limits<float>::quiet_NaN(), true, 4, 1.0f, 1.0f},
        {-1.0f, true, 4, 1.0f, 1.0f},
        {0.0f, true, 4, 0.0f, 1.0e-6f},
        {0.7f, true, 4, 0.7f, 0.7f},
        {infinity, false, 4, 1.0f, 1.0e-6f},
        {infinity, true, 1, 1.0f, 1.0e-6f},
        {0.7f, true, 1, 0.7f, 1.0e-6f},
    };
    std::list<GenerateStreamPtr> streams;
    for (const auto& test_case : cases) {
        auto cfg        = config();
        cfg.temperature = test_case.temperature;
        cfg.do_sample   = test_case.do_sample;
        cfg.top_k       = test_case.top_k;
        auto stream     = makeStream(101 + streams.size(), {1, 2}, cfg);
        commitTokens(stream, intTensor({0}).reshape({1, 1}));
        setProposals(stream, {0, 0, 0});
        streams.push_back(stream);
    }
    auto       processor = makeProcessor(SP_TYPE_DSPARK, RoleType::DECODE, 2);
    const auto plan      = makePlan(processor, StreamGroups(streams));
    ASSERT_TRUE(plan.model_input.is_target_verify);
    auto                                   sampling_output = makeSamplingOutput(plan.sampling_plan);
    SamplingStates                         states;
    SamplerOutput                          draft;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);
    const auto   inputs = processor.gatherSamplerInputs(plan.sampling_plan,
                                                      plan.output_config,
                                                      torch::zeros({static_cast<int64_t>(cases.size()) * 3, 64}),
                                                      states,
                                                      true,
                                                      2,
                                                      plan.model_input.combo_tokens);
    TensorHolder holder;
    const auto   draft_temperature = processor.prepareDSparkDraftTemperature(plan.sampling_plan, torch::kCPU, holder);
    auto         stream_it         = streams.begin();
    for (size_t i = 0; i < cases.size(); ++i, ++stream_it) {
        SCOPED_TRACE(i);
        EXPECT_FLOAT_EQ(plan.sampling_plan.temperature[i].item<float>(), cases[i].target_temperature);
        for (int row = 0; row < 3; ++row) {
            EXPECT_FLOAT_EQ(inputs.temperature[i * 3 + row].item<float>(), cases[i].target_temperature);
        }
        EXPECT_FLOAT_EQ(draft_temperature[i].item<float>(), cases[i].draft_temperature);
        const auto configured_temperature = (*stream_it)->generateConfig()->temperature;
        if (std::isnan(cases[i].temperature)) {
            EXPECT_TRUE(std::isnan(configured_temperature));
        } else {
            EXPECT_FLOAT_EQ(configured_temperature, cases[i].temperature);
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, TemperatureNormalizationPreservesPrefillAndOtherModes) {
    const float infinity = std::numeric_limits<float>::infinity();
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (bool decode : {false, true}) {
            if (type == SP_TYPE_DSPARK && decode) {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "type=" << type << ", decode=" << decode);
            auto cfg        = config();
            cfg.temperature = infinity;
            cfg.do_sample   = true;
            cfg.top_k       = 4;
            auto stream     = makeStream(101, {1, 2}, cfg);
            if (decode) {
                commitTokens(stream, intTensor({0}).reshape({1, 1}));
                setProposals(stream, {0, 0, 0});
            }
            auto       processor = makeProcessor(type, RoleType::PDFUSION, 2);
            const auto plan      = makePlan(processor, StreamGroups({stream}));
            EXPECT_EQ(plan.is_decode, decode);
            EXPECT_EQ(plan.sampling_plan.temperature.item<float>(), infinity);
            if (type == SP_TYPE_DSPARK) {
                TensorHolder holder;
                const auto   draft_temperature =
                    processor.prepareDSparkDraftTemperature(plan.sampling_plan, torch::kCPU, holder);
                EXPECT_FLOAT_EQ(draft_temperature.item<float>(), 1.0f);
            }
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, OutputConfigAggregatesFlagsAndPreservesRequestRequirements) {
    auto first_config                             = config();
    first_config.return_logits                    = true;
    first_config.return_hidden_states             = true;
    first_config.return_all_hidden_states         = true;
    first_config.return_prompt_logits             = true;
    first_config.prompt_logits_top_k              = 2;
    first_config.prompt_logits_start              = 1;
    first_config.prompt_logits_end                = 3;
    first_config.return_target_logprob            = false;
    auto second_config                            = config();
    second_config.return_softmax_probs            = true;
    second_config.return_cum_log_probs            = true;
    second_config.return_all_probs                = ReturnAllProbsMode::DEFAULT;
    auto                                   first  = makeStream(101, {1, 2, 3}, first_config);
    auto                                   second = makeStream(202, {4, 5}, second_config);
    StreamGroups                           groups({first, second});
    auto                                   processor = makeProcessor();
    std::unordered_map<RequestId, int32_t> history_lengths;
    const auto                             output = makePlan(processor, groups, history_lengths).output_config;
    EXPECT_TRUE(output.return_logits);
    EXPECT_TRUE(output.return_hidden_states);
    EXPECT_TRUE(output.return_all_hidden_states);
    EXPECT_TRUE(output.return_softmax_probs);
    EXPECT_TRUE(output.return_cum_log_probs);
    EXPECT_FALSE(output.calculate_loss);
    EXPECT_EQ(output.return_all_probs, ReturnAllProbsMode::DEFAULT);
    ASSERT_EQ(output.requests.size(), 2);
    EXPECT_TRUE(output.requests[0].return_logits);
    EXPECT_TRUE(output.requests[0].return_hidden_states);
    EXPECT_TRUE(output.requests[0].return_all_hidden_states);
    EXPECT_FALSE(output.requests[0].return_softmax_probs);
    EXPECT_FALSE(output.requests[0].return_cum_log_probs);
    EXPECT_FALSE(output.requests[0].calculate_loss);
    EXPECT_FALSE(output.requests[0].return_all_probs);
    EXPECT_FALSE(output.requests[1].return_logits);
    EXPECT_FALSE(output.requests[1].return_hidden_states);
    EXPECT_FALSE(output.requests[1].return_all_hidden_states);
    EXPECT_TRUE(output.requests[1].return_softmax_probs);
    EXPECT_TRUE(output.requests[1].return_cum_log_probs);
    EXPECT_FALSE(output.requests[1].calculate_loss);
    EXPECT_TRUE(output.requests[1].return_all_probs);
    const auto& prompt = output.requests[0].prompt_logits;
    EXPECT_TRUE(prompt.enabled);
    EXPECT_EQ(prompt.top_k, 2);
    EXPECT_EQ(prompt.start, 1);
    EXPECT_EQ(prompt.end, 3);
    EXPECT_FALSE(prompt.return_target_logprob);
    EXPECT_FALSE(output.requests[1].prompt_logits.enabled);
    second->setReturnAllProbs(ReturnAllProbsMode::ORIGINAL);
    EXPECT_EQ(makePlan(processor, groups, history_lengths).output_config.return_all_probs,
              ReturnAllProbsMode::ORIGINAL);
    second->setReturnAllProbs(ReturnAllProbsMode::NONE);
    EXPECT_EQ(makePlan(processor, groups, history_lengths).output_config.return_all_probs, ReturnAllProbsMode::NONE);
}

TEST_F(PPBatchStreamProcessorTest, SpeculativeOutputConfigDisablesPerStepOutputs) {
    auto cfg                     = config();
    cfg.return_logits            = true;
    cfg.return_hidden_states     = true;
    cfg.return_all_hidden_states = true;
    cfg.return_cum_log_probs     = true;
    cfg.return_all_probs         = ReturnAllProbsMode::DEFAULT;
    auto stream                  = makeStream(101, {1, 2}, cfg);
    for (auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        auto       processor = makeProcessor(type);
        const auto output    = makePlan(processor, StreamGroups({stream})).output_config;
        EXPECT_FALSE(output.return_logits);
        EXPECT_FALSE(output.return_hidden_states);
        EXPECT_FALSE(output.return_all_hidden_states);
        EXPECT_FALSE(output.return_cum_log_probs);
        EXPECT_EQ(output.return_all_probs, ReturnAllProbsMode::NONE);
        EXPECT_FALSE(output.return_softmax_probs);
        EXPECT_FALSE(output.calculate_loss);
        ASSERT_EQ(output.requests.size(), 1u);
        EXPECT_FALSE(output.requests[0].return_logits);
        EXPECT_FALSE(output.requests[0].return_hidden_states);
        EXPECT_FALSE(output.requests[0].return_all_hidden_states);
        EXPECT_FALSE(output.requests[0].return_cum_log_probs);
        EXPECT_FALSE(output.requests[0].return_all_probs);
        EXPECT_FALSE(output.requests[0].prompt_logits.enabled);
    }
}

/** Sampling preparation retains request state and selects the Q associated with each proposal. */
TEST_F(PPBatchStreamProcessorTest, IncrementalHistorySurvivesBatchChangesAndBuildsSamplerInputs) {
    auto cfg                                         = config(2);
    cfg.random_seed                                  = 123;
    cfg.repetition_penalty                           = 1.5f;
    cfg.presence_penalty                             = 0.2f;
    cfg.frequency_penalty                            = 0.3f;
    cfg.no_repeat_ngram_size                         = 2;
    auto                                   multi     = makeStream(101, {1, 2}, cfg);
    auto                                   paused    = makeStream(202, {3, 4, 5});
    auto                                   processor = makeProcessor();
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto  execution_plan = makePlan(processor, StreamGroups({multi, paused}), history_lengths);
    auto& plan           = execution_plan.sampling_plan;
    EXPECT_EQ(history_lengths.at(101), 2);
    EXPECT_EQ(history_lengths.at(202), 3);
    SamplingStates                         states;
    SamplerOutput                          draft;
    speculative::SpeculativeSamplingParams params;
    auto                                   sampling_output = makeSamplingOutput(plan);
    processor.prepareSampling(execution_plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);
    const auto generator                      = states.at(101).sampling.generator;
    auto       recorder                       = std::make_shared<RecordingLogitsProcessor>();
    states.at(101).sampling.logits_processors = {recorder};

    commitTokens(multi, intTensor({6, 7, 8, 9}).reshape({2, 2}));
    auto added     = makeStream(303, {10});
    execution_plan = makePlan(processor, StreamGroups({added, multi}), history_lengths);
    EXPECT_EQ(history_lengths.at(101), 4);
    EXPECT_EQ(history_lengths.at(202), 3);
    EXPECT_EQ(history_lengths.at(303), 1);
    EXPECT_EQ(tensorToVector<RequestId>(plan.request_ids), (std::vector<RequestId>{101, 303}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.token_ids), (std::vector<int32_t>{6, 7, 8, 9, 10}));
    EXPECT_EQ(tensorToVector<int64_t>(plan.token_offsets), (std::vector<int64_t>{0, 2, 4, 5}));
    sampling_output = makeSamplingOutput(plan);
    processor.prepareSampling(execution_plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);
    EXPECT_EQ(states.at(101).token_history, (std::vector<std::vector<int32_t>>{{1, 2, 6, 7}, {1, 2, 8, 9}}));
    EXPECT_EQ(states.at(202).token_history, (std::vector<std::vector<int32_t>>{{3, 4, 5}}));
    EXPECT_EQ(states.at(101).sampling.generator, generator);
    EXPECT_EQ(recorder->updates, 0);

    commitTokens(paused, intTensor({11}).reshape({1, 1}));
    const StreamGroups reordered({added, paused, multi});
    execution_plan = makePlan(processor, reordered, history_lengths);
    EXPECT_EQ(history_lengths.at(202), 4);
    EXPECT_EQ(tensorToVector<RequestId>(plan.request_ids), (std::vector<RequestId>{202, 101, 303}));
    EXPECT_EQ(tensorToVector<int32_t>(plan.token_ids), (std::vector<int32_t>{11}));
    EXPECT_EQ(tensorToVector<int64_t>(plan.token_offsets), (std::vector<int64_t>{0, 1, 1, 1, 1}));
    sampling_output = makeSamplingOutput(plan);
    processor.prepareSampling(execution_plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);

    const auto inputs = processor.gatherSamplerInputs(plan, {}, torch::zeros({4, 64}), states);
    EXPECT_EQ(tensorToVector<int32_t>(inputs.sequence_lengths), (std::vector<int32_t>{4, 4, 4, 1}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[0].narrow(0, 0, 4)), (std::vector<int32_t>{3, 4, 5, 11}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[1].narrow(0, 0, 4)), (std::vector<int32_t>{1, 2, 6, 7}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.token_ids[2].narrow(0, 0, 4)), (std::vector<int32_t>{1, 2, 8, 9}));
    EXPECT_EQ(inputs.token_ids[3][0].item<int32_t>(), 10);
    EXPECT_EQ(tensorToVector<float>(inputs.repetition_penalty), (std::vector<float>{1, 1.5f, 1.5f, 1}));
    EXPECT_EQ(tensorToVector<float>(inputs.presence_penalty), (std::vector<float>{0, .2f, .2f, 0}));
    EXPECT_EQ(tensorToVector<float>(inputs.frequency_penalty), (std::vector<float>{0, .3f, .3f, 0}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.no_repeat_ngram_size), (std::vector<int32_t>{0, 2, 2, 0}));

    execution_plan = makePlan(processor, reordered, history_lengths);
    EXPECT_TRUE(plan.token_ids.defined());
    EXPECT_EQ(plan.token_ids.numel(), 0);
    EXPECT_EQ(tensorToVector<int64_t>(plan.token_offsets), (std::vector<int64_t>{0, 0, 0, 0, 0}));
    sampling_output = makeSamplingOutput(plan);
    processor.prepareSampling(execution_plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);
    EXPECT_EQ(states.at(202).token_history, (std::vector<std::vector<int32_t>>{{3, 4, 5, 11}}));
    EXPECT_EQ(recorder->updates, 0);
}

TEST_F(PPBatchStreamProcessorTest, SamplingStatesPreserveExistingRequestWhenAddingRequests) {
    auto cfg                                         = config(2);
    cfg.random_seed                                  = 123;
    auto                                   existing  = makeStream(101, {1, 2}, cfg);
    auto                                   processor = makeProcessor();
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto           execution_plan = makePlan(processor, StreamGroups({existing}), history_lengths);
    auto&          plan           = execution_plan.sampling_plan;
    auto           sampling_output         = makeSamplingOutput(plan);
    SamplingStates states;
    SamplerOutput  draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    ASSERT_TRUE(sampling_output.request_errors[0].ok());
    auto& state = states.at(101).sampling;
    ASSERT_TRUE(state.generator.defined());
    EXPECT_EQ(state.generator.current_seed(), 123);
    EXPECT_EQ(tensorToVector<float>(state.cum_log_probs), (std::vector<float>{0, 0}));
    const auto generator  = state.generator;
    const auto processors = state.logits_processors;
    ASSERT_EQ(processors.size(), 1);
    torch::rand({4}, generator, std::nullopt, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    const auto rng_state = generator.get_state().clone();
    state.cum_log_probs.copy_(torch::tensor({-2.0f, -4.0f}));

    auto added     = makeStream(202, {3}, config(0));
    execution_plan = makePlan(processor, StreamGroups({existing, added}), history_lengths);
    sampling_output         = makeSamplingOutput(plan);
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    ASSERT_EQ(states.size(), 2);
    EXPECT_TRUE(sampling_output.request_errors[0].ok());
    EXPECT_TRUE(sampling_output.request_errors[1].ok());
    EXPECT_EQ(states.at(101).sampling.generator, generator);
    EXPECT_TRUE(torch::equal(states.at(101).sampling.generator.get_state(), rng_state));
    EXPECT_EQ(states.at(101).sampling.logits_processors, processors);
    EXPECT_EQ(tensorToVector<float>(states.at(101).sampling.cum_log_probs), (std::vector<float>{-2, -4}));
    EXPECT_FALSE(states.at(202).sampling.generator.defined());
    EXPECT_TRUE(states.at(202).sampling.logits_processors.empty());
    EXPECT_EQ(tensorToVector<float>(states.at(202).sampling.cum_log_probs), (std::vector<float>{0}));
}

TEST_F(PPBatchStreamProcessorTest, SamplingStatesReplayPdAnchorAtSequenceOffsetOnlyOnce) {
    initGrammar();
    auto processor = makeProcessor();
    auto existing  = makeStream(101, {3, 3}, config(2));
    commitTokens(existing, intTensor({0, 1}).reshape({2, 1}));
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto           execution_plan = makePlan(processor, StreamGroups({existing}), history_lengths);
    auto&          plan           = execution_plan.sampling_plan;
    auto           sampling_output         = makeSamplingOutput(plan);
    SamplingStates states;
    SamplerOutput  draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);

    auto cfg   = config();
    cfg.regex  = "abc";
    auto added = makeStream(202, {3}, cfg);
    commitTokens(added, intTensor({0}).reshape({1, 1}));
    execution_plan = makePlan(processor, StreamGroups({existing, added}), history_lengths);
    sampling_output         = makeSamplingOutput(plan);
    ASSERT_EQ(plan.logits_processor_configs[1].grammar_type, "regex");
    EXPECT_EQ(plan.logits_processor_configs[1].grammar_value, "abc");
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    ASSERT_TRUE(sampling_output.request_errors[1].ok());
    const auto processors = states.at(202).sampling.logits_processors;
    ASSERT_EQ(processors.size(), 1);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(1));
    execution_plan = makePlan(processor, StreamGroups({existing, added}), history_lengths);
    sampling_output         = makeSamplingOutput(plan);
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    EXPECT_TRUE(sampling_output.request_errors[1].ok());
    EXPECT_EQ(states.at(202).sampling.logits_processors, processors);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(1));
}

TEST_F(PPBatchStreamProcessorTest, SamplingStatesDoNotReplayThePrefillPromptAsGeneratedTokens) {
    initGrammar();
    auto cfg                                              = config();
    cfg.regex                                             = "abc";
    auto                                   stream         = makeStream(101, {2, 1}, cfg);
    auto                                   processor      = makeProcessor();
    auto                                   execution_plan = makePlan(processor, StreamGroups({stream}));
    const auto&                            plan           = execution_plan.sampling_plan;
    auto                                   sampling_output         = makeSamplingOutput(plan);
    SamplingStates                         states;
    SamplerOutput                          draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    EXPECT_TRUE(sampling_output.request_errors[0].ok());
    EXPECT_FALSE(draft_sampler_output.token_ids.defined());
    EXPECT_FALSE(draft_sampler_output.all_probs.defined());
    EXPECT_FALSE(params.do_sample.defined());
    EXPECT_FALSE(params.force_accept.defined());
    EXPECT_TRUE(params.generators.empty());
    const auto& processors = states.at(101).sampling.logits_processors;
    ASSERT_EQ(processors.size(), 1);
    EXPECT_EQ(processors[0]->committedOutputLen(), std::optional<int64_t>(0));
}

TEST_F(PPBatchStreamProcessorTest, GrammarPrefetchKeepsStableUniqueKeysForNewRequestsOnly) {
    initGrammar();
    auto processor = makeProcessor();

    PPSamplingPlan plan;
    plan.request_ids = torch::tensor({101, 202, 303, 404}, torch::kInt64);
    plan.logits_processor_configs.resize(4);
    plan.logits_processor_configs[0].grammar_type  = "regex";
    plan.logits_processor_configs[0].grammar_value = "a";
    plan.logits_processor_configs[1].grammar_type  = "regex";
    plan.logits_processor_configs[1].grammar_value = "b";
    plan.logits_processor_configs[2].grammar_type  = "regex";
    plan.logits_processor_configs[2].grammar_value = "b";
    plan.logits_processor_configs[3].grammar_type  = "unsupported";
    plan.logits_processor_configs[3].grammar_value = "ignored";

    SamplingStates states;
    states.try_emplace(101);
    auto context = processor.prepareLogitsProcessorsAsync(plan, states);

    ASSERT_EQ(context.handles.size(), 1u);
    EXPECT_EQ(context.find({"regex", "a"}), nullptr);
    EXPECT_NE(context.find({"regex", "b"}), nullptr);
    EXPECT_EQ(context.find({"unsupported", "ignored"}), nullptr);
}

TEST_F(PPBatchStreamProcessorTest, UnsupportedGrammarIsSkippedByAsyncPreparationAndRejectedByInitialization) {
    auto processor = makeProcessor();
    auto stream    = makeStream(101, {1, 2});

    auto execution_plan                                                   = makePlan(processor, StreamGroups({stream}));
    execution_plan.sampling_plan.logits_processor_configs[0].grammar_type = "unsupported";
    execution_plan.sampling_plan.logits_processor_configs[0].grammar_value = "ignored";

    SamplingStates states;
    auto           context = processor.prepareLogitsProcessorsAsync(execution_plan.sampling_plan, states);
    EXPECT_TRUE(context.handles.empty());

    auto                                   sampling_output = makeSamplingOutput(execution_plan.sampling_plan);
    SamplerOutput                          draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params,
                              &context);

    ASSERT_EQ(sampling_output.request_errors.size(), 1u);
    EXPECT_EQ(sampling_output.request_errors[0].code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(sampling_output.request_errors[0].ToString().find("unsupported grammar type"), std::string::npos);
}

TEST_F(PPBatchStreamProcessorTest, SamplingStateInitializationFailureIsRequestLocal) {
    initGrammar();
    auto cfg                   = config();
    cfg.regex                  = "abc";
    cfg.random_seed            = 456;
    auto        failed         = makeStream(101, {3}, cfg);
    auto        healthy        = makeStream(202, {1, 2});
    auto        processor      = makeProcessor();
    auto        execution_plan = makePlan(processor, StreamGroups({failed, healthy}));
    const auto& plan           = execution_plan.sampling_plan;
    /** A tail with no grammar backend must return an error while retaining rows for ordinary sampling. */
    LogitsProcessorFactory::grammarBackend().reset();
    auto                                   sampling_output = makeSamplingOutput(plan);
    SamplingStates                         states;
    SamplerOutput                          draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    EXPECT_EQ(sampling_output.request_errors[0].code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(sampling_output.request_errors[1].ok());
    ASSERT_EQ(states.size(), 2);
    EXPECT_TRUE(states.at(101).sampling.logits_processors.empty());
    EXPECT_EQ(states.at(101).sampling.generator.current_seed(), 456);
    EXPECT_EQ(tensorToVector<float>(states.at(101).sampling.cum_log_probs), (std::vector<float>{0}));
    const auto inputs = processor.gatherSamplerInputs(plan, PPOutputConfig{}, torch::zeros({2, 4}), states);
    EXPECT_EQ(inputs.batch_size, 2);
    EXPECT_EQ(inputs.generator[0], states.at(101).sampling.generator);
    EXPECT_FALSE(inputs.generator[1].defined());
}

TEST_F(PPBatchStreamProcessorTest, DSparkQFollowsRequestOrderAcrossDenseAndHandoffBatches) {
    setVocab(16);
    auto processor                = makeProcessor(SP_TYPE_DSPARK, RoleType::DECODE, 2);
    auto first_config             = config();
    first_config.random_seed      = 123;
    first_config.top_k            = 4;
    auto second_config            = config();
    second_config.top_k           = 1;
    second_config.force_sp_accept = true;
    auto third_config             = config();
    third_config.random_seed      = 456;
    third_config.top_k            = 4;
    auto first                    = makeStream(101, {1, 2}, first_config);
    auto second                   = makeStream(202, {3}, second_config);
    auto third                    = makeStream(303, {4}, third_config);
    for (const auto& stream : {first, second, third}) {
        commitTokens(stream, intTensor({0}).reshape({1, 1}));
        setProposals(stream, {0, 0, 0});
    }

    SamplingStates                               states;
    std::unordered_map<RequestId, torch::Tensor> dspark_q;
    speculative::SpeculativeSamplingParams       verify_params;
    std::unordered_map<RequestId, int32_t>       history_lengths;
    auto          plan     = makePlan(processor, StreamGroups({first, second, third}), history_lengths);
    auto&         sampling = plan.sampling_plan;
    auto          sampling_output   = makeSamplingOutput(sampling);
    SamplerOutput draft_sampler_output;
    processor.prepareSampling(plan,
                              dspark_q,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              verify_params);
    EXPECT_FALSE(draft_sampler_output.token_ids_are_point_mass);
    ASSERT_TRUE(draft_sampler_output.all_probs.defined());
    EXPECT_TRUE(draft_sampler_output.all_probs.is_cuda());
    EXPECT_EQ(draft_sampler_output.all_probs.scalar_type(), torch::kFloat32);
    EXPECT_EQ(draft_sampler_output.all_probs.sizes().vec(), (std::vector<int64_t>{3, 2, 16}));
    EXPECT_TRUE(torch::equal(
        draft_sampler_output.all_probs.select(2, 0),
        torch::ones({draft_sampler_output.token_ids.size(0), 2}, draft_sampler_output.all_probs.options())));
    EXPECT_EQ(draft_sampler_output.all_probs.sum().item<float>(), draft_sampler_output.token_ids.numel());
    EXPECT_TRUE(torch::equal(draft_sampler_output.token_ids, torch::zeros({3, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(verify_params.do_sample, sampling.spec_do_sample));
    EXPECT_TRUE(torch::equal(verify_params.force_accept, sampling.force_sp_accept));
    EXPECT_EQ(tensorToVector<bool>(verify_params.do_sample), (std::vector<bool>{true, false, true}));
    EXPECT_EQ(tensorToVector<bool>(verify_params.force_accept), (std::vector<bool>{false, true, false}));
    ASSERT_EQ(verify_params.generators.size(), 3u);
    ASSERT_TRUE(verify_params.generators[0].defined());
    EXPECT_EQ(verify_params.generators[0].current_seed(), 123);
    EXPECT_FALSE(verify_params.generators[1].defined());
    ASSERT_TRUE(verify_params.generators[2].defined());
    EXPECT_EQ(verify_params.generators[2].current_seed(), 456);
    ASSERT_EQ(states.size(), 3u);
    for (const auto& error : sampling_output.request_errors) {
        EXPECT_TRUE(error.ok());
    }

    const auto initial_storage = draft_sampler_output.all_probs.data_ptr<float>();
    const auto q = torch::zeros({3, 2, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    for (int64_t row = 0; row < 3; ++row) {
        q[row].select(1, row).fill_(1);
        dspark_q.emplace(101 * (row + 1), q[row]);
    }

    /** A request from another inflight batch remains cached while these requests change order. */
    setProposals(third, {0, 2, 2});
    plan                 = makePlan(processor, StreamGroups({third, first}), history_lengths);
    sampling_output               = makeSamplingOutput(sampling);
    verify_params        = speculative::SpeculativeSamplingParams{};
    draft_sampler_output = SamplerOutput{};
    processor.prepareSampling(plan,
                              dspark_q,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              verify_params);
    EXPECT_TRUE(torch::equal(draft_sampler_output.all_probs, torch::stack({q[2], q[0]})));
    EXPECT_TRUE(draft_sampler_output.all_probs.is_contiguous());
    EXPECT_EQ(draft_sampler_output.all_probs.data_ptr<float>(), initial_storage);
    EXPECT_TRUE(torch::equal(draft_sampler_output.token_ids, intTensor({2, 2, 0, 0}).reshape({2, 2})));
    EXPECT_TRUE(draft_sampler_output.token_ids.is_contiguous());
    EXPECT_FALSE(draft_sampler_output.token_ids_are_point_mass);
    ASSERT_EQ(verify_params.generators.size(), 2u);
    EXPECT_EQ(verify_params.generators[0], states.at(303).sampling.generator);
    EXPECT_EQ(verify_params.generators[1], states.at(101).sampling.generator);
    EXPECT_EQ(dspark_q.size(), 1u);
    EXPECT_TRUE(torch::equal(dspark_q.at(202), q[1]));
    dspark_q.emplace(101, q[0]);
    dspark_q.emplace(303, q[2]);

    auto fourth = makeStream(404, {5});
    auto fifth  = makeStream(505, {6});
    for (const auto& stream : {fourth, fifth}) {
        commitTokens(stream, intTensor({0}).reshape({1, 1}));
        setProposals(stream, {0, 0, 0});
    }
    plan                 = makePlan(processor, StreamGroups({fourth, third, fifth, first}), history_lengths);
    sampling_output               = makeSamplingOutput(sampling);
    verify_params        = speculative::SpeculativeSamplingParams{};
    draft_sampler_output = SamplerOutput{};
    processor.prepareSampling(plan,
                              dspark_q,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              verify_params);
    EXPECT_TRUE(torch::equal(draft_sampler_output.token_ids, intTensor({0, 0, 2, 2, 0, 0, 0, 0}).reshape({4, 2})));
    EXPECT_FALSE(draft_sampler_output.token_ids_are_point_mass);
    EXPECT_TRUE(torch::equal(draft_sampler_output.all_probs[1], q[2]));
    EXPECT_TRUE(torch::equal(draft_sampler_output.all_probs[3], q[0]));
    EXPECT_EQ(draft_sampler_output.all_probs[0].select(1, 0).sum().item<float>(), 2);
    EXPECT_EQ(draft_sampler_output.all_probs[0].sum().item<float>(), 2);
    EXPECT_EQ(draft_sampler_output.all_probs[2].select(1, 0).sum().item<float>(), 2);
    EXPECT_EQ(draft_sampler_output.all_probs[2].sum().item<float>(), 2);
    ASSERT_EQ(verify_params.generators.size(), 4u);
    EXPECT_FALSE(verify_params.generators[0].defined());
    EXPECT_EQ(verify_params.generators[1], states.at(303).sampling.generator);
    EXPECT_FALSE(verify_params.generators[2].defined());
    EXPECT_EQ(verify_params.generators[3], states.at(101).sampling.generator);
    const auto grown_storage = draft_sampler_output.all_probs.data_ptr<float>();
    EXPECT_TRUE(draft_sampler_output.all_probs.is_contiguous());
    draft_sampler_output.all_probs[1].zero_();
    EXPECT_EQ(q[2].sum().item<float>(), 2);
    EXPECT_EQ(dspark_q.count(303), 0u);
    EXPECT_EQ(dspark_q.count(101), 0u);
    EXPECT_TRUE(torch::equal(dspark_q.at(202), q[1]));
    for (const auto& error : sampling_output.request_errors) {
        EXPECT_TRUE(error.ok());
    }

    auto sixth   = makeStream(606, {7});
    auto seventh = makeStream(707, {8});
    for (const auto& stream : {sixth, seventh}) {
        commitTokens(stream, intTensor({0}).reshape({1, 1}));
        setProposals(stream, {0, 0, 0});
    }
    plan                 = makePlan(processor, StreamGroups({sixth, seventh}), history_lengths);
    sampling_output               = makeSamplingOutput(sampling);
    verify_params        = speculative::SpeculativeSamplingParams{};
    draft_sampler_output = SamplerOutput{};
    processor.prepareSampling(plan,
                              dspark_q,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              verify_params);
    EXPECT_FALSE(draft_sampler_output.token_ids_are_point_mass);
    ASSERT_TRUE(draft_sampler_output.all_probs.defined());
    EXPECT_EQ(draft_sampler_output.all_probs.sizes().vec(), (std::vector<int64_t>{2, 2, 16}));
    EXPECT_EQ(draft_sampler_output.all_probs.data_ptr<float>(), grown_storage);
    EXPECT_TRUE(draft_sampler_output.all_probs.is_contiguous());
    EXPECT_TRUE(torch::equal(
        draft_sampler_output.all_probs.select(2, 0),
        torch::ones({draft_sampler_output.token_ids.size(0), 2}, draft_sampler_output.all_probs.options())));
    EXPECT_EQ(draft_sampler_output.all_probs.sum().item<float>(), draft_sampler_output.token_ids.numel());
    EXPECT_TRUE(torch::equal(draft_sampler_output.token_ids, torch::zeros({2, 2}, torch::kInt32)));
    ASSERT_EQ(verify_params.generators.size(), 2u);
    EXPECT_FALSE(verify_params.generators[0].defined());
    EXPECT_FALSE(verify_params.generators[1].defined());
    EXPECT_EQ(dspark_q.size(), 1u);
    for (const auto& error : sampling_output.request_errors) {
        EXPECT_TRUE(error.ok());
    }
}

TEST_F(PPBatchStreamProcessorTest, DSparkPreparesMappedAndHandoffQBeforeSampling) {
    setVocab(4);
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    /** Vocabulary size alone cannot identify token IDs; both reduced and full vocabularies require mapping. */
    for (const bool reduced : {true, false}) {
        SCOPED_TRACE(reduced);
        auto           map       = reduced ? torch::tensor({1, 3}, i64) : torch::tensor({2, 0, 3, 1}, i64);
        auto           processor = makeProcessor(SP_TYPE_DSPARK, RoleType::DECODE, 2);
        auto           existing  = makeStream(101, {1});
        auto           handoff   = makeStream(202, {2});
        SamplingStates states;
        std::unordered_map<RequestId, int32_t> history_lengths;
        auto                                   plan   = makePlan(processor, StreamGroups({existing}), history_lengths);
        auto                                   sampling_output = makeSamplingOutput(plan.sampling_plan);
        SamplerOutput                          draft;
        speculative::SpeculativeSamplingParams params;
        processor.prepareSampling(plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);
        commitTokens(existing, intTensor({1}).reshape({1, 1}));
        commitTokens(handoff, intTensor({2}).reshape({1, 1}));
        setProposals(existing, reduced ? std::vector<int32_t>{1, 1, 3} : std::vector<int32_t>{1, 2, 3});
        setProposals(handoff, {2, 0, 0});
        auto                                         cached = reduced ? torch::tensor({{.25f, .75f}, {.5f, .5f}}, f32) :
                                                                        torch::tensor({{.1f, .2f, .3f, .4f}, {.4f, .3f, .2f, .1f}}, f32);
        const auto                                   snapshot = cached.clone();
        std::unordered_map<RequestId, torch::Tensor> cache{{101, cached}};
        /** Handoff candidates are target token 0, so their Q bypasses the draft vocabulary mapping. */
        plan   = makePlan(processor, StreamGroups({existing, handoff}), history_lengths);
        sampling_output = makeSamplingOutput(plan.sampling_plan);
        processor.prepareSampling(plan, cache, map, states, sampling_output.request_errors, draft, params);
        auto expected_q = torch::zeros({2, 2, 4}, f32);
        expected_q[0].copy_(reduced ? torch::tensor({{0.f, .25f, 0.f, .75f}, {0.f, .5f, 0.f, .5f}}, f32) :
                                      torch::tensor({{.2f, .4f, .1f, .3f}, {.3f, .1f, .4f, .2f}}, f32));
        expected_q[1][0][0] = 1;
        expected_q[1][1][0] = 1;
        EXPECT_TRUE(torch::equal(draft.all_probs, expected_q));
        EXPECT_TRUE(draft.all_probs.is_contiguous());
        EXPECT_TRUE(draft.token_ids.device().is_cpu());
        EXPECT_TRUE(draft.token_ids.is_contiguous());
        EXPECT_FALSE(draft.token_ids_are_point_mass);
        EXPECT_TRUE(torch::equal(cached, snapshot));
        EXPECT_TRUE(cache.empty());
        for (const auto& error : sampling_output.request_errors) {
            EXPECT_TRUE(error.ok());
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, DSparkMissingCachedQReportsErrorAndClearsStaleRow) {
    setVocab(4);
    auto                                   processor = makeProcessor(SP_TYPE_DSPARK, RoleType::DECODE, 1);
    auto                                   first     = makeStream(101, {1});
    auto                                   second    = makeStream(202, {2});
    SamplingStates                         states;
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto                                   plan   = makePlan(processor, StreamGroups({first, second}), history_lengths);
    auto                                   sampling_output = makeSamplingOutput(plan.sampling_plan);
    SamplerOutput                          draft;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(plan, dspark_q_, {}, states, sampling_output.request_errors, draft, params);
    commitTokens(first, intTensor({1}).reshape({1, 1}));
    commitTokens(second, intTensor({2}).reshape({1, 1}));
    setProposals(first, {1, 3});
    setProposals(second, {2, 2});
    plan           = makePlan(processor, StreamGroups({first, second}), history_lengths);
    sampling_output         = makeSamplingOutput(plan.sampling_plan);
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    std::unordered_map<RequestId, torch::Tensor> cache{{101, torch::tensor({{0.f, .25f, 0.f, .75f}}, f32)},
                                                       {202, torch::tensor({{0.f, 0.f, 1.f, 0.f}}, f32)}};
    const auto first_q = cache.at(101);
    const auto second_q = cache.at(202);
    processor.prepareSampling(plan, cache, {}, states, sampling_output.request_errors, draft, params);
    EXPECT_TRUE(torch::equal(draft.all_probs, torch::stack({first_q, second_q})));
    EXPECT_TRUE(cache.empty());

    cache.emplace(101, first_q);
    plan   = makePlan(processor, StreamGroups({first, second}), history_lengths);
    sampling_output = makeSamplingOutput(plan.sampling_plan);
    processor.prepareSampling(plan, cache, {}, states, sampling_output.request_errors, draft, params);
    EXPECT_TRUE(sampling_output.request_errors[0].ok());
    EXPECT_NE(sampling_output.request_errors[1].ToString().find("missing PP DSpARK proposal probabilities"), std::string::npos);
    EXPECT_TRUE(torch::equal(draft.all_probs[0], first_q));
    EXPECT_EQ(draft.all_probs[1].count_nonzero().item<int64_t>(), 0);
    EXPECT_TRUE(cache.empty());
}

/** Verify rows append candidate prefixes to committed history without advancing persistent state. */
TEST_F(PPBatchStreamProcessorTest, SamplerInputsRouteMultiSequenceStateAndClampPositiveTopK) {
    setVocab(16);
    auto cfg                 = config(2);
    cfg.top_k                = 99;
    cfg.random_seed          = 123;
    cfg.return_cum_log_probs = true;
    auto multi               = makeStream(101, {1, 2}, cfg);
    auto single              = makeStream(202, {3}, config(0));
    commitTokens(multi, intTensor({10, 11}).reshape({2, 1}));
    auto                                   processor = makeProcessor();
    StreamGroups                           groups({multi, single});
    const auto                             execution_plan = makePlan(processor, groups);
    const auto&                            plan           = execution_plan.sampling_plan;
    auto                                   sampling_output         = makeSamplingOutput(plan);
    SamplingStates                         states;
    SamplerOutput                          draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    states.at(101).sampling.cum_log_probs.copy_(torch::tensor({-2.0f, -4.0f}));
    auto first_processor                      = std::make_shared<RecordingLogitsProcessor>();
    auto second_processor                     = std::make_shared<RecordingLogitsProcessor>();
    states.at(101).sampling.logits_processors = {first_processor};
    states.at(202).sampling.logits_processors = {second_processor};
    auto inputs = processor.gatherSamplerInputs(plan, execution_plan.output_config, torch::zeros({3, 16}), states);
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
    EXPECT_EQ(inputs.generator[0], states.at(101).sampling.generator);
    EXPECT_EQ(inputs.generator[1], states.at(101).sampling.generator);
    EXPECT_FALSE(inputs.generator[2].defined());
    inputs.logits_processor_states_ptr->batchProcess(inputs);
    EXPECT_EQ(first_processor->intervals, (std::vector<std::pair<size_t, size_t>>{{0, 2}}));
    EXPECT_EQ(second_processor->intervals, (std::vector<std::pair<size_t, size_t>>{{2, 3}}));
    EXPECT_EQ(first_processor->updates, 0);
    EXPECT_EQ(second_processor->updates, 0);
}

TEST_F(PPBatchStreamProcessorTest, VerifySamplerInputsExpandEachCandidatePrefix) {
    auto cfg                 = config();
    cfg.top_k                = 1;
    cfg.repetition_penalty   = 1.7f;
    cfg.presence_penalty     = 0.4f;
    cfg.frequency_penalty    = 0.3f;
    cfg.no_repeat_ngram_size = 2;
    auto first               = makeStream(101, {1, 2}, cfg);
    auto second              = makeStream(202, {7});
    commitTokens(first, intTensor({1}).reshape({1, 1}));
    commitTokens(second, intTensor({8}).reshape({1, 1}));
    setProposals(first, {1, 2, 9, 10});
    setProposals(second, {8, 8, 11, 12});
    auto                                   processor      = makeProcessor(SP_TYPE_MTP);
    const auto                             execution_plan = makePlan(processor, StreamGroups({first, second}));
    const auto&                            candidates     = execution_plan.model_input.combo_tokens;
    const auto&                            plan           = execution_plan.sampling_plan;
    auto                                   sampling_output         = makeSamplingOutput(plan);
    SamplingStates                         states;
    SamplerOutput                          draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    EXPECT_TRUE(torch::equal(draft_sampler_output.token_ids, intTensor({2, 9, 10, 8, 11, 12}).reshape({2, 3})));
    EXPECT_TRUE(draft_sampler_output.token_ids_are_point_mass);
    EXPECT_FALSE(draft_sampler_output.all_probs.defined());
    ASSERT_EQ(params.generators.size(), 2u);
    EXPECT_FALSE(params.generators[0].defined());
    EXPECT_FALSE(params.generators[1].defined());
    auto first_processor                      = std::make_shared<RecordingLogitsProcessor>();
    auto second_processor                     = std::make_shared<RecordingLogitsProcessor>();
    states.at(101).sampling.logits_processors = {first_processor};
    states.at(202).sampling.logits_processors = {second_processor};
    auto logits                               = torch::zeros({8, 64});
    auto inputs = processor.gatherSamplerInputs(plan, PPOutputConfig{}, logits, states, true, 3, candidates);
    EXPECT_EQ(inputs.step, 6);
    EXPECT_EQ(inputs.batch_size, 8);
    EXPECT_EQ(states.at(101).token_history, (std::vector<std::vector<int32_t>>{{1, 2, 1}}));
    EXPECT_EQ(states.at(202).token_history, (std::vector<std::vector<int32_t>>{{7, 8}}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.sequence_lengths), (std::vector<int32_t>{3, 4, 5, 6, 2, 3, 4, 5}));
    EXPECT_EQ(tensorToVector<int32_t>(inputs.input_lengths), (std::vector<int32_t>{2, 2, 2, 2, 1, 1, 1, 1}));
    const std::vector<std::vector<int32_t>> histories{{1, 2, 1},
                                                      {1, 2, 1, 2},
                                                      {1, 2, 1, 2, 9},
                                                      {1, 2, 1, 2, 9, 10},
                                                      {7, 8},
                                                      {7, 8, 8},
                                                      {7, 8, 8, 11},
                                                      {7, 8, 8, 11, 12}};
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
    setVocab(4);
    auto                                   stream         = makeStream(101, {1, 2});
    auto                                   processor      = makeProcessor();
    auto                                   execution_plan = makePlan(processor, StreamGroups({stream}));
    const auto&                            plan           = execution_plan.sampling_plan;
    auto                                   sampling_output         = makeSamplingOutput(plan);
    SamplingStates                         states;
    SamplerOutput                          draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    processor.prepareSampling(execution_plan,
                              dspark_q_,
                              {},
                              states,
                              sampling_output.request_errors,
                              draft_sampler_output,
                              params);
    for (bool return_logits : {false, true}) {
        PPOutputConfig output;
        output.return_logits        = return_logits;
        output.return_softmax_probs = !return_logits;
        auto logits                 = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({1, 4});
        auto inputs                 = processor.gatherSamplerInputs(plan, output, logits, states);
        inputs.logits.zero_();
        EXPECT_EQ(tensorToVector<float>(logits), (std::vector<float>{1, 2, 3, 4}));
        EXPECT_FALSE(inputs.cum_log_probs.defined());
        EXPECT_FALSE(inputs.all_probs.defined());
    }
    for (auto mode : {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::ORIGINAL}) {
        PPOutputConfig output;
        output.return_all_probs = mode;
        auto inputs             = processor.gatherSamplerInputs(plan, output, torch::zeros({1, 4}), states);
        ASSERT_TRUE(inputs.all_probs.defined());
        EXPECT_EQ(inputs.all_probs.sizes().vec(), (std::vector<int64_t>{1, 4}));
        EXPECT_EQ(inputs.all_probs.sum().item<float>(), 0);
        EXPECT_EQ(inputs.return_original_all_probs, mode == ReturnAllProbsMode::ORIGINAL);
    }
}

TEST_F(PPBatchStreamProcessorTest, OrdinarySamplerMasksPaddingAndKeepsProbabilityWidth) {
    setVocab(4);
    for (bool compact_vocab : {false, true}) {
        SCOPED_TRACE(compact_vocab);
        model_config_.output_vocab_ids = compact_vocab ? std::vector<int64_t>{0, 2, 3} : std::vector<int64_t>{};
        const int64_t                          valid_vocab_size = compact_vocab ? 3 : 4;
        auto                                   stream           = makeStream(101, {1, 2});
        auto                                   processor        = makeProcessor();
        auto                                   execution_plan   = makePlan(processor, StreamGroups({stream}));
        const auto&                            plan             = execution_plan.sampling_plan;
        auto                                   sampling_output           = makeSamplingOutput(plan);
        SamplingStates                         states;
        SamplerOutput                          draft_sampler_output;
        speculative::SpeculativeSamplingParams params;
        processor.prepareSampling(execution_plan,
                                  dspark_q_,
                                  {},
                                  states,
                                  sampling_output.request_errors,
                                  draft_sampler_output,
                                  params);
        for (bool preserve_logits : {false, true}) {
            for (auto mode : {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::ORIGINAL}) {
                PPOutputConfig output;
                output.return_logits    = preserve_logits;
                output.return_all_probs = mode;
                auto logits             = torch::full({1, 8}, 100.0f);
                logits.narrow(1, 0, valid_vocab_size)
                    .copy_(torch::arange(valid_vocab_size, torch::kFloat32).unsqueeze(0) - 4);
                const auto original = logits.clone();
                const auto inputs   = processor.gatherSamplerInputs(plan, output, logits, states);
                EXPECT_EQ(inputs.vocab_size, 8u);
                EXPECT_EQ(inputs.all_probs.sizes().vec(), (std::vector<int64_t>{1, 8}));
                EXPECT_EQ(inputs.return_original_all_probs, mode == ReturnAllProbsMode::ORIGINAL);
                EXPECT_EQ(inputs.logits.argmax(-1).item<int64_t>(), valid_vocab_size - 1);
                for (int64_t column = valid_vocab_size; column < 8; ++column) {
                    EXPECT_EQ(inputs.logits[0][column].item<float>(), -std::numeric_limits<float>::infinity());
                }
                if (preserve_logits) {
                    EXPECT_TRUE(torch::equal(logits, original));
                }
            }
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, VerifySamplerUsesOnlyTheConfiguredTargetVocabulary) {
    setVocab(4);
    auto stream = makeStream(101, {1, 2});
    commitTokens(stream, intTensor({0}).reshape({1, 1}));
    setProposals(stream, {0, 0, 0});
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(static_cast<int>(type));
        auto                                   processor      = makeProcessor(type, RoleType::PDFUSION, 2);
        auto                                   execution_plan = makePlan(processor, StreamGroups({stream}));
        const auto&                            plan           = execution_plan.sampling_plan;
        auto                                   sampling_output         = makeSamplingOutput(plan);
        SamplingStates                         states;
        SamplerOutput                          draft_sampler_output;
        speculative::SpeculativeSamplingParams params;
        processor.prepareSampling(execution_plan,
                                  dspark_q_,
                                  {},
                                  states,
                                  sampling_output.request_errors,
                                  draft_sampler_output,
                                  params);
        PPOutputConfig output;
        output.return_all_probs = ReturnAllProbsMode::DEFAULT;
        auto logits             = torch::full({3, 8}, 100.0f);
        logits.narrow(1, 0, 4).copy_(torch::tensor({-4.f, -3.f, -2.f, -1.f}).repeat({3, 1}));
        const auto original = logits.clone();
        const auto inputs   = processor.gatherSamplerInputs(
            plan, output, logits, states, true, 2, execution_plan.model_input.combo_tokens);
        EXPECT_EQ(inputs.vocab_size, 4u);
        EXPECT_EQ(inputs.logits.sizes().vec(), (std::vector<int64_t>{3, 4}));
        EXPECT_EQ(inputs.all_probs.sizes().vec(), (std::vector<int64_t>{3, 4}));
        EXPECT_TRUE(inputs.logits.is_contiguous());
        EXPECT_EQ(tensorToVector<int64_t>(inputs.logits.argmax(-1)), (std::vector<int64_t>{3, 3, 3}));
        inputs.logits.zero_();
        EXPECT_TRUE(torch::equal(logits, original));
    }
}

/** Target verification expands the anchor into model rows without committing candidates. */
TEST_F(PPBatchStreamProcessorTest, DraftNextPositionsFollowEachRequestAndPositionWidth) {
    for (bool mrope : {false, true}) {
        SCOPED_TRACE(mrope);
        model_config_.has_positional_encoding = true;
        model_config_.mm_model_config.mm_position_ids_style =
            mrope ? PositionIdsStyle::MROPE : PositionIdsStyle::DEFAULT;
        model_config_.attn_config.rope_config.index_factor = mrope ? 3 : 1;
        auto                                   first       = makeStream(101, {1, 2});
        auto                                   second      = makeStream(202, {3, 4, 5});
        auto                                   processor   = makeProcessor(SP_TYPE_MTP);
        TensorHolder                           holder;
        StreamGroups                           groups({first, second});
        std::unordered_map<RequestId, int32_t> history_lengths;
        auto gathered = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
        ASSERT_TRUE(gathered.ok());
        const auto& positions = gathered->draft_next_position_ids;
        EXPECT_EQ(tensorToVector<int32_t>(positions),
                  mrope ? (std::vector<int32_t>{1, 1, 1, 2, 2, 2}) : (std::vector<int32_t>{1, 2}));
        EXPECT_TRUE(positions.device().is_cpu());
        EXPECT_TRUE(positions.is_pinned());
        EXPECT_EQ(history_lengths.at(101), 2);
        EXPECT_EQ(history_lengths.at(202), 3);
        auto ordinary = makeProcessor();
        EXPECT_FALSE(makePlan(ordinary, groups).draft_next_position_ids.defined());

        model_config_.has_positional_encoding               = false;
        model_config_.mm_model_config.mm_position_ids_style = PositionIdsStyle::DEFAULT;
        auto without_positions                              = makeProcessor(SP_TYPE_MTP);
        EXPECT_FALSE(makePlan(without_positions, groups).draft_next_position_ids.defined());
    }
}

TEST_F(PPBatchStreamProcessorTest, VerifyModelInputUsesCurrentAnchorAndPreservesProposalsAndCache) {
    initCache();
    for (int position_width : {0, 1, 3}) {
        for (int draft_count : {1, 3}) {
            SCOPED_TRACE(::testing::Message() << "position_width=" << position_width << ", drafts=" << draft_count);
            const bool mrope                      = position_width == 3;
            model_config_.has_positional_encoding = position_width != 0;
            model_config_.mm_model_config.mm_position_ids_style =
                mrope ? PositionIdsStyle::MROPE : PositionIdsStyle::DEFAULT;
            model_config_.attn_config.rope_config.index_factor = mrope ? 3 : 1;
            auto first                                         = makeStream(101, {1, 2});
            auto second                                        = makeStream(202, {3, 4, 5});
            commitTokens(first, intTensor({10}).reshape({1, 1}));
            commitTokens(second, intTensor({20}).reshape({1, 1}));
            for (const auto& stream : {first, second}) {
                stream->generateContextPositionIds();
                stream->fakeInitKVBlock(2);
            }
            first->kvCacheMutable().cacheResource(0).mutableBlockIds("default").assign(BlockIndicesType{1, 2});
            second->kvCacheMutable().cacheResource(0).mutableBlockIds("default").assign(BlockIndicesType{3, 4});
            /** The anchor comes from current model input, not the draft buffer's previous anchor. */
            const std::vector<int32_t> first_proposals =
                draft_count == 1 ? std::vector<int32_t>{0, 11} : std::vector<int32_t>{0, 11, 12, 13};
            const std::vector<int32_t> second_proposals =
                draft_count == 1 ? std::vector<int32_t>{1, 21} : std::vector<int32_t>{1, 21, 22, 23};
            setProposals(first, first_proposals);
            setProposals(second, second_proposals);
            auto         processor = makeProcessor(SP_TYPE_MTP, RoleType::PDFUSION, draft_count);
            StreamGroups groups({first, second});
            TensorHolder holder;
            auto         ordinary = processor.gatherModelInput(groups, holder);
            ASSERT_TRUE(ordinary.ok());
            std::unordered_map<RequestId, int32_t> history_lengths;
            auto verified = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
            ASSERT_TRUE(verified.ok());
            const auto& input = verified->model_input;
            EXPECT_FALSE(verified->draft_next_position_ids.defined());
            EXPECT_EQ(tensorToVector<int32_t>(verified->sampling_plan.token_ids),
                      (std::vector<int32_t>{1, 2, 10, 3, 4, 5, 20}));
            EXPECT_EQ(tensorToVector<int32_t>(verified->sampling_plan.sequence_lengths), (std::vector<int32_t>{3, 4}));
            EXPECT_EQ(history_lengths.at(101), 3);
            EXPECT_EQ(history_lengths.at(202), 4);
            std::vector<int32_t> expected_tokens(first_proposals);
            expected_tokens.insert(expected_tokens.end(), second_proposals.begin(), second_proposals.end());
            expected_tokens[0]               = 10;
            expected_tokens[draft_count + 1] = 20;
            EXPECT_EQ(tensorToVector<int32_t>(input.combo_tokens), expected_tokens);
            EXPECT_TRUE(input.combo_tokens.device().is_cpu());
            EXPECT_TRUE(input.combo_tokens.is_pinned());
            EXPECT_TRUE(input.is_target_verify);
            EXPECT_EQ(input.sequence_lengths.numel(), 0);
            EXPECT_EQ(tensorToVector<int32_t>(input.prefix_lengths), (std::vector<int32_t>{2, 3}));
            EXPECT_EQ(tensorToVector<int32_t>(input.input_lengths),
                      (std::vector<int32_t>{draft_count + 1, draft_count + 1}));
            EXPECT_TRUE(
                torch::equal(input.lm_output_indexes.cpu(), torch::arange(2 * (draft_count + 1), torch::kInt32)));
            ASSERT_EQ(input.combo_position_ids.defined(), position_width != 0);
            if (position_width != 0) {
                std::vector<int32_t> expected_positions;
                for (int start : {2, 3}) {
                    for (int step = 0; step <= draft_count; ++step) {
                        expected_positions.insert(expected_positions.end(), position_width, start + step);
                    }
                }
                EXPECT_EQ(tensorToVector<int32_t>(input.combo_position_ids), expected_positions);
                EXPECT_TRUE(input.combo_position_ids.is_pinned());
            }
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

TEST_F(PPBatchStreamProcessorTest, VerifyEngramWindowsFollowAcceptedHistory) {
    model_config_.attn_config.v41_kv_source_layer_ids = {0};
    auto stream                                       = makeStream(101, {1, 2, 3, 4});
    commitTokens(stream, intTensor({10}).reshape({1, 1}));
    setProposals(stream, {10, 11, 12});
    auto                                   processor = makeProcessor(SP_TYPE_DSPARK, RoleType::PDFUSION, 2);
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto                                   plan = makePlan(processor, StreamGroups({stream}), history_lengths);
    EXPECT_EQ(tensorToVector<int32_t>(plan.model_input.engram_token_windows),
              (std::vector<int32_t>{10, 4, 3, 2, 11, 10, 4, 3, 12, 11, 10, 4}));

    auto commit_input    = plan.model_input;
    auto target_features = torch::zeros({3, 4});
    processor.prepareDSparkCommitInput(commit_input, target_features, false);
    EXPECT_FALSE(commit_input.engram_token_windows.defined());
    EXPECT_TRUE(torch::equal(commit_input.combo_tokens, plan.model_input.combo_tokens));
    EXPECT_TRUE(torch::equal(commit_input.last_hidden_states, target_features));
    EXPECT_TRUE(plan.model_input.engram_token_windows.defined());

    auto result              = makeResult(plan.sampling_plan);
    result.new_token_ids     = intTensor({20, 0, 0}).reshape({1, 3});
    result.new_token_lengths = intTensor({1});
    result.propose_token_ids = intTensor({21, 22}).reshape({1, 2});
    processor.dispatchExecutionResult(StreamGroups({stream}), result);
    auto next_plan = makePlan(processor, StreamGroups({stream}), history_lengths);
    EXPECT_EQ(tensorToVector<int32_t>(next_plan.sampling_plan.token_ids), (std::vector<int32_t>{20}));
    EXPECT_EQ(tensorToVector<int32_t>(next_plan.model_input.combo_tokens), (std::vector<int32_t>{20, 21, 22}));
    EXPECT_EQ(tensorToVector<int32_t>(next_plan.model_input.engram_token_windows),
              (std::vector<int32_t>{20, 10, 4, 3, 21, 20, 10, 4, 22, 21, 20, 10}));
}

/** Accepted output determines the next anchor and position; rejected suffixes must not advance either. */
TEST_F(PPBatchStreamProcessorTest, DSparkProposalUsesNextRoundStateInBothAnchorModes) {
    constexpr int32_t width                             = 3;
    model_config_.has_positional_encoding               = true;
    model_config_.mm_model_config.mm_position_ids_style = PositionIdsStyle::MROPE;
    model_config_.attn_config.rope_config.index_factor  = 3;
    for (bool sample_from_anchor : {true, false}) {
        SCOPED_TRACE(sample_from_anchor);
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = RoleType::PDFUSION;
        SpeculativeExecutionConfig sp_config;
        sp_config.type                         = SP_TYPE_DSPARK;
        sp_config.gen_num_per_cycle            = width;
        sp_config.sp_dspark_mask_token_id      = 63;
        sp_config.sp_dspark_sample_from_anchor = sample_from_anchor;
        PPBatchStreamProcessor processor(
            model_config_, pd_sep_config, ProfilingDebugLoggingConfig{}, cache_config_, sp_config, true);
        TensorHolder host_holder;

        PPExecutionPlan plan;
        auto& target_input = plan.model_input;
        PPSamplingOutput sampling;
        target_input.combo_tokens          = intTensor({1, 2, 3, 4, 5});
        target_input.input_lengths         = intTensor({2, 3});
        target_input.prefix_lengths        = intTensor({5, 8});
        target_input.sequence_lengths      = intTensor({});
        target_input.lm_output_indexes     = intTensor({1, 4});
        target_input.combo_position_ids    = intTensor({10, 20, 30, 11, 21, 31, 40, 50, 60, 41, 51, 61, 42, 52, 62});
        target_input.engram_token_windows  = torch::zeros({5, 4}, torch::kInt32);
        target_input.request_id            = torch::tensor({101, 202}, torch::kInt64);
        target_input.request_pd_separation = torch::ones({2}, torch::kBool);
        target_input.cache_keys            = torch::zeros({2, 1}, torch::kInt64);

        plan.draft_next_position_ids = intTensor({31, 31, 31, 62, 62, 62});
        sampling.sampler_output.token_ids = intTensor({1, 2, 7, 3, 4, 8}).reshape({2, 3});
        auto round_state = processor.buildDSparkRoundState(plan, sampling, host_holder);
        EXPECT_EQ(tensorToVector<int32_t>(round_state.anchors), (std::vector<int32_t>{7, 8}));
        EXPECT_TRUE(round_state.anchors.is_cuda());
        EXPECT_EQ(tensorToVector<int32_t>(round_state.committed_ends), (std::vector<int32_t>{7, 11}));
        EXPECT_EQ(tensorToVector<int32_t>(round_state.position_bases), (std::vector<int32_t>{32, 32, 32, 63, 63, 63}));
        auto draft_input = target_input;
        processor.prepareDSparkProposeModelInput(round_state, draft_input, host_holder);
        const int32_t query_width = width + static_cast<int32_t>(!sample_from_anchor);
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.input_lengths), (std::vector<int32_t>{query_width, query_width}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_tokens),
                  sample_from_anchor ? (std::vector<int32_t>{7, 63, 63, 8, 63, 63}) :
                                       (std::vector<int32_t>{7, 63, 63, 63, 8, 63, 63, 63}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.lm_output_indexes),
                  sample_from_anchor ? (std::vector<int32_t>{0, 1, 2, 3, 4, 5}) :
                                       (std::vector<int32_t>{1, 2, 3, 5, 6, 7}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.prefix_lengths), (std::vector<int32_t>{7, 11}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_position_ids),
                  sample_from_anchor ?
                      (std::vector<int32_t>{32, 32, 32, 33, 33, 33, 34, 34, 34, 63, 63, 63, 64, 64, 64, 65, 65, 65}) :
                      (std::vector<int32_t>{32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35,
                                            63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66}));
        EXPECT_TRUE(draft_input.combo_position_ids.is_cuda());
        EXPECT_FALSE(draft_input.engram_token_windows.defined());
        EXPECT_FALSE(draft_input.request_id.defined());
        EXPECT_FALSE(draft_input.request_pd_separation.defined());
        EXPECT_FALSE(draft_input.cache_keys.defined());
        EXPECT_TRUE(target_input.engram_token_windows.defined());
        EXPECT_TRUE(target_input.request_id.defined());

        /** Batch growth includes immediate rejection, partial acceptance and full acceptance. */
        target_input                   = GptModelInputs{};
        target_input.sequence_lengths  = intTensor({});
        target_input.lm_output_indexes = torch::arange(12, torch::kInt32);
        target_input.is_target_verify  = true;
        target_input.combo_tokens      = intTensor({7, 11, 12, 13, 8, 21, 22, 23, 9, 31, 32, 33});
        target_input.input_lengths     = intTensor({4, 4, 4});
        target_input.prefix_lengths    = intTensor({7, 11, 20});
        target_input.combo_position_ids =
            intTensor({32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 63, 63, 63, 64, 64, 64,
                       65, 65, 65, 66, 66, 66, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93});
        sampling.verify_output.accept_tokens =
            intTensor({15, 0, 0, 0, 21, 25, 0, 0, 31, 32, 33, 35}).reshape({3, 4});
        sampling.verify_output.accept_len = intTensor({1, 2, 4});
        round_state = processor.buildDSparkRoundState(plan, sampling, host_holder);
        EXPECT_EQ(tensorToVector<int32_t>(round_state.anchors), (std::vector<int32_t>{15, 25, 35}));
        EXPECT_EQ(tensorToVector<int32_t>(round_state.committed_ends), (std::vector<int32_t>{8, 13, 24}));
        EXPECT_EQ(tensorToVector<int32_t>(round_state.position_bases),
                  (std::vector<int32_t>{33, 33, 33, 65, 65, 65, 94, 94, 94}));
        draft_input = target_input;
        processor.prepareDSparkProposeModelInput(round_state, draft_input, host_holder);
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_tokens.reshape({3, query_width}).select(1, 0)),
                  (std::vector<int32_t>{15, 25, 35}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.prefix_lengths), (std::vector<int32_t>{8, 13, 24}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.lm_output_indexes),
                  sample_from_anchor ? (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}) :
                                       (std::vector<int32_t>{1, 2, 3, 5, 6, 7, 9, 10, 11}));

        /** A smaller batch without explicit positions must not retain the previous round's metadata. */
        target_input.combo_tokens       = intTensor({15, 16, 17, 18});
        target_input.input_lengths      = intTensor({4});
        target_input.prefix_lengths     = intTensor({8});
        target_input.lm_output_indexes  = torch::arange(4, torch::kInt32);
        target_input.combo_position_ids = torch::Tensor();
        sampling.verify_output.accept_tokens = intTensor({16, 0, 0, 0}).reshape({1, 4});
        sampling.verify_output.accept_len = intTensor({1});
        round_state = processor.buildDSparkRoundState(plan, sampling, host_holder);
        draft_input = target_input;
        processor.prepareDSparkProposeModelInput(round_state, draft_input, host_holder);
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_tokens),
                  sample_from_anchor ? (std::vector<int32_t>{16, 63, 63}) : (std::vector<int32_t>{16, 63, 63, 63}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.lm_output_indexes),
                  sample_from_anchor ? (std::vector<int32_t>{0, 1, 2}) : (std::vector<int32_t>{1, 2, 3}));
        EXPECT_FALSE(draft_input.combo_position_ids.defined());
    }
}

/** Result assembly maps sampler rows back to requests and preserves request-local errors. */
TEST_F(PPBatchStreamProcessorTest, FillResultCommitsLastSampleAndCopiesRequestedOutputs) {
    auto cfg                     = config(2);
    cfg.return_logits            = true;
    cfg.return_softmax_probs     = true;
    cfg.return_cum_log_probs     = true;
    cfg.return_all_probs         = ReturnAllProbsMode::DEFAULT;
    cfg.return_hidden_states     = true;
    cfg.return_all_hidden_states = true;
    auto         multi           = makeStream(101, {1, 2}, cfg);
    auto         single          = makeStream(202, {3});
    StreamGroups groups({multi, single});
    auto         processor             = makeProcessor();
    auto         plan                  = makePlan(processor, groups);
    plan.model_input.lm_output_indexes = intTensor({1, 3, 4});

    GptModelOutputs output;
    auto            raw_logits = torch::zeros({3, 64});
    for (int row = 0; row < 3; ++row) {
        raw_logits[row][7 + row] = std::log(3.0f + row);
    }
    output.logits            = raw_logits.to(torch::kCUDA);
    output.hidden_states     = torch::arange(12, torch::kFloat32).reshape({3, 4}).to(torch::kCUDA);
    output.all_hidden_states = torch::arange(20, torch::kFloat32).reshape({5, 4}).to(torch::kCUDA);
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    auto& sampled = sampling.sampler_output;
    sampled.token_ids     = intTensor({1, 7, 2, 8, 3, 9}).reshape({3, 2}).to(torch::kCUDA);
    sampled.success       = torch::tensor({true, true, true}, torch::kBool).to(torch::kCUDA);
    sampled.cum_log_probs = torch::tensor({-1.0f, -2.0f, -3.0f}).to(torch::kCUDA);
    sampled.all_probs     = torch::full({3, 64}, 1.0f / 64, output.logits.options());
    PPExecutionResult result;
    processor.fillExecutionResult(plan, output, sampling, {}, result);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{7, 8, 9}));
    EXPECT_TRUE(result.new_token_ids.device().is_cpu());
    ASSERT_EQ(result.request_outputs.size(), 2u);
    const auto& requested = result.request_outputs[0];
    EXPECT_TRUE(torch::equal(requested.logits, raw_logits.narrow(0, 0, 2)));
    for (int row = 0; row < 2; ++row) {
        EXPECT_NEAR(requested.softmax_probs[row][0].item<float>(), (3.0f + row) / (66.0f + row), 1e-6);
    }
    EXPECT_EQ(tensorToVector<float>(requested.cum_log_probs), (std::vector<float>{-1, -2}));
    EXPECT_TRUE(torch::equal(requested.all_probs, sampled.all_probs.narrow(0, 0, 2).cpu()));
    EXPECT_TRUE(torch::equal(requested.hidden_states, output.hidden_states.narrow(0, 0, 2).cpu()));
    EXPECT_TRUE(torch::equal(requested.all_hidden_states, output.all_hidden_states.narrow(0, 0, 4).cpu()));
    EXPECT_FALSE(requested.loss.defined());
    const auto& unrequested = result.request_outputs[1];
    EXPECT_FALSE(unrequested.logits.defined());
    EXPECT_FALSE(unrequested.softmax_probs.defined());
    EXPECT_FALSE(unrequested.cum_log_probs.defined());
    EXPECT_FALSE(unrequested.all_probs.defined());
    EXPECT_FALSE(unrequested.hidden_states.defined());
    EXPECT_FALSE(unrequested.all_hidden_states.defined());
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{1, 1, 1}));
    ASSERT_EQ(result.request_errors.size(), 2);
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_TRUE(result.request_errors[1].ok());
}

TEST_F(PPBatchStreamProcessorTest, MixedBatchReturnsOnlyRequestedRowsAtNonzeroOffsets) {
    auto decode = makeStream(101, {1, 2});
    commitTokens(decode, intTensor({7}).reshape({1, 1}));
    auto cfg                                      = config(2);
    cfg.return_logits                             = true;
    cfg.return_hidden_states                      = true;
    cfg.return_all_hidden_states                  = true;
    auto                                   multi  = makeStream(202, {3, 4, 5}, cfg);
    auto                                   single = makeStream(303, {6, 7});
    StreamGroups                           groups({multi, single, decode});
    auto                                   processor = makeProcessor();
    TensorHolder                           holder;
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto gathered = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
    ASSERT_TRUE(gathered.ok());
    auto plan = std::move(gathered.value());
    ASSERT_EQ(tensorToVector<int64_t>(plan.sampling_plan.request_ids), (std::vector<int64_t>{101, 202, 303}));
    ASSERT_EQ(tensorToVector<int32_t>(plan.model_input.lm_output_indexes), (std::vector<int32_t>{0, 3, 6, 8}));

    GptModelOutputs model_output;
    model_output.logits            = torch::arange(256, torch::kFloat32).reshape({4, 64}).cuda();
    model_output.hidden_states     = torch::arange(16, torch::kFloat32).reshape({4, 4}).cuda();
    model_output.all_hidden_states = torch::arange(36, torch::kFloat32).reshape({9, 4}).cuda();
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    auto& sampled = sampling.sampler_output;
    sampled.token_ids = intTensor({10, 11, 12, 13}).reshape({4, 1}).cuda();
    sampled.success   = torch::ones({4}, torch::TensorOptions(torch::kBool).device(torch::kCUDA));
    PPExecutionResult result;
    processor.fillExecutionResult(plan, model_output, sampling, {}, result);

    ASSERT_EQ(result.request_outputs.size(), 3u);
    const auto& output = result.request_outputs[1];
    EXPECT_TRUE(output.logits.is_cpu());
    EXPECT_TRUE(torch::equal(output.logits, torch::arange(64, 192, torch::kFloat32).reshape({2, 64})));
    EXPECT_TRUE(torch::equal(output.hidden_states, torch::arange(4, 12, torch::kFloat32).reshape({2, 4})));
    EXPECT_TRUE(torch::equal(output.all_hidden_states, torch::arange(4, 28, torch::kFloat32).reshape({6, 4})));
    for (auto index : {0, 2}) {
        const auto& empty = result.request_outputs[index];
        EXPECT_FALSE(empty.logits.defined());
        EXPECT_FALSE(empty.hidden_states.defined());
        EXPECT_FALSE(empty.all_hidden_states.defined());
    }
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{10, 11, 12, 13}));
    processor.dispatchExecutionResult(groups, result);
    EXPECT_EQ(multi->completeTokenIdsVec(0), (std::vector<int>{3, 4, 5, 11}));
    EXPECT_EQ(multi->completeTokenIdsVec(1), (std::vector<int>{3, 4, 5, 12}));
    for (const auto& stream : {decode, single}) {
        auto received = stream->nextOutput(1);
        ASSERT_TRUE(received.ok());
        const auto& value = received.value().generate_outputs.at(0);
        EXPECT_FALSE(value.logits.has_value());
        EXPECT_FALSE(value.hidden_states.has_value());
        EXPECT_FALSE(value.all_hidden_states.has_value());
    }
}

TEST_F(PPBatchStreamProcessorTest, VerifyResultReturnsAcceptedTokensAndErrorsWithoutOptionalOutputs) {
    auto         first  = makeStream(101, {1, 2});
    auto         second = makeStream(202, {3});
    auto         third  = makeStream(303, {4});
    StreamGroups groups({first, second, third});
    auto         processor = makeProcessor(SP_TYPE_MTP, RoleType::PDFUSION, 2);
    auto         plan      = makePlan(processor, groups);
    plan.is_decode         = true;
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    auto& target_sampled = sampling.sampler_output;
    target_sampled.success =
        torch::tensor({true, true, true, false, true, true, true, false, true}, torch::kBool).to(torch::kCUDA);
    const auto accepted_tokens = intTensor({10, 11, 12, 13, 0, 0, 16, 17, 0}).reshape({3, 3});
    auto& sp_output = sampling.verify_output;
    sp_output.accept_tokens_cpu = accepted_tokens;
    sp_output.accept_len_cpu    = intTensor({3, 1, 2});
    sp_output.processor_errors = {std::nullopt, ErrorInfo(ErrorCode::INVALID_PARAMS, "verify failed"), std::nullopt};
    PPExecutionResult result;
    processor.fillExecutionResult(plan, GptModelOutputs{}, sampling, {}, result);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{10, 11, 12, 13, 0, 0, 16, 17, 0}));
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{3, 1, 2}));
    for (const auto& output : result.request_outputs) {
        EXPECT_FALSE(output.logits.defined());
        EXPECT_FALSE(output.softmax_probs.defined());
        EXPECT_FALSE(output.cum_log_probs.defined());
        EXPECT_FALSE(output.all_probs.defined());
        EXPECT_FALSE(output.hidden_states.defined());
        EXPECT_FALSE(output.all_hidden_states.defined());
        EXPECT_FALSE(output.loss.defined());
        EXPECT_FALSE(output.prompt_logits.has_value());
    }
    /** Processor errors take precedence over sampler failures. */
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_EQ(result.request_errors[1].code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(result.request_errors[2].code(), ErrorCode::UNKNOWN_ERROR);
}

TEST_F(PPBatchStreamProcessorTest, FillResultPreservesEarlierErrorsAndMapsSamplerRowsToRequests) {
    auto            first     = makeStream(101, {1});
    auto            multi     = makeStream(202, {2}, config(2));
    auto            failed    = makeStream(303, {3});
    auto            healthy   = makeStream(404, {4});
    auto            processor = makeProcessor();
    auto            plan      = makePlan(processor, StreamGroups({first, multi, failed, healthy}));
    PPExecutionResult result;
    const ErrorInfo initialization_error(ErrorCode::INVALID_PARAMS, "tail grammar backend unavailable");
    const ErrorInfo processor_error(ErrorCode::EXECUTION_EXCEPTION, "logits processor failed");
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    auto& sampled = sampling.sampler_output;
    sampling.request_errors[0] = initialization_error;
    sampled.token_ids = intTensor({0, 10, 0, 0, 12}).reshape({5, 1});
    sampled.success   = torch::tensor({false, true, false, false, true}, torch::kBool);
    sampled.processor_errors.resize(5);
    sampled.processor_errors[3] = processor_error;
    processor.fillExecutionResult(plan, GptModelOutputs{}, sampling, {}, result);
    EXPECT_EQ(result.request_errors[0].ToString(), initialization_error.ToString());
    EXPECT_EQ(result.request_errors[1].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_EQ(result.request_errors[2].ToString(), processor_error.ToString());
    EXPECT_TRUE(result.request_errors[3].ok());
    EXPECT_EQ(result.new_token_ids.size(0), 5);
    for (const auto& output : result.request_outputs) {
        EXPECT_FALSE(output.logits.defined());
        EXPECT_FALSE(output.softmax_probs.defined());
        EXPECT_FALSE(output.cum_log_probs.defined());
        EXPECT_FALSE(output.all_probs.defined());
        EXPECT_FALSE(output.hidden_states.defined());
        EXPECT_FALSE(output.all_hidden_states.defined());
    }
}

TEST_F(PPBatchStreamProcessorTest, FillResultUsesCompactProbabilityBeforeRestoringTokenIds) {
    model_config_.output_vocab_ids = {0, 2, 7, 63};
    auto cfg                       = config();
    cfg.return_softmax_probs       = true;
    auto            failed         = makeStream(101, {1, 2}, cfg);
    auto            healthy        = makeStream(202, {3}, cfg);
    auto            processor      = makeProcessor();
    StreamGroups    groups({failed, healthy});
    auto            plan = makePlan(processor, groups);
    GptModelOutputs output;
    output.logits =
        torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, std::log(3.0f), 0.0f}).reshape({2, 4}).to(torch::kCUDA);
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    auto& sampled = sampling.sampler_output;
    sampled.token_ids = intTensor({0, 2}).reshape({2, 1}).to(torch::kCUDA);
    sampled.success   = torch::tensor({false, true}, torch::kBool).to(torch::kCUDA);
    PPExecutionResult result;
    processor.fillExecutionResult(plan, output, sampling, {}, result);
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{0, 7}));
    EXPECT_NEAR(result.request_outputs[1].softmax_probs[0][0].item<float>(), 0.5f, 1e-6);
}

TEST_F(PPBatchStreamProcessorTest, FillResultSkipsFailedSamplerSentinelsDuringVocabularyMapping) {
    model_config_.output_vocab_ids = {2, 7, 63};
    auto failed = makeStream(101, {1});
    auto healthy = makeStream(202, {3});
    auto processor = makeProcessor();
    auto plan = makePlan(processor, StreamGroups({failed, healthy}));
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    sampling.sampler_output.token_ids = intTensor({-1, 1}).reshape({2, 1}).cuda();
    sampling.sampler_output.success = torch::tensor({false, true}, torch::kBool).cuda();
    PPExecutionResult result;
    processor.fillExecutionResult(plan, {}, sampling, {}, result);
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{-1, 7}));
}

TEST_F(PPBatchStreamProcessorTest, FillResultPromptLogitsAreRequestLocalAndUseFirstSequenceOnlyInPrefill) {
    setVocab(4);
    auto cfg                                           = config(2);
    cfg.return_prompt_logits                           = true;
    cfg.prompt_logits_top_k                            = 2;
    cfg.prompt_logits_start                            = 1;
    cfg.prompt_logits_end                              = 3;
    auto multi                                         = makeStream(202, {0, 1, 2}, cfg);
    auto short_config                                  = config();
    short_config.return_prompt_logits                  = true;
    short_config.prompt_logits_top_k                   = 2;
    auto                                   single      = makeStream(303, {1}, short_config);
    auto                                   unrequested = makeStream(404, {2});
    auto                                   processor   = makeProcessor();
    StreamGroups                           groups({multi, single, unrequested});
    TensorHolder                           holder;
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto gathered = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
    ASSERT_TRUE(gathered.ok());
    auto plan = std::move(gathered.value());
    ASSERT_EQ(tensorToVector<int32_t>(plan.model_input.lm_output_indexes), (std::vector<int32_t>{2, 5, 6, 7}));
    GptModelOutputs output;
    output.all_logits = torch::zeros({8, 4});
    output.all_logits[0].copy_(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).log());
    output.all_logits[1].copy_(torch::tensor({4.0f, 3.0f, 2.0f, 1.0f}).log());
    output.all_logits[2].copy_(torch::tensor({1.0f, 3.0f, 2.0f, 4.0f}).log());
    output.all_logits[6].copy_(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).log());
    auto sampling = makeSamplingOutput(plan.sampling_plan);
    auto& sampled = sampling.sampler_output;
    sampled.token_ids = intTensor({2, 2, 0, 0}).reshape({4, 1});
    sampled.success   = torch::ones({4}, torch::kBool);
    PPExecutionResult result;
    processor.fillExecutionResult(plan, output, sampling, {}, result);
    ASSERT_EQ(result.request_outputs.size(), 3u);
    for (const auto& request : result.request_outputs) {
        EXPECT_FALSE(request.loss.defined());
    }
    EXPECT_FALSE(result.request_outputs[2].prompt_logits.has_value());
    ASSERT_TRUE(result.request_outputs[0].prompt_logits.has_value());
    const auto& prompt = result.request_outputs[0].prompt_logits.value();
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
    ASSERT_TRUE(result.request_outputs[1].prompt_logits.has_value());
    EXPECT_EQ(result.request_outputs[1].prompt_logits->topk_logprobs.size(0), 1);
    EXPECT_FALSE(result.request_outputs[1].prompt_logits->target_logprobs.defined());

    plan.is_decode     = true;
    PPExecutionResult decode_result;
    processor.fillExecutionResult(plan, output, sampling, {}, decode_result);
    for (const auto& request : decode_result.request_outputs) {
        EXPECT_FALSE(request.prompt_logits.has_value());
        EXPECT_FALSE(request.loss.defined());
    }
}

/** Dispatch commits each successful request's slice and retires every returned inflight row. */
TEST_F(PPBatchStreamProcessorTest, PrefillOnlyDispatchFinishesWithoutTokensOrPdHandoff) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
            for (bool pd_separation : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", pd=" << pd_separation << ", role=" << role);
                auto cfg                                   = config();
                cfg.max_new_tokens                         = 0;
                cfg.pd_separation                          = pd_separation;
                auto healthy                               = makeStream(101, {1, 2}, cfg);
                auto cancelled                             = makeStream(202, {3, 4}, cfg);
                auto failed                                = makeStream(303, {5, 6}, cfg);
                auto logits_processor                      = std::make_shared<RecordingLogitsProcessor>();
                healthy->sampling_state_.logits_processors = {logits_processor};
                const StreamGroups groups({healthy, cancelled, failed});
                for (const auto& stream : groups.allStreams()) {
                    stream->setPPInflight();
                }
                cancelled->reportError(ErrorCode::CANCELLED, "cancelled while inflight");
                const auto cancellation  = cancelled->statusInfo().ToString();
                auto       result        = makeResult(std::vector<int64_t>{101, 202, 303});
                result.new_token_ids     = torch::full({3, 1}, 9, torch::kInt32);
                result.new_token_lengths = torch::ones({3}, torch::kInt32);
                /** DP alignment can produce unused drafts; zero-token completion must discard their outputs. */
                if (type != SP_TYPE_NONE && (type != SP_TYPE_DSPARK || role == RoleType::PDFUSION)) {
                    const int width          = role == RoleType::PREFILL ? 1 : 3;
                    result.propose_token_ids = torch::full({3, width}, 9, torch::kInt32);
                    if (role == RoleType::PREFILL) {
                        result.propose_all_probs = torch::zeros({3, 1, model_config_.vocab_size}, torch::kFloat32);
                        result.propose_all_probs.select(2, 9).fill_(1);
                        result.propose_hidden_states = torch::ones({3, 4}, torch::kFloat32);
                    }
                }
                result.request_errors[1] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "late tail error");
                result.request_errors[2] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "tail error");
                auto processor           = makeProcessor(type, role);
                processor.dispatchExecutionResult(groups, result);

                EXPECT_FALSE(healthy->hasError());
                EXPECT_TRUE(healthy->hasEvent(StreamEvents::GenerateDone));
                EXPECT_FALSE(healthy->hasEvent(StreamEvents::NeedRemoteGenerate));
                EXPECT_EQ(healthy->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
                EXPECT_EQ(logits_processor->updates, 0);
                auto output = healthy->nextOutput(1);
                ASSERT_TRUE(output.ok());
                ASSERT_EQ(output.value().generate_outputs.size(), 1u);
                EXPECT_TRUE(output.value().generate_outputs[0].finished);
                EXPECT_EQ(output.value().generate_outputs[0].output_ids.numel(), 0);
                EXPECT_EQ(cancelled->statusInfo().ToString(), cancellation);
                EXPECT_EQ(failed->statusInfo().code(), ErrorCode::UNKNOWN_ERROR);
                EXPECT_FALSE(cancelled->hasOutput());
                EXPECT_FALSE(failed->hasOutput());
                for (const auto& stream : groups.allStreams()) {
                    EXPECT_FALSE(stream->isPPInflight());
                    EXPECT_FALSE(stream->getSPOutputBuffer());
                    EXPECT_TRUE(stream->getProposeToken().empty());
                }
            }
        }
    }
}

TEST_F(PPBatchStreamProcessorTest, NormalDispatchPreservesProcessorStateAfterLocalHandoff) {
    auto cfg                                  = config();
    cfg.max_new_tokens                        = 2;
    auto stream                               = makeStream(101, {1, 2}, cfg);
    auto head_processor                       = std::make_shared<RecordingLogitsProcessor>();
    stream->sampling_state_.logits_processors = {head_processor};

    stream->update({intTensor({9}).reshape({1, 1}), 1});
    ASSERT_FALSE(stream->hasError());
    EXPECT_EQ(head_processor->updates, 1);
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 9}));
    auto handoff_output = stream->nextOutput(1);
    ASSERT_TRUE(handoff_output.ok());
    ASSERT_EQ(handoff_output.value().generate_outputs.size(), 1u);
    const auto& handoff = handoff_output.value().generate_outputs[0];
    EXPECT_EQ(tensorToVector<int32_t>(handoff.output_ids), (std::vector<int32_t>{9}));
    EXPECT_FALSE(handoff.finished);

    auto result              = makeResult(std::vector<int64_t>{101});
    result.new_token_ids     = intTensor({10}).reshape({1, 1});
    result.new_token_lengths = intTensor({1});
    stream->setPPInflight();
    auto processor = makeProcessor();
    processor.dispatchExecutionResult(StreamGroups({stream}), result);

    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(head_processor->updates, 1);
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 9, 10}));
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_FALSE(stream->isPPInflight());
    auto output = stream->nextOutput(1);
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1u);
    const auto& final_output = output.value().generate_outputs[0];
    EXPECT_EQ(tensorToVector<int32_t>(final_output.output_ids), (std::vector<int32_t>{10}));
    EXPECT_TRUE(final_output.finished);
}

TEST_F(PPBatchStreamProcessorTest, NormalDispatchUsesRequestOutputsAcrossMixedPhasesAndSequences) {
    auto decode = makeStream(101, {1, 2});
    commitTokens(decode, intTensor({9}).reshape({1, 1}));
    auto cfg                           = config(2);
    cfg.return_logits                  = true;
    cfg.return_softmax_probs           = true;
    cfg.return_cum_log_probs           = true;
    cfg.return_all_probs               = ReturnAllProbsMode::DEFAULT;
    cfg.return_hidden_states           = true;
    cfg.return_all_hidden_states       = true;
    cfg.return_prompt_logits           = true;
    auto multi                         = makeStream(202, {3, 4, 5}, cfg);
    auto single_config                 = config();
    single_config.return_hidden_states = true;
    auto         single                = makeStream(303, {6, 7}, single_config);
    StreamGroups groups({multi, single, decode});
    auto         processor = makeProcessor();
    TensorHolder holder;
    ASSERT_TRUE(processor.gatherModelInput(groups, holder).ok());

    auto result                    = makeResult(std::vector<int64_t>{101, 202, 303});
    result.new_token_ids           = intTensor({10, 11, 12, 13}).reshape({4, 1});
    result.new_token_lengths       = torch::ones({4}, torch::kInt32);
    auto& multi_result             = result.request_outputs[1];
    multi_result.logits            = torch::arange(64, 192, torch::kFloat32).reshape({2, 64});
    multi_result.hidden_states     = torch::arange(104, 112, torch::kFloat32).reshape({2, 4});
    multi_result.all_hidden_states = torch::arange(204, 228, torch::kFloat32).reshape({6, 4});
    multi_result.softmax_probs     = torch::tensor({0.2f, 0.3f}).reshape({2, 1});
    multi_result.cum_log_probs     = torch::tensor({-2.0f, -3.0f});
    multi_result.all_probs         = torch::zeros({2, 64});
    for (int row = 0; row < 2; ++row) {
        multi_result.all_probs[row][11 + row] = 1;
    }
    result.request_outputs[2].hidden_states = torch::arange(112, 116, torch::kFloat32).reshape({1, 4});
    result.request_outputs[1].prompt_logits = PromptLogitsOutput{
        torch::tensor({-0.5f}).reshape({1, 1}), intTensor({4}).reshape({1, 1}), torch::tensor({-0.5f}), 0, 1};
    for (const auto& stream : groups.allStreams()) {
        stream->setPPInflight();
    }
    processor.dispatchExecutionResult(groups, result);
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
        EXPECT_TRUE(torch::equal(output.logits.value(), multi_result.logits.narrow(0, sequence, 1)));
        ASSERT_TRUE(output.hidden_states.has_value());
        EXPECT_TRUE(torch::equal(output.hidden_states.value(), multi_result.hidden_states.narrow(0, sequence, 1)));
        ASSERT_TRUE(output.all_hidden_states.has_value());
        EXPECT_TRUE(torch::equal(output.all_hidden_states.value(), multi_result.all_hidden_states));
        EXPECT_FALSE(output.loss.has_value());
        ASSERT_TRUE(output.prompt_logits.has_value());
        EXPECT_TRUE(torch::equal(output.prompt_logits->topk_token_ids,
                                 result.request_outputs[1].prompt_logits->topk_token_ids));
        ASSERT_TRUE(output.aux_info.softmax_probs.has_value());
        EXPECT_NEAR(output.aux_info.softmax_probs->item<float>(), sequence == 0 ? 0.2f : 0.3f, 1e-6);
        ASSERT_TRUE(output.aux_info.cum_log_probs.has_value());
        EXPECT_FLOAT_EQ(output.aux_info.cum_log_probs->item<float>(), -2.0f - sequence);
        ASSERT_TRUE(output.aux_info.all_probs.has_value());
        EXPECT_TRUE(torch::equal(output.aux_info.all_probs.value(), multi_result.all_probs.narrow(0, sequence, 1)));
    }
    auto single_output = single->nextOutput(1);
    ASSERT_TRUE(single_output.ok());
    const auto& single_value = single_output.value().generate_outputs.at(0);
    ASSERT_TRUE(single_value.hidden_states.has_value());
    EXPECT_TRUE(torch::equal(single_value.hidden_states.value(), result.request_outputs[2].hidden_states));
    EXPECT_FALSE(single_value.prompt_logits.has_value());
    EXPECT_FALSE(single_value.all_hidden_states.has_value());
    auto decode_output = decode->nextOutput(1);
    ASSERT_TRUE(decode_output.ok());
    EXPECT_FALSE(decode_output.value().generate_outputs.at(0).logits.has_value());
}

TEST_F(PPBatchStreamProcessorTest, NormalDispatchSkipsFailedSequencesWithoutShiftingHealthyRows) {
    auto         failed  = makeStream(101, {1, 2}, config(2));
    auto         healthy = makeStream(202, {3, 4});
    StreamGroups groups({failed, healthy});
    auto         result      = makeResult(std::vector<int64_t>{101, 202});
    result.new_token_ids     = intTensor({0, 0, 12}).reshape({3, 1});
    result.new_token_lengths = torch::ones({3}, torch::kInt32);
    result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
    for (const auto& stream : groups.allStreams()) {
        stream->setPPInflight();
    }
    auto processor = makeProcessor();
    processor.dispatchExecutionResult(groups, result);
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

TEST_F(PPBatchStreamProcessorTest, PrefillDispatchPreservesEachRequestsDraftHandoff) {
    initCache();
    auto first  = makeStream(101, {1, 2});
    auto second = makeStream(202, {3, 4, 5});
    for (const auto& stream : {first, second}) {
        setProposals(stream, {0, 0, 0, 0});
        stream->setProposeToken({});
        stream->setPPInflight();
    }
    StreamGroups groups({first, second});
    auto         result                = makeResult(std::vector<int64_t>{101, 202});
    result.new_token_ids               = intTensor({7, 8}).reshape({2, 1});
    result.new_token_lengths           = torch::ones({2}, torch::kInt32);
    result.propose_token_ids           = intTensor({10, 11}).reshape({2, 1});
    result.propose_all_probs           = torch::zeros({2, 1, model_config_.vocab_size}, torch::kFloat32);
    result.propose_all_probs[0][0][10] = 1;
    result.propose_all_probs[1][0][11] = 1;
    result.propose_hidden_states       = torch::arange(12, torch::kFloat32).reshape({2, 6}).to(torch::kBFloat16);
    auto processor                     = makeProcessor(SP_TYPE_MTP, RoleType::PREFILL);
    processor.dispatchSpeculativeExecutionResult(groups, result);
    int row = 0;
    for (const auto& stream : {first, second}) {
        const auto buffer = stream->getSPOutputBuffer();
        ASSERT_NE(buffer, nullptr);
        EXPECT_EQ(buffer->all_probs.sizes().vec(), (std::vector<int64_t>{1, model_config_.vocab_size}));
        EXPECT_TRUE(torch::equal(buffer->all_probs, result.propose_all_probs[row]));
        EXPECT_EQ(buffer->hidden_states.sizes().vec(), (std::vector<int64_t>{1, 6}));
        EXPECT_EQ(buffer->hidden_states.scalar_type(), torch::kBFloat16);
        EXPECT_TRUE(torch::equal(buffer->hidden_states, result.propose_hidden_states.narrow(0, row, 1)));
        EXPECT_EQ(stream->getProposeToken(), (std::vector<int>{7 + row, 10 + row}));
        EXPECT_FALSE(stream->isPPInflight());
        ++row;
    }
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
        auto       result         = makeResult(std::vector<int64_t>{101, 202});
        result.new_token_ids =
            decode ? intTensor({30, -1, -1, -1, 20, 21, 22, -1}).reshape({2, 4}) : intTensor({30, 20}).reshape({2, 1});
        result.new_token_lengths = intTensor({1, decode ? 3 : 1});
        result.propose_token_ids = intTensor({31, 32, 33, 23, 24, 25}).reshape({2, 3});
        auto processor           = makeProcessor(SP_TYPE_MTP);
        processor.dispatchExecutionResult(StreamGroups({first, second}), result);
        auto expected_first = first_history;
        expected_first.push_back(30);
        auto expected_second = second_history;
        if (decode) {
            expected_second.insert(expected_second.end(), {20, 21, 22});
        } else {
            expected_second.push_back(20);
        }
        EXPECT_EQ(first->spIterCount(), decode ? 1u : 0u);
        EXPECT_EQ(second->spIterCount(), decode ? 1u : 0u);
        EXPECT_EQ(first->speculativeAcceptedTokensPerPos(),
                  decode ? std::vector<int32_t>({0, 0, 0}) : std::vector<int32_t>{});
        EXPECT_EQ(second->speculativeAcceptedTokensPerPos(),
                  decode ? std::vector<int32_t>({1, 1, 0}) : std::vector<int32_t>{});
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

TEST_F(PPBatchStreamProcessorTest, MixedEosAtLengthBoundaryFinishesOnlyThatRequest) {
    initCache();
    for (const auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        const int reserve    = type == SP_TYPE_DSPARK ? 9 : 7;
        auto      finished   = makeStream(101, std::vector<int32_t>(model_config_.max_seq_len - reserve - 2, 1));
        auto      continuing = makeStream(202, {3, 4});
        for (const auto& stream : {finished, continuing}) {
            stream->setReserveStep(reserve);
            commitTokens(stream, intTensor({9}).reshape({1, 1}));
            ASSERT_FALSE(stream->hasEvent(StreamEvents::GenerateDone));
            setProposals(stream, {9, 10, 11, 12});
            stream->setPPInflight();
        }
        auto expected_finished = finished->completeTokenIdsVec(0);
        expected_finished.insert(expected_finished.end(), {30, 63});
        auto result              = makeResult(std::vector<int64_t>{101, 202});
        result.new_token_ids     = intTensor({30, 63, 31, -1, 20, 21, 22, -1}).reshape({2, 4});
        result.new_token_lengths = intTensor({3, 3});
        result.propose_token_ids = intTensor({32, 33, 34, 23, 24, 25}).reshape({2, 3});
        auto processor           = makeProcessor(type);
        processor.dispatchExecutionResult(StreamGroups({finished, continuing}), result);
        EXPECT_EQ(finished->completeTokenIdsVec(0), expected_finished);
        EXPECT_TRUE(finished->hasEvent(StreamEvents::GenerateDone));
        EXPECT_FALSE(continuing->hasEvent(StreamEvents::GenerateDone));
        EXPECT_EQ(continuing->completeTokenIdsVec(0), (std::vector<int>{3, 4, 9, 20, 21, 22}));
        EXPECT_EQ(continuing->getProposeToken(), (std::vector<int>{22, 23, 24, 25}));
        /** EOS truncation must leave the batch's verification lengths unchanged. */
        EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{3, 3}));
        for (const auto& stream : {finished, continuing}) {
            ASSERT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
            EXPECT_FALSE(stream->isPPInflight());
            ASSERT_TRUE(stream->hasOutput());
            const auto output = stream->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1u);
            EXPECT_EQ(output.value().generate_outputs[0].finished, stream == finished);
            EXPECT_TRUE(torch::equal(output.value().generate_outputs[0].output_ids,
                                     stream == finished ? intTensor({30, 63}).reshape({1, 2}) :
                                                          intTensor({20, 21, 22}).reshape({1, 3})));
        }
        EXPECT_EQ(finished->moveToNext(), StreamState::FINISHED);
        EXPECT_EQ(continuing->moveToNext(), StreamState::RUNNING);
        TensorHolder                           holder;
        std::unordered_map<RequestId, int32_t> history_lengths;
        const auto                             next_plan =
            processor.gatherExecutionPlan(StreamGroups({continuing}), {}, false, history_lengths, holder);
        ASSERT_TRUE(next_plan.ok());
        EXPECT_EQ(tensorToVector<int32_t>(next_plan->model_input.combo_tokens), (std::vector<int32_t>{22, 23, 24, 25}));
    }
}

TEST_F(PPBatchStreamProcessorTest, SpeculativeDispatchKeepsFailedResultsOutOfRequestHistory) {
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
    /** Failed rows retain verifier outputs; the error prevents both token and proposal commits. */
    result.new_token_ids     = intTensor({10, 11, 13, -1, 20, 21, -1, -1}).reshape({2, 4});
    result.new_token_lengths = intTensor({3, 2});
    result.propose_token_ids = intTensor({10, 11, 12, 22, 23, 24}).reshape({2, 3});
    result.request_errors[0] = ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, "verify failed");
    auto processor           = makeProcessor(SP_TYPE_MTP);
    processor.dispatchExecutionResult(StreamGroups({failed, healthy}), result);
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
    auto stream          = makeStream(101, {1, 2});
    auto buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
    buffer->propose_step = 3;
    buffer->tokens       = torch::zeros({1, 4}, torch::kInt32);
    stream->setSPOutputBuffer(buffer);
    stream->setPPInflight();
    auto result              = makeResult(std::vector<int64_t>{101});
    result.new_token_ids     = intTensor({20}).reshape({1, 1});
    result.new_token_lengths = intTensor({1});
    auto processor           = makeProcessor(SP_TYPE_DSPARK);
    processor.dispatchExecutionResult(StreamGroups({stream}), result);
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 20}));
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_EQ(stream->getSPOutputBuffer()->tokens.flatten()[0].item<int32_t>(), 20);
    EXPECT_FALSE(stream->getSPOutputBuffer()->all_probs.defined());
    EXPECT_FALSE(stream->isPPInflight());
}

TEST_F(PPBatchStreamProcessorTest, DSparkDispatchKeepsOnlyTokensForNextPlan) {
    auto first  = makeStream(101, {1, 2});
    auto second = makeStream(202, {3, 4});
    for (const auto& stream : {first, second}) {
        setProposals(stream, {9, 10, 11, 12});
        stream->setPPInflight();
    }

    auto result              = makeResult(std::vector<int64_t>{101, 202});
    result.new_token_ids     = intTensor({20, 30}).reshape({2, 1});
    result.new_token_lengths = intTensor({1, 1});
    result.propose_token_ids = intTensor({21, 22, 23, 31, 32, 33}).reshape({2, 3});

    auto                                   processor = makeProcessor(SP_TYPE_DSPARK);
    const StreamGroups                     groups({first, second});
    std::unordered_map<RequestId, int32_t> history_lengths;
    const auto                             initial_plan = makePlan(processor, groups, history_lengths);
    EXPECT_EQ(tensorToVector<int32_t>(initial_plan.sampling_plan.token_ids), (std::vector<int32_t>{1, 2, 3, 4}));
    processor.dispatchExecutionResult(groups, result);
    EXPECT_EQ(first->getProposeToken(), (std::vector<int>{20, 21, 22, 23}));
    EXPECT_EQ(second->getProposeToken(), (std::vector<int>{30, 31, 32, 33}));
    for (const auto& stream : {first, second}) {
        EXPECT_FALSE(stream->getSPOutputBuffer()->all_probs.defined());
        EXPECT_FALSE(stream->isPPInflight());
    }
    const auto next_plan = makePlan(processor, StreamGroups({first, second}), history_lengths).sampling_plan;
    EXPECT_EQ(tensorToVector<int64_t>(next_plan.request_ids), (std::vector<int64_t>{101, 202}));
    EXPECT_EQ(tensorToVector<int32_t>(next_plan.token_ids), (std::vector<int32_t>{20, 30}));
    EXPECT_EQ(tensorToVector<int64_t>(next_plan.token_offsets), (std::vector<int64_t>{0, 1, 2}));
}

TEST_F(PPBatchStreamProcessorTest, SamplingPlanAndHeadCommitShareCapBeyondNextRoundLimit) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    initCache();
    auto cfg           = config();
    cfg.max_new_tokens = 16;
    auto stream        = makeStream(101, std::vector<int32_t>(54, 1), cfg);
    commitTokens(stream, intTensor({9}).reshape({1, 1}));
    stream->setReserveStep(7);
    setProposals(stream, {9, 10, 11, 12});
    auto       processor = makeProcessor(SP_TYPE_MTP);
    const auto plan      = makePlan(processor, StreamGroups({stream})).sampling_plan;
    EXPECT_EQ(stream->nextStepSeqLengthLimit(), 57u);
    EXPECT_EQ(plan.max_tokens.item<int32_t>(), 64);
    EXPECT_EQ(plan.sequence_lengths.item<int32_t>(), 55);
    EXPECT_FALSE(stream->needFinish());
    auto result              = makeResult(plan);
    result.new_token_ids     = intTensor({10, 11, 12, 13}).reshape({1, 4});
    result.new_token_lengths = intTensor({4});
    result.propose_token_ids = intTensor({14, 15, 16}).reshape({1, 3});
    stream->setPPInflight();
    processor.dispatchExecutionResult(StreamGroups({stream}), result);
    EXPECT_EQ(stream->seqLength(), 59);
    EXPECT_EQ(stream->getProposeToken(), (std::vector<int>{13, 14, 15, 16}));
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_TRUE(stream->needFinish());
    EXPECT_FALSE(stream->isPPInflight());
}

TEST_F(PPBatchStreamProcessorTest, SpeculativeDispatchClipsRequestBudgetWithoutChangingReturnedAcceptance) {
    auto cfg           = config();
    cfg.max_new_tokens = 2;
    auto stream        = makeStream(101, {1, 2}, cfg);
    commitTokens(stream, intTensor({9}).reshape({1, 1}));
    setProposals(stream, {9, 10, 11, 12});
    stream->setPPInflight();
    auto result              = makeResult(std::vector<int64_t>{101});
    result.new_token_ids     = intTensor({10, 11, 12, 13}).reshape({1, 4});
    result.new_token_lengths = intTensor({4});
    result.propose_token_ids = intTensor({14, 15, 16}).reshape({1, 3});
    auto       processor     = makeProcessor(SP_TYPE_MTP);
    const auto plan          = makePlan(processor, StreamGroups({stream})).sampling_plan;
    EXPECT_EQ(plan.max_tokens.item<int32_t>(), 4);
    processor.dispatchExecutionResult(StreamGroups({stream}), result);
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 9, 10}));
    EXPECT_EQ(result.new_token_lengths.item<int32_t>(), 4);
    EXPECT_EQ(stream->spIterCount(), 1u);
    EXPECT_EQ(stream->speculativeAcceptedTokensPerPos(), (std::vector<int32_t>{1, 1, 1}));
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
        cache_config_            = test::makeSimpleHybridMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1);
        resources_.cache_manager = std::make_shared<KVCacheManager>(cache_config_);
        GenerateConfig cfg;
        cfg.num_return_sequences                 = 0;
        cfg.max_new_tokens                       = 20;
        cfg.do_sample                            = false;
        auto stream                              = makeStream(101, std::vector<int32_t>(c.input_len, 1), cfg);
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

        auto              processor = makeProcessor(SP_TYPE_MTP);
        PPExecutionResult result;
        result.request_ids       = torch::tensor({101}, torch::kInt64);
        result.new_token_ids     = torch::arange(10, 11 + c.draft_count, torch::kInt32).reshape({1, c.draft_count + 1});
        result.new_token_lengths = intTensor({c.num_new_tokens});
        result.propose_token_ids = torch::arange(20, 20 + c.draft_count, torch::kInt32).reshape({1, c.draft_count});
        result.request_outputs.resize(1);
        result.request_errors.resize(1);
        processor.dispatchExecutionResult(StreamGroups({stream}), result);
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
                auto       result         = makeResult(std::vector<int64_t>{101, 202});
                result.new_token_ids      = intTensor({tail_error ? 0 : 10, 20}).reshape({2, 1});
                if (tail_error) {
                    result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "tail sampling failed");
                }
                if (type != SP_TYPE_NONE) {
                    result.new_token_ids = intTensor({tail_error ? 0 : 30, 0, 0, 0, 20, -1, -1, -1}).reshape({2, 4});
                    result.new_token_lengths = intTensor({1, 1});
                    result.propose_token_ids = intTensor({11, 12, 13, 21, 22, 23}).reshape({2, 3});
                }
                auto processor = makeProcessor(type);
                processor.dispatchExecutionResult(groups, result);
                EXPECT_EQ(cancelled->statusInfo().ToString(), original_error.ToString());
                EXPECT_EQ(cancelled->completeTokenIdsVec(0),
                          type == SP_TYPE_NONE ? (std::vector<int>{1, 2}) : (std::vector<int>{1, 2, 9}));
                EXPECT_FALSE(cancelled->hasOutput());
                EXPECT_FALSE(cancelled->isPPInflight());
                if (type != SP_TYPE_NONE) {
                    EXPECT_EQ(cancelled->getProposeToken(), (std::vector<int>{9, 10, 11, 12}));
                    EXPECT_EQ(tensorToVector<int32_t>(cancelled->getSPOutputBuffer()->tokens),
                              (std::vector<int32_t>{9, 10, 11, 12}));
                }
                EXPECT_EQ(healthy->completeTokenIdsVec(0),
                          type == SP_TYPE_NONE ? (std::vector<int>{3, 4, 20}) : (std::vector<int>{3, 4, 9, 20}));
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
        auto result          = makeResult(std::vector<int64_t>{101, 202});
        result.new_token_ids = intTensor({10, 20}).reshape({2, 1});
        if (type != SP_TYPE_NONE) {
            result.new_token_ids     = intTensor({30, -1, -1, -1, 20, -1, -1, -1}).reshape({2, 4});
            result.new_token_lengths = intTensor({1, 1});
            result.propose_token_ids = intTensor({11, 12, 13, 21, 22, 23}).reshape({2, 3});
        }
        auto processor = makeProcessor(type);
        processor.dispatchExecutionResult(groups, result);
        EXPECT_EQ(first->completeTokenIdsVec(0),
                  type == SP_TYPE_NONE ? (std::vector<int>{1, 2}) : (std::vector<int>{1, 2, 9}));
        EXPECT_EQ(second->completeTokenIdsVec(0),
                  type == SP_TYPE_NONE ? (std::vector<int>{3, 4}) : (std::vector<int>{3, 4, 9}));
        for (const auto& stream : groups.allStreams()) {
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
            EXPECT_FALSE(stream->hasOutput());
            EXPECT_FALSE(stream->isPPInflight());
        }
    }
}

}  // namespace rtp_llm
