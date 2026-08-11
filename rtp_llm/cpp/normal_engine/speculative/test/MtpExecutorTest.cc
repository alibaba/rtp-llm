#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpSamplerFailureValidator.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {

using namespace std;
namespace spec = speculative;

struct MtpExecutorTestConfig {
    size_t                  max_seq_len                       = 2048;
    size_t                  vocab_size                        = 4;
    size_t                  num_layers                        = 1;
    size_t                  gen_num_per_cycle                 = 4;
    size_t                  vocab_size_override               = 0;  // 0 means use vocab_size
    size_t                  input_vocab_size_override         = 0;
    size_t                  proposal_input_vocab_size_override = 0;
    size_t                  target_embedding_size_override    = 0;
    size_t                  proposal_embedding_size_override  = 0;
};

template<typename T>
struct TestDataHolder {
    queue<T> test_data;

    T get() {
        if (test_data.empty()) {
            throw std::runtime_error("[test] Test data is empty");
        }

        T res = test_data.front();
        test_data.pop();
        return res;
    }

    void push(const T& res) {
        test_data.push(res);
    }

    void push(const vector<T>& res) {
        for (const auto& r : res) {
            test_data.push(r);
        }
    }
};

template<typename T>
vector<T> createRandomVector(size_t size, int max_val) {
    std::random_device                rd;
    std::mt19937                      gen(rd());
    std::uniform_real_distribution<T> dis(0.0, max_val);
    vector<T>                         vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// for int type, use uniform_int_distribution
vector<int> createRandomVector(size_t size, int max_val) {
    std::random_device                 rd;
    std::mt19937                       gen(rd());
    std::uniform_int_distribution<int> dis(0, max_val);
    vector<int>                        vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

void checkTensorEqual(const torch::Tensor& t1, const torch::Tensor& t2) {
    bool t1_empty = !t1.defined() || t1.numel() == 0;
    bool t2_empty = !t2.defined() || t2.numel() == 0;
    if (t1_empty && t2_empty)
        return;
    if (t1_empty || t2_empty) {
        string t1_info = t1_empty ? "t1 is empty" : "t1 size: " + to_string(t1.numel());
        string t2_info = t2_empty ? "t2 is empty" : "t2 size: " + to_string(t2.numel());
        throw std::runtime_error("[test] Tensor mismatch: " + t1_info + " " + t2_info);
    }
    auto a = t1.cpu().contiguous();
    auto b = t2.cpu().contiguous();
    EXPECT_TRUE(torch::equal(a, b)) << "Tensors are not equal:\n" << a << "\nvs\n" << b;
}

template<typename T>
vector<T> toVec(const torch::Tensor& t) {
    auto c = t.cpu().contiguous();
    return vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

template<typename T>
vector<T> catVectors(const vector<vector<T>>& vectors) {
    vector<T> result;
    for (const auto& vec : vectors) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

class FakeModel: public ModelBase {
public:
    FakeModel(const GptModelInitParams& params) {
        weights_  = params.weights;
        model_id_ = params.model_id;
    }

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        ++forward_call_count_;
        checkInputs(inputs);
        return output_holder.get();
    }

    void releaseBuffers() override {
        ++release_call_count_;
    }

    size_t forwardCallCount() const {
        return forward_call_count_;
    }

    size_t releaseCallCount() const {
        return release_call_count_;
    }

    void checkTensorField(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
        RTP_LLM_LOG_INFO("check %s", name);
        checkTensorEqual(actual, expected);
    }

    void checkInputs(const GptModelInputs& inputs) {
        GptModelInputs expected_inputs = input_holder.get();
        checkTensorField("combo_tokens", inputs.combo_tokens, expected_inputs.combo_tokens);
        checkTensorField("input_lengths", inputs.input_lengths, expected_inputs.input_lengths);
        checkTensorField("sequence_lengths", inputs.sequence_lengths, expected_inputs.sequence_lengths);
        checkTensorField("prefix_lengths", inputs.prefix_lengths, expected_inputs.prefix_lengths);
        checkTensorField("lm_output_indexes", inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
        checkTensorField("last_hidden_states", inputs.last_hidden_states, expected_inputs.last_hidden_states);
    }

    void setOutputs(const vector<GptModelOutputs>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<GptModelInputs>& inputs) {
        input_holder.push(inputs);
    }

private:
    TestDataHolder<GptModelInputs>  input_holder;
    TestDataHolder<GptModelOutputs> output_holder;
    size_t                          forward_call_count_ = 0;
    size_t                          release_call_count_ = 0;
};

class FakeFastTopKSampler: public spec::FastTopKSampler {
public:
    FakeFastTopKSampler() {}

    spec::FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1) override {
        checkInputs(logits);
        auto output = output_holder.get();
        if (output.token_ids.defined() && !output.token_ids.is_cuda()) {
            output.token_ids = output.token_ids.to(torch::kCUDA);
        }
        if (output.all_probs.defined() && !output.all_probs.is_cuda()) {
            output.all_probs = output.all_probs.to(torch::kCUDA);
        }
        return output;
    }

    void checkInputs(const torch::Tensor& logits) {
        auto expected_logits = logits_holder.get();
        RTP_LLM_LOG_INFO("check fast_topk_sampler logits");
        checkTensorEqual(logits, expected_logits);
    }

    void setOutputs(const vector<spec::FastTopKSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<torch::Tensor>& inputs) {
        logits_holder.push(inputs);
    }

private:
    TestDataHolder<torch::Tensor>               logits_holder;
    TestDataHolder<spec::FastTopKSamplerOutput> output_holder;
};

class FakeSpeculativeSampler: public spec::SpeculativeSampler {
public:
    FakeSpeculativeSampler(size_t propose_step): spec::SpeculativeSampler(propose_step) {}

    spec::SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                           SamplerOutput&                      draft_sampler_output,
                                           SamplerOutput&                      target_sampler_output) override {
        ++forward_call_count_;
        if (advance_seeded_generators_) {
            for (const auto& stream : streams) {
                auto generator = stream->getGenerator();
                if (generator.defined()) {
                    generator.set_current_seed(generator.current_seed() + 1);
                }
            }
        }
        return output_holder.get();
    }

    size_t forwardCallCount() const {
        return forward_call_count_;
    }

    void setAdvanceSeededGenerators(bool advance) {
        advance_seeded_generators_ = advance;
    }

    void checkInputs(const std::list<GenerateStreamPtr>& streams,
                     SamplerOutput&                      draft_sampler_output,
                     SamplerOutput&                      target_sampler_output) {
        auto [expected_draft_sampler_input, expected_target_sampler_input] = input_holder.get();
        RTP_LLM_LOG_INFO("check draft_sampler_output.token_ids");
        checkTensorEqual(draft_sampler_output.token_ids, expected_draft_sampler_input.token_ids);
        RTP_LLM_LOG_INFO("check draft_sampler_output.all_probs");
        checkTensorEqual(draft_sampler_output.all_probs, expected_draft_sampler_input.all_probs);
        RTP_LLM_LOG_INFO("check target_sampler_output.all_probs");
        checkTensorEqual(target_sampler_output.all_probs, expected_target_sampler_input.all_probs);
    }

    void setOutputs(const vector<spec::SpeculativeSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const pair<SamplerOutput, SamplerOutput>& inputs) {
        input_holder.push(inputs);
    }

private:
    TestDataHolder<pair<SamplerOutput, SamplerOutput>> input_holder;
    TestDataHolder<spec::SpeculativeSamplerOutput>     output_holder;
    size_t                                             forward_call_count_ = 0;
    bool                                               advance_seeded_generators_ = false;
};

class FakeSampler: public Sampler {
public:
    FakeSampler(const SamplerInitParams& params): Sampler(params) {}

    SamplerOutput forward(const SamplerInputs& inputs) override {
        checkInputs(inputs);
        if (advance_seeded_generators_) {
            for (const auto& input_generator : inputs.generator) {
                auto generator = input_generator;
                if (generator.defined()) {
                    generator.set_current_seed(generator.current_seed() + 1);
                }
            }
        }
        if (!exception_message_.empty()) {
            throw std::runtime_error(exception_message_);
        }
        return output_holder.get();
    }

    void setException(const std::string& message) {
        exception_message_ = message;
    }

    void setAdvanceSeededGenerators(bool advance) {
        advance_seeded_generators_ = advance;
    }

    void checkInputs(const SamplerInputs& inputs) {
        auto expected_inputs = input_holder.get();
        RTP_LLM_LOG_INFO("check sampler logits");
        checkTensorEqual(inputs.logits, expected_inputs.logits);
    }

    void setInputs(const vector<SamplerInputs>& inputs) {
        input_holder.push(inputs);
    }

    void setOutputs(const vector<SamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

private:
    TestDataHolder<SamplerInputs> input_holder;
    TestDataHolder<SamplerOutput> output_holder;
    std::string                   exception_message_;
    bool                          advance_seeded_generators_ = false;
};

struct MtpExecutorComponents {
    std::unique_ptr<MtpExecutor>            executor;
    std::unique_ptr<FakeModel>              fake_target_model;
    std::unique_ptr<FakeModel>              fake_draft_model;
    std::unique_ptr<FakeFastTopKSampler>    fake_fast_topk_sampler;
    std::unique_ptr<FakeSpeculativeSampler> fake_speculative_sampler;
    std::unique_ptr<FakeSampler>            fake_sampler;
    ModelConfig                             model_config;
    RuntimeConfig                           runtime_config;
    ResourceContext                         resource_context;
};

class MtpExecutorTest: public DeviceTestBase {
public:
    GenerateStreamPtr createContextStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const vector<int>&     input_ids,
                                          std::optional<int>     random_seed             = std::nullopt,
                                          bool                   return_all_hidden_states = false,
                                          std::optional<vector<int>> text_tokens_mask     = std::nullopt) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        query->generate_config->random_seed              = random_seed;
        query->generate_config->return_all_hidden_states = return_all_hidden_states;
        if (text_tokens_mask.has_value()) {
            query->text_tokens_mask = torch::tensor(*text_tokens_mask, torch::kInt32);
        }
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        return stream;
    }

    GenerateStreamPtr createDecodeStream(const ModelConfig&          model_config,
                                         const RuntimeConfig&        runtime_config,
                                         const ResourceContext&      resource_context,
                                         const vector<int>&          input_ids,
                                         const StreamSpecUpdateInfo& spec_update_info,
                                         std::optional<int>          random_seed = std::nullopt) {
        GenerateStreamPtr stream =
            createContextStream(model_config, runtime_config, resource_context, input_ids, random_seed);

        auto sp_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_buffer->tokens = torch::tensor({-1, -1}, torch::kInt32).reshape({1, 2});

        stream->setSPOutputBuffer(sp_buffer);
        stream->specUpdate(spec_update_info);
        if (sp_buffer->all_probs.defined() && !sp_buffer->all_probs.is_cuda()) {
            sp_buffer->all_probs = sp_buffer->all_probs.to(torch::kCUDA);
        }
        if (sp_buffer->hidden_states.defined() && !sp_buffer->hidden_states.is_cuda()) {
            sp_buffer->hidden_states = sp_buffer->hidden_states.to(torch::kCUDA);
        }
        return stream;
    }

    void checkOutput(const GenerateStreamPtr& stream,
                     const vector<int>&       expect_token_ids,
                     const vector<int>&       expect_propose_tokens,
                     const vector<float>&     expect_all_probs,
                     const vector<float>&     expect_last_hidden_states) {
        auto token_ids = stream->getCompleteTokenIds()->completeTokenIdsVec(0);
        EXPECT_EQ(expect_token_ids, token_ids);

        auto sp_output_buffer = stream->getSPOutputBuffer();
        auto tokens           = sp_output_buffer->tokens;
        auto tokens_h         = tokens.cpu().clone();
        EXPECT_EQ(expect_propose_tokens, toVec<int>(tokens_h));

        auto all_probs   = sp_output_buffer->all_probs;
        auto all_probs_h = all_probs.is_cuda() ? all_probs.cpu() : all_probs;
        EXPECT_EQ(expect_all_probs, toVec<float>(all_probs_h));

        if (expect_last_hidden_states.size() > 0) {
            auto last_hidden_states   = sp_output_buffer->hidden_states;
            auto last_hidden_states_h = last_hidden_states.is_cuda() ? last_hidden_states.cpu() : last_hidden_states;
            EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states_h));
        } else {
            EXPECT_TRUE(!sp_output_buffer->hidden_states.defined());
        }
    }

    MtpExecutorComponents createMtpExecutorComponents(const MtpExecutorTestConfig& test_config) {
        CustomConfig               config;
        ModelConfig                model_config;
        RuntimeConfig              runtime_config;
        KVCacheConfig              kv_cache_config;
        ResourceContext            resource_context;
        SpeculativeExecutionConfig sp_config;

        model_config.max_seq_len    = test_config.max_seq_len;
        model_config.vocab_size     = test_config.vocab_size;
        model_config.num_layers     = test_config.num_layers;
        sp_config.gen_num_per_cycle = test_config.gen_num_per_cycle;

        resource_context.cache_manager =
            std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                                            /*block_num=*/10,
                                                                            /*tokens_per_block=*/2,
                                                                            rtp_llm::TYPE_INT8,
                                                                            /*local_head_num_kv=*/128,
                                                                            /*size_per_head=*/256));

        auto cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                           /*block_num=*/10,
                                                           /*tokens_per_block=*/2,
                                                           rtp_llm::TYPE_INT8,
                                                           /*local_head_num_kv=*/128,
                                                           /*size_per_head=*/256);

        auto mtp_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                         /*block_num=*/10,
                                                         /*tokens_per_block=*/2,
                                                         rtp_llm::TYPE_INT8,
                                                         /*local_head_num_kv=*/128,
                                                         /*size_per_head=*/256);
        cache_config.mtp_sub_configs.push_back(std::make_shared<CacheConfig>(mtp_config));

        EngineInitParams params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
        params.sp_config        = sp_config;
        if (test_config.vocab_size_override > 0) {
            params.model_config_.vocab_size = test_config.vocab_size_override;
        }
        if (test_config.input_vocab_size_override > 0) {
            params.model_config_.input_vocab_size = test_config.input_vocab_size_override;
            model_config.input_vocab_size         = test_config.input_vocab_size_override;
        }
        if (test_config.target_embedding_size_override > 0) {
            params.model_config_.embedding_size = test_config.target_embedding_size_override;
            model_config.embedding_size         = test_config.target_embedding_size_override;
        }

        // Create propose model engine init params
        auto mtp_model_params   = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        auto mtp_params         = std::make_unique<EngineInitParams>(params);
        mtp_params->model_id    = 1;
        mtp_params->model_config_.num_layers = 1;
        if (test_config.proposal_input_vocab_size_override > 0) {
            mtp_params->model_config_.input_vocab_size = test_config.proposal_input_vocab_size_override;
        }
        if (test_config.proposal_embedding_size_override > 0) {
            mtp_params->model_config_.embedding_size = test_config.proposal_embedding_size_override;
        }
        mtp_params->gpt_weights.layers.resize(1);
        mtp_params->py_sp_model = py::none();

        mtp_model_params->push_back(std::move(mtp_params));

        auto propose_params = std::make_unique<ProposeModelEngineInitParams>(
            SP_TYPE_MTP, sp_config.gen_num_per_cycle, std::move(mtp_model_params));

        // Create cache managers
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();

        // Create MtpExecutor
        auto executor = std::make_unique<MtpExecutor>(params, propose_params, cache_manager);
        const auto& draft_params = propose_params->getMtpEngineInitParams();

        // Create fake models
        GptModelInitParams target_model_params(
            {params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id,
             params.parallelism_config});

        GptModelInitParams draft_model_params(
            {draft_params.gpt_weights,
             Executor::genModelDescription(
                 draft_params.model_config_,
                 draft_params.parallelism_config,
                 draft_params.eplb_config,
                 draft_params.moe_config),
             std::nullopt,
             draft_params.model_id,
             draft_params.parallelism_config});

        auto fake_target_model        = std::make_unique<FakeModel>(target_model_params);
        auto fake_draft_model         = std::make_unique<FakeModel>(draft_model_params);
        auto fake_fast_topk_sampler   = std::make_unique<FakeFastTopKSampler>();
        auto fake_speculative_sampler = std::make_unique<FakeSpeculativeSampler>(sp_config.gen_num_per_cycle);
        auto fake_sampler             = std::make_unique<FakeSampler>(SamplerInitParams{});

        MtpExecutorComponents components;
        components.executor                 = std::move(executor);
        components.fake_target_model        = std::move(fake_target_model);
        components.fake_draft_model         = std::move(fake_draft_model);
        components.fake_fast_topk_sampler   = std::move(fake_fast_topk_sampler);
        components.fake_speculative_sampler = std::move(fake_speculative_sampler);
        components.fake_sampler             = std::move(fake_sampler);
        components.model_config             = model_config;
        components.runtime_config           = runtime_config;
        components.resource_context         = resource_context;

        return components;
    }

    void setupFakeModels(MtpExecutor*                            executor,
                         std::unique_ptr<FakeModel>              fake_target_model,
                         std::unique_ptr<FakeModel>              fake_draft_model,
                         std::unique_ptr<FakeFastTopKSampler>    fake_fast_topk_sampler,
                         std::unique_ptr<FakeSpeculativeSampler> fake_speculative_sampler,
                         std::unique_ptr<FakeSampler>            fake_sampler) {
        executor->setTargetModel(std::move(fake_target_model));
        executor->setDraftModel(std::move(fake_draft_model));
        executor->setFastTopKSampler(std::move(fake_fast_topk_sampler));
        executor->setSpeculativeSampler(std::move(fake_speculative_sampler));
        executor->setSampler(std::move(fake_sampler));
    }

    GptModelOutputs createRandomGptModelOutputs(size_t token_num, size_t vocab_size, size_t hidden_size) {
        auto output              = GptModelOutputs{};
        output.logits            = torch::rand({(int64_t)token_num, (int64_t)vocab_size}, torch::kFloat32);
        output.all_hidden_states = torch::rand({(int64_t)token_num, (int64_t)hidden_size}, torch::kFloat32);
        return output;
    }
};

TEST_F(MtpExecutorTest, testSingleBatchPrefill) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 4;
    auto components               = createMtpExecutorComponents(test_config);

    size_t batch_size = 1;

    // Create context stream
    GenerateStreamPtr stream1 = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});

    // set fake model outputs
    auto target_input  = GptModelInputs{};
    auto target_output = GptModelOutputs{};

    // set fake target model inputs
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);
    target_output.logits           = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({(int64_t)batch_size, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake draft model outputs
    auto draft_input               = GptModelInputs{};
    auto draft_output              = GptModelOutputs{};
    draft_input.combo_tokens       = torch::tensor({1, 2, 3, 1}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({4}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({0}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({3}, torch::kInt32);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits            = torch::tensor({0.5f, 0.6f, 0.7f, 0.8f}).reshape({(int64_t)batch_size, 4});
    draft_output.all_hidden_states =
        torch::tensor({0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f}).reshape({4, 2});

    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1}, torch::kInt32).reshape({(int64_t)batch_size, 1})};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake fast topk sampler outputs
    auto fast_topk_sampler_output =
        spec::FastTopKSamplerOutput{torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({(int64_t)batch_size, 4}),
                                    torch::tensor({2}, torch::kInt32).reshape({(int64_t)batch_size, 1})};
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({fast_topk_sampler_output});

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    // Verify executor was created successfully
    auto status = components.executor->process({stream1});
    ASSERT_TRUE(status.ok());

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {0.17, 0.18});
}

TEST_F(MtpExecutorTest, testMultiBatchPrefill) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 4;
    auto components               = createMtpExecutorComponents(test_config);

    size_t batch_size = 2;

    // Create context stream
    GenerateStreamPtr stream1 = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
    GenerateStreamPtr stream2 =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {2, 3});

    // set fake model outputs
    auto target_input  = GptModelInputs{};
    auto target_output = GptModelOutputs{};

    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0, 0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3, 5}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f}).reshape({(int64_t)batch_size, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 1.01f, 1.02f, 1.03f, 1.04f})
            .reshape({6, 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake draft model inputs
    auto draft_input  = GptModelInputs{};
    auto draft_output = GptModelOutputs{};

    draft_input.combo_tokens       = torch::tensor({1, 2, 3, 1, 3, 0}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({4, 2}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({0, 0}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({3, 5}, torch::kInt32);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits =
        torch::tensor({0.5f, 0.6f, 0.7f, 0.8f, 1.5f, 1.6f, 1.7f, 1.8f}).reshape({(int64_t)batch_size, 4});
    draft_output.all_hidden_states =
        torch::tensor({0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 1.11f, 1.12f, 1.13f, 1.14f})
            .reshape({6, 2});

    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1, 0}, torch::kInt32).reshape({(int64_t)batch_size, 1})};

    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake fast topk sampler inputs
    auto fast_topk_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({(int64_t)batch_size, 4}),
        torch::tensor({2, 1}, torch::kInt32).reshape({(int64_t)batch_size, 1})};

    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({fast_topk_sampler_output});

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    // Verify executor was created successfully
    auto status = components.executor->process({stream1, stream2});
    ASSERT_TRUE(status.ok());

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {0.17, 0.18});
    checkOutput(stream2, {2, 3, 0}, {0, 1}, {0.0, 0.0, 1.0, 0.0}, {1.13, 1.14});
}

TEST_F(MtpExecutorTest, prefillSamplerFailureAbortsBeforeDraftForward) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 4;
    auto components               = createMtpExecutorComponents(test_config);

    auto stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
    const auto tokens_before = stream->getCompleteTokenIds()->completeTokenIdsVec(0);

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1}, torch::kInt32).reshape({1, 1});
    sampler_output.success   = torch::tensor({false}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->getCompleteTokenIds()->completeTokenIdsVec(0), tokens_before);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
    EXPECT_GE(target_model->releaseCallCount(), 2);
    EXPECT_GE(draft_model->releaseCallCount(), 2);
}

TEST_F(MtpExecutorTest, prefillInputVocabViolationKeepsValidPeerRetryable) {
    const auto verify_isolated = [&](size_t target_input_vocab_size, size_t proposal_input_vocab_size) {
        MtpExecutorTestConfig test_config;
        test_config.vocab_size                         = 4;
        test_config.vocab_size_override                = 4;
        test_config.input_vocab_size_override          = target_input_vocab_size;
        test_config.proposal_input_vocab_size_override = proposal_input_vocab_size;
        auto components                                = createMtpExecutorComponents(test_config);

        auto valid_stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {1, 2});
        auto invalid_stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 3});
        const auto valid_tokens_before = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
        const auto valid_iter_before   = valid_stream->iterCount();
        auto* target_model             = components.fake_target_model.get();
        auto* draft_model              = components.fake_draft_model.get();
        setupFakeModels(components.executor.get(),
                        std::move(components.fake_target_model),
                        std::move(components.fake_draft_model),
                        std::move(components.fake_fast_topk_sampler),
                        std::move(components.fake_speculative_sampler),
                        std::move(components.fake_sampler));

        absl::Status status;
        EXPECT_NO_THROW(status = components.executor->process({valid_stream, invalid_stream}));
        EXPECT_TRUE(status.ok()) << status;
        EXPECT_FALSE(valid_stream->hasError());
        EXPECT_TRUE(invalid_stream->hasError());
        EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
        EXPECT_EQ(valid_stream->iterCount(), valid_iter_before);
        EXPECT_EQ(target_model->forwardCallCount(), 0);
        EXPECT_EQ(draft_model->forwardCallCount(), 0);
    };

    verify_isolated(/*target_input_vocab_size=*/3, /*proposal_input_vocab_size=*/4);
    verify_isolated(/*target_input_vocab_size=*/4, /*proposal_input_vocab_size=*/3);
}

TEST_F(MtpExecutorTest, sampledTokenInputVocabGuardOnlyRunsForNarrowInputVocabulary) {
    MtpExecutorTestConfig equal_vocab_config;
    equal_vocab_config.vocab_size_override                = 4;
    equal_vocab_config.input_vocab_size_override          = 4;
    equal_vocab_config.proposal_input_vocab_size_override = 4;
    auto equal_vocab_components                           = createMtpExecutorComponents(equal_vocab_config);
    EXPECT_FALSE(equal_vocab_components.executor->sampled_token_input_vocab_guard_required_);

    auto narrow_proposal_config = equal_vocab_config;
    narrow_proposal_config.proposal_input_vocab_size_override = 3;
    auto narrow_proposal_components = createMtpExecutorComponents(narrow_proposal_config);
    EXPECT_TRUE(narrow_proposal_components.executor->sampled_token_input_vocab_guard_required_);

    auto narrow_target_config = equal_vocab_config;
    narrow_target_config.input_vocab_size_override = 3;
    auto narrow_target_components = createMtpExecutorComponents(narrow_target_config);
    EXPECT_TRUE(narrow_target_components.executor->sampled_token_input_vocab_guard_required_);

    auto narrow_target_embedding_config = equal_vocab_config;
    narrow_target_embedding_config.input_vocab_size_override      = 0;
    narrow_target_embedding_config.target_embedding_size_override = 3;
    auto narrow_target_embedding_components = createMtpExecutorComponents(narrow_target_embedding_config);
    EXPECT_EQ(narrow_target_embedding_components.executor->target_input_vocab_size_, 3);
    EXPECT_TRUE(narrow_target_embedding_components.executor->sampled_token_input_vocab_guard_required_);

    auto narrow_proposal_embedding_config = equal_vocab_config;
    narrow_proposal_embedding_config.proposal_input_vocab_size_override = 0;
    narrow_proposal_embedding_config.proposal_embedding_size_override   = 3;
    auto narrow_proposal_embedding_components = createMtpExecutorComponents(narrow_proposal_embedding_config);
    EXPECT_EQ(narrow_proposal_embedding_components.executor->propose_input_vocab_size_, 3);
    EXPECT_TRUE(narrow_proposal_embedding_components.executor->sampled_token_input_vocab_guard_required_);
}

TEST_F(MtpExecutorTest, prefillNegativeTokenAbortsBeforeModelForward) {
    MtpExecutorTestConfig test_config;
    test_config.vocab_size          = 4;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);
    auto stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, -1});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->iterCount(), 0);
    EXPECT_EQ(target_model->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
}

TEST_F(MtpExecutorTest, prefillRejectsNonTextTokensBeforeModelForward) {
    MtpExecutorTestConfig test_config;
    auto components = createMtpExecutorComponents(test_config);
    auto stream = createContextStream(components.model_config,
                                      components.runtime_config,
                                      components.resource_context,
                                      {0, -1},
                                      std::nullopt,
                                      false,
                                      std::vector<int>{1, 0});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(stream->iterCount(), 0);
    EXPECT_EQ(target_model->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
}

TEST_F(MtpExecutorTest, prefillSampledTokenOutsideProposalInputVocabIsolatesInvalidStream) {
    MtpExecutorTestConfig test_config;
    test_config.vocab_size_override                = 4;
    test_config.input_vocab_size_override          = 4;
    test_config.proposal_input_vocab_size_override = 3;
    auto components                                = createMtpExecutorComponents(test_config);
    auto invalid_stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1});
    auto valid_stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {1, 2}, 123);
    const auto invalid_tokens_before = invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_tokens_before   = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_generator_state_before = valid_stream->getGenerator().get_state().clone();

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 1, 2}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0, 0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({1, 3}, torch::kInt32);
    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({3, 1}, torch::kInt32).reshape({2, 1});
    sampler_output.success   = torch::tensor({true, true}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output});
    components.fake_sampler->setAdvanceSeededGenerators(true);

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({invalid_stream, valid_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::OUT_OF_VOCAB_RANGE);
    EXPECT_FALSE(valid_stream->hasError());
    EXPECT_EQ(invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), invalid_tokens_before);
    EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
    EXPECT_EQ(invalid_stream->iterCount(), 0);
    EXPECT_EQ(valid_stream->iterCount(), 0);
    EXPECT_TRUE(torch::equal(valid_stream->getGenerator().get_state(), valid_generator_state_before));
    EXPECT_EQ(target_model->forwardCallCount(), 1);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
}

TEST_F(MtpExecutorTest, prefillDraftTokenOutsideInputVocabIsolatesInvalidStream) {
    MtpExecutorTestConfig test_config;
    test_config.vocab_size_override                = 4;
    test_config.input_vocab_size_override          = 3;
    test_config.proposal_input_vocab_size_override = 3;
    auto components                                = createMtpExecutorComponents(test_config);
    auto invalid_stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1});
    auto valid_stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {1, 2}, 123);
    const auto invalid_tokens_before = invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_tokens_before   = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_generator_state_before = valid_stream->getGenerator().get_state().clone();

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 1, 2}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0, 0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({1, 3}, torch::kInt32);
    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    GptModelInputs draft_input;
    draft_input.combo_tokens       = torch::tensor({1, 1, 2, 1}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({2, 2}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({0, 0}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({1, 3}, torch::kInt32);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    GptModelOutputs draft_output;
    draft_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    draft_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1, 1}, torch::kInt32).reshape({2, 1});
    sampler_output.success   = torch::tensor({true, true}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output});
    components.fake_sampler->setAdvanceSeededGenerators(true);

    spec::FastTopKSamplerOutput draft_sampler_output;
    draft_sampler_output.token_ids = torch::tensor({3, 1}, torch::kInt64).reshape({2, 1});
    draft_sampler_output.all_probs = torch::zeros({2, 4}, torch::kFloat32);
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({invalid_stream, valid_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::OUT_OF_VOCAB_RANGE);
    EXPECT_FALSE(valid_stream->hasError());
    EXPECT_EQ(invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), invalid_tokens_before);
    EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
    EXPECT_EQ(invalid_stream->iterCount(), 0);
    EXPECT_EQ(valid_stream->iterCount(), 0);
    EXPECT_TRUE(torch::equal(valid_stream->getGenerator().get_state(), valid_generator_state_before));
    EXPECT_EQ(target_model->forwardCallCount(), 1);
    EXPECT_EQ(draft_model->forwardCallCount(), 1);
}

TEST_F(MtpExecutorTest, prefillSamplerFailureKeepsSuccessfulPeerRetryable) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 4;
    auto components               = createMtpExecutorComponents(test_config);

    auto failed_stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
    auto successful_stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {2, 3}, 123, true);
    const auto failed_tokens_before = failed_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto successful_tokens_before = successful_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto successful_generator_state_before = successful_stream->getGenerator().get_state().clone();

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0, 0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3, 5}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({6, 2}, torch::kFloat32);

    GptModelInputs retry_target_input;
    retry_target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    retry_target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    retry_target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    retry_target_input.lm_output_indexes = torch::tensor({1}, torch::kInt32);

    GptModelOutputs retry_target_output;
    retry_target_output.logits            = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    retry_target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});
    components.fake_target_model->setInputs({target_input, retry_target_input});
    components.fake_target_model->setOutputs({target_output, retry_target_output});

    GptModelInputs retry_draft_input;
    retry_draft_input.combo_tokens       = torch::tensor({3, 1}, torch::kInt32);
    retry_draft_input.input_lengths      = torch::tensor({2}, torch::kInt32);
    retry_draft_input.prefix_lengths     = torch::tensor({0}, torch::kInt32);
    retry_draft_input.lm_output_indexes  = torch::tensor({1}, torch::kInt32);
    retry_draft_input.last_hidden_states = retry_target_output.all_hidden_states;

    GptModelOutputs retry_draft_output;
    retry_draft_output.logits            = torch::tensor({0.5f, 0.6f, 0.7f, 0.8f}).reshape({1, 4});
    retry_draft_output.all_hidden_states = torch::tensor({0.11f, 0.12f, 0.13f, 0.14f}).reshape({2, 2});
    components.fake_draft_model->setInputs({retry_draft_input});
    components.fake_draft_model->setOutputs({retry_draft_output});

    components.fake_sampler->setInputs(
        {SamplerInputs{target_output.logits}, SamplerInputs{retry_target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1, 0}, torch::kInt32).reshape({2, 1});
    sampler_output.success   = torch::tensor({false, true}, torch::kBool);
    SamplerOutput retry_sampler_output;
    retry_sampler_output.token_ids = torch::tensor({1}, torch::kInt32).reshape({1, 1});
    retry_sampler_output.success   = torch::tensor({true}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output, retry_sampler_output});
    components.fake_sampler->setAdvanceSeededGenerators(true);

    auto retry_fast_topk_output =
        spec::FastTopKSamplerOutput{torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}),
                                    torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({retry_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({retry_fast_topk_output});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({failed_stream, successful_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(failed_stream->hasError());
    EXPECT_FALSE(successful_stream->hasError());
    EXPECT_EQ(failed_stream->getCompleteTokenIds()->completeTokenIdsVec(0), failed_tokens_before);
    EXPECT_EQ(successful_stream->getCompleteTokenIds()->completeTokenIdsVec(0), successful_tokens_before);
    EXPECT_TRUE(torch::equal(successful_stream->getGenerator().get_state(), successful_generator_state_before));
    EXPECT_EQ(successful_stream->iterCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);

    ASSERT_NO_THROW(status = components.executor->process({successful_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_FALSE(successful_stream->hasError());
    EXPECT_EQ(target_model->forwardCallCount(), 2);
    EXPECT_EQ(draft_model->forwardCallCount(), 1);
    EXPECT_EQ(successful_stream->iterCount(), 1);
    checkOutput(successful_stream, {2, 3, 1}, {1, 2}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.13f, 0.14f});

    auto all_hidden_states = torch::tensor({{0.21f, 0.22f}, {0.23f, 0.24f}});
    StreamUpdateInfo output_info{torch::Tensor(),
                                 0,
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 all_hidden_states,
                                 false};
    auto normal_stream = std::dynamic_pointer_cast<NormalGenerateStream>(successful_stream);
    ASSERT_NE(normal_stream, nullptr);
    auto generated_output = normal_stream->prepareGenerateOutput(output_info);
    ASSERT_EQ(generated_output.generate_outputs.size(), 1);
    ASSERT_TRUE(generated_output.generate_outputs[0].all_hidden_states.has_value());
    EXPECT_TRUE(torch::equal(*generated_output.generate_outputs[0].all_hidden_states, all_hidden_states));
}

TEST_F(MtpExecutorTest, testSingleBatchDecode) {
    // test single batch decode accept partial
    // input [0, 1, 2] + [3]
    // darft [3] + [2, 1, 3]
    // verify [3, 2, 0, 0, 0]
    // accept [3, 2, 0]
    // next draft [1]
    size_t propose_step = 4;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);

    size_t batch_size = 1;

    auto stream1_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto stream1_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto stream1_draft_token_probs = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}});

    StreamSpecUpdateInfo spec_update_info1{stream1_new_tokens, 1, 3, stream1_hidden_states, stream1_draft_token_probs};

    GenerateStreamPtr stream1 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info1);

    // set 3 step draft model outputs
    auto draft_input_1  = GptModelInputs{};
    auto draft_input_2  = GptModelInputs{};
    auto draft_input_3  = GptModelInputs{};
    auto draft_output_1 = createRandomGptModelOutputs(1, 4, 2);
    auto draft_output_2 = createRandomGptModelOutputs(1, 4, 2);
    auto draft_output_3 = createRandomGptModelOutputs(1, 4, 2);

    draft_input_1.combo_tokens       = torch::tensor({3}, torch::kInt32);
    draft_input_1.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_1.sequence_lengths   = torch::tensor({2}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_1.last_hidden_states = stream1_hidden_states;

    draft_input_2.combo_tokens       = torch::tensor({2}, torch::kInt32);
    draft_input_2.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_2.sequence_lengths   = torch::tensor({3}, torch::kInt32);
    draft_input_2.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = torch::tensor({1}, torch::kInt32);
    draft_input_3.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_3.sequence_lengths   = torch::tensor({4}, torch::kInt32);
    draft_input_3.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_output.logits            = torch::tensor({1.9f, 1.10f, 1.11f, 1.12f}).reshape({(int64_t)batch_size, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.1f, 0.1f, 0.2f, 0.22f, 0.3f, 0.33f}).reshape({3, 2});

    next_draft_input.combo_tokens       = torch::tensor({3, 2, 0}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({3}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({2}, torch::kInt32);
    next_draft_input.last_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f}).reshape({3, 2});

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

    // set fake model outputs
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3, 2, 1, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({5}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3, 4}, torch::kInt32);

    target_output.logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f, 2.1f, 2.2f,
                                          2.3f, 2.4f, 3.1f, 3.2f, 3.3f, 3.4f, 4.1f, 4.2f, 4.3f, 4.4f})
                               .reshape({(int64_t)(batch_size * (propose_step + 1)), 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f})
            .reshape({(int64_t)(propose_step + 1), 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake sampler outputs
    auto target_sample_all_probs_data = createRandomVector<float>(batch_size * (propose_step + 1) * vocab_size, 1);
    auto sampler_input                = SamplerInputs{target_output.logits};
    auto sampler_output =
        SamplerOutput{torch::tensor({3, 2, 0, 0, 0}, torch::kInt32).reshape({(int64_t)batch_size, 5})};
    sampler_output.all_probs = torch::tensor(target_sample_all_probs_data)
                                   .reshape({(int64_t)batch_size, (int64_t)(propose_step + 1), (int64_t)vocab_size});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // draft sampler output [2, 1, 3, 0]
    auto draft_sampler_output_1    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_2    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_3    = spec::FastTopKSamplerOutput{};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{};

    draft_sampler_output_1.token_ids    = torch::tensor({2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_1.all_probs    = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_2.token_ids    = torch::tensor({1}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_2.all_probs    = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_3.token_ids    = torch::tensor({3}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_3.all_probs    = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});
    next_draft_sampler_output.token_ids = torch::tensor({1}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    next_draft_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});

    components.fake_fast_topk_sampler->setInputs(
        {draft_output_1.logits, draft_output_2.logits, draft_output_3.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3, next_draft_sampler_output});

    // set fake speculative sampler outputs
    auto accept_tokens              = torch::tensor({{3, 2, 0}}, torch::kInt32);
    auto speculative_sampler_output = spec::SpeculativeSamplerOutput{{accept_tokens}, {3}};
    auto draft_spec_sample_input    = SamplerOutput{};
    auto target_spec_sample_input   = SamplerOutput{};

    vector<vector<float>> draft_all_probs_list;
    draft_all_probs_list.push_back(toVec<float>(stream1_draft_token_probs));
    draft_all_probs_list.push_back(toVec<float>(draft_output_1.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_2.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_3.logits));
    draft_spec_sample_input.token_ids  = torch::tensor({3, 2, 1, 3}, torch::kInt32).reshape({1, 4});
    draft_spec_sample_input.all_probs  = torch::tensor(catVectors(draft_all_probs_list)).reshape({4, 4});
    target_spec_sample_input.all_probs = draft_spec_sample_input.all_probs;

    components.fake_speculative_sampler->setInputs({draft_spec_sample_input, target_spec_sample_input});
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    // Verify executor was created successfully
    auto status = components.executor->process({stream1});
    ASSERT_TRUE(status.ok());

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 2, 0}, {0, 1}, {0.0, 1.0, 0.0, 0.0}, {0.3, 0.33});
}

TEST_F(MtpExecutorTest, decodeSamplerExceptionAbortsAndCleansBuffers) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = 1;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);

    auto new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto draft_token_probs = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}});
    StreamSpecUpdateInfo spec_update_info{new_tokens, 1, 3, hidden_states, draft_token_probs};
    auto stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    const auto tokens_before = stream->getCompleteTokenIds()->completeTokenIdsVec(0);

    auto sp_output_buffer = stream->getSPOutputBuffer();
    sp_output_buffer->tensors_holder = {
        draft_token_probs.clone(), torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({2, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    components.fake_sampler->setException("target sampler failed");

    auto* target_model        = components.fake_target_model.get();
    auto* draft_model         = components.fake_draft_model.get();
    auto* speculative_sampler = components.fake_speculative_sampler.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->getCompleteTokenIds()->completeTokenIdsVec(0), tokens_before);
    EXPECT_EQ(speculative_sampler->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
    EXPECT_TRUE(sp_output_buffer->tensors_holder.empty());
    EXPECT_GE(target_model->releaseCallCount(), 2);
    EXPECT_GE(draft_model->releaseCallCount(), 2);
}

TEST_F(MtpExecutorTest, decodeMalformedTensorHolderAbortsBeforeModelForward) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = 1;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);

    auto stream = createDecodeStream(components.model_config,
                                     components.runtime_config,
                                     components.resource_context,
                                     {0, 1},
                                     {torch::tensor({{2}}, torch::kInt32),
                                      1,
                                      3,
                                      torch::tensor({{0.03f, 0.04f}}),
                                      torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
    auto sp_output_buffer = stream->getSPOutputBuffer();
    sp_output_buffer->tensors_holder = {sp_output_buffer->all_probs.cpu().clone(),
                                        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16)),
                                        torch::zeros({1})};

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(stream->hasError());
    EXPECT_TRUE(sp_output_buffer->tensors_holder.empty());
    EXPECT_EQ(target_model->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
}

TEST_F(MtpExecutorTest, decodeInvalidProposalTokensAbortBeforeModelForward) {
    const auto verify_rejected = [&](int first_token, int draft_token) {
        MtpExecutorTestConfig test_config;
        test_config.gen_num_per_cycle   = 1;
        test_config.vocab_size_override = 4;
        auto components                 = createMtpExecutorComponents(test_config);

        auto stream = createDecodeStream(components.model_config,
                                         components.runtime_config,
                                         components.resource_context,
                                         {0, 1},
                                         {torch::tensor({{2}}, torch::kInt32),
                                          1,
                                          3,
                                          torch::tensor({{0.03f, 0.04f}}),
                                          torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
        auto output               = stream->getSPOutputBuffer();
        output->tokens            = torch::tensor({first_token, draft_token}, torch::kInt32).reshape({1, 2});
        output->tensors_holder    = {output->all_probs.cpu().clone(),
                                     torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
        auto* const target_model  = components.fake_target_model.get();
        auto* const draft_model   = components.fake_draft_model.get();
        setupFakeModels(components.executor.get(),
                        std::move(components.fake_target_model),
                        std::move(components.fake_draft_model),
                        std::move(components.fake_fast_topk_sampler),
                        std::move(components.fake_speculative_sampler),
                        std::move(components.fake_sampler));

        absl::Status status;
        EXPECT_NO_THROW(status = components.executor->process({stream}));
        EXPECT_TRUE(status.ok()) << status;
        EXPECT_TRUE(stream->hasError());
        EXPECT_TRUE(output->tensors_holder.empty());
        EXPECT_EQ(target_model->forwardCallCount(), 0);
        EXPECT_EQ(draft_model->forwardCallCount(), 0);
    };

    verify_rejected(/*first_token=*/1, /*draft_token=*/3);
    verify_rejected(/*first_token=*/2, /*draft_token=*/-1);
    verify_rejected(/*first_token=*/2, /*draft_token=*/4);
}

TEST_F(MtpExecutorTest, decodeInvalidSpeculativeTensorContractAbortsBeforeModelForward) {
    const auto verify_rejected = [&](const auto& mutate_output) {
        MtpExecutorTestConfig test_config;
        test_config.gen_num_per_cycle   = 4;
        test_config.vocab_size_override = 4;
        auto components                 = createMtpExecutorComponents(test_config);

        auto stream = createDecodeStream(components.model_config,
                                         components.runtime_config,
                                         components.resource_context,
                                         {0, 1},
                                         {torch::tensor({{2}}, torch::kInt32),
                                          1,
                                          3,
                                          torch::tensor({{0.03f, 0.04f}}),
                                          torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
        auto output = stream->getSPOutputBuffer();
        mutate_output(*output);

        auto* const target_model = components.fake_target_model.get();
        auto* const draft_model  = components.fake_draft_model.get();
        setupFakeModels(components.executor.get(),
                        std::move(components.fake_target_model),
                        std::move(components.fake_draft_model),
                        std::move(components.fake_fast_topk_sampler),
                        std::move(components.fake_speculative_sampler),
                        std::move(components.fake_sampler));

        absl::Status status;
        EXPECT_NO_THROW(status = components.executor->process({stream}));
        EXPECT_TRUE(status.ok()) << status;
        EXPECT_TRUE(stream->hasError());
        EXPECT_TRUE(output->tensors_holder.empty());
        EXPECT_EQ(target_model->forwardCallCount(), 0);
        EXPECT_EQ(draft_model->forwardCallCount(), 0);
    };

    verify_rejected([](SpeculativeExecutorStreamOutput& output) { output.all_probs = output.all_probs.cpu(); });
    verify_rejected([](SpeculativeExecutorStreamOutput& output) { output.hidden_states = output.hidden_states.cpu(); });
    verify_rejected([](SpeculativeExecutorStreamOutput& output) {
        output.all_probs = output.all_probs.to(torch::kFloat16);
    });
    verify_rejected([](SpeculativeExecutorStreamOutput& output) {
        output.hidden_states = torch::zeros(
            {1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    });
    verify_rejected([](SpeculativeExecutorStreamOutput& output) {
        output.tensors_holder = {output.all_probs.clone(), output.hidden_states.clone()};
    });
}

TEST_F(MtpExecutorTest, decodeMalformedTensorHolderKeepsValidPeerHandoffForRetry) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = 1;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);

    auto make_stream = [&](const std::vector<int>& input_ids, int token, int draft_token) {
        return createDecodeStream(components.model_config,
                                  components.runtime_config,
                                  components.resource_context,
                                  input_ids,
                                  {torch::tensor({{token}}, torch::kInt32),
                                   1,
                                   draft_token,
                                   torch::tensor({{0.03f, 0.04f}}),
                                   torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
    };
    auto invalid_stream = make_stream({0, 1}, 2, 3);
    auto valid_stream   = make_stream({3, 2}, 1, 0);
    const auto valid_tokens_before = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);

    auto invalid_output = invalid_stream->getSPOutputBuffer();
    auto valid_output   = valid_stream->getSPOutputBuffer();
    invalid_output->tensors_holder = {invalid_output->all_probs.cpu().clone(),
                                      torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16)),
                                      torch::zeros({1})};
    valid_output->tensors_holder = {valid_output->all_probs.cpu().clone(),
                                    torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
    const auto valid_probs_before  = valid_output->tensors_holder[0].clone();
    const auto valid_hidden_before = valid_output->tensors_holder[1].clone();

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({invalid_stream, valid_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_FALSE(valid_stream->hasError());
    EXPECT_TRUE(invalid_output->tensors_holder.empty());
    ASSERT_EQ(valid_output->tensors_holder.size(), 2);
    EXPECT_TRUE(torch::equal(valid_output->tensors_holder[0], valid_probs_before));
    EXPECT_TRUE(torch::equal(valid_output->tensors_holder[1], valid_hidden_before));
    EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
    EXPECT_EQ(target_model->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
}

TEST_F(MtpExecutorTest, decodeInputVocabViolationKeepsValidPeerRetryable) {
    const auto verify_isolated = [&](size_t target_input_vocab_size,
                                     size_t proposal_input_vocab_size,
                                     int    invalid_token,
                                     int    invalid_draft_token) {
        MtpExecutorTestConfig test_config;
        test_config.vocab_size_override                = 4;
        test_config.input_vocab_size_override          = target_input_vocab_size;
        test_config.proposal_input_vocab_size_override = proposal_input_vocab_size;
        test_config.gen_num_per_cycle                  = 1;
        auto components                                = createMtpExecutorComponents(test_config);

        auto make_stream = [&](const std::vector<int>& input_ids, int token, int draft_token) {
            return createDecodeStream(components.model_config,
                                      components.runtime_config,
                                      components.resource_context,
                                      input_ids,
                                      {torch::tensor({{token}}, torch::kInt32),
                                       1,
                                       draft_token,
                                       torch::tensor({{0.03f, 0.04f}}),
                                       torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
        };
        auto valid_stream   = make_stream({1, 0}, 2, 1);
        auto invalid_stream = make_stream({0, 1}, invalid_token, invalid_draft_token);
        auto valid_output   = valid_stream->getSPOutputBuffer();
        auto invalid_output = invalid_stream->getSPOutputBuffer();
        valid_output->tensors_holder = {valid_output->all_probs.cpu().clone(),
                                        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
        invalid_output->tensors_holder = {invalid_output->all_probs.cpu().clone(),
                                          torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
        const auto valid_probs_before  = valid_output->tensors_holder[0].clone();
        const auto valid_hidden_before = valid_output->tensors_holder[1].clone();
        const auto valid_iter_before   = valid_stream->iterCount();

        auto* target_model = components.fake_target_model.get();
        auto* draft_model  = components.fake_draft_model.get();
        setupFakeModels(components.executor.get(),
                        std::move(components.fake_target_model),
                        std::move(components.fake_draft_model),
                        std::move(components.fake_fast_topk_sampler),
                        std::move(components.fake_speculative_sampler),
                        std::move(components.fake_sampler));

        absl::Status status;
        EXPECT_NO_THROW(status = components.executor->process({valid_stream, invalid_stream}));
        EXPECT_TRUE(status.ok()) << status;
        EXPECT_FALSE(valid_stream->hasError());
        EXPECT_TRUE(invalid_stream->hasError());
        EXPECT_TRUE(invalid_output->tensors_holder.empty());
        ASSERT_EQ(valid_output->tensors_holder.size(), 2);
        EXPECT_TRUE(torch::equal(valid_output->tensors_holder[0], valid_probs_before));
        EXPECT_TRUE(torch::equal(valid_output->tensors_holder[1], valid_hidden_before));
        EXPECT_EQ(valid_stream->iterCount(), valid_iter_before);
        EXPECT_EQ(target_model->forwardCallCount(), 0);
        EXPECT_EQ(draft_model->forwardCallCount(), 0);
    };

    verify_isolated(/*target_input_vocab_size=*/3,
                    /*proposal_input_vocab_size=*/4,
                    /*invalid_token=*/3,
                    /*invalid_draft_token=*/2);
    verify_isolated(/*target_input_vocab_size=*/4,
                    /*proposal_input_vocab_size=*/3,
                    /*invalid_token=*/2,
                    /*invalid_draft_token=*/3);
}

TEST_F(MtpExecutorTest, decodeIntermediateDraftTokenOutsideInputVocabIsolatesInvalidStream) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle                  = 3;
    test_config.vocab_size_override                = 4;
    test_config.input_vocab_size_override          = 4;
    test_config.proposal_input_vocab_size_override = 3;
    auto components                                = createMtpExecutorComponents(test_config);

    auto invalid_stream = createDecodeStream(components.model_config,
                                             components.runtime_config,
                                             components.resource_context,
                                             {0, 1},
                                             {torch::tensor({{1}}, torch::kInt32),
                                              1,
                                              2,
                                              torch::tensor({{0.03f, 0.04f}}),
                                              torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
    auto valid_stream = createDecodeStream(components.model_config,
                                           components.runtime_config,
                                           components.resource_context,
                                           {1, 2},
                                           {torch::tensor({{2}}, torch::kInt32),
                                            1,
                                            1,
                                            torch::tensor({{0.13f, 0.14f}}),
                                            torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f}})});
    const auto invalid_tokens_before = invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_tokens_before   = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto invalid_iter_before   = invalid_stream->iterCount();
    const auto valid_iter_before     = valid_stream->iterCount();

    GptModelInputs draft_input;
    draft_input.combo_tokens       = torch::tensor({2, 1}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({2, 2}, torch::kInt32);
    draft_input.sequence_lengths   = torch::tensor({2, 2}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input.last_hidden_states = torch::tensor({0.03f, 0.04f, 0.13f, 0.14f}).reshape({2, 2});
    auto draft_output              = createRandomGptModelOutputs(2, 4, 2);
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    spec::FastTopKSamplerOutput draft_sampler_output;
    draft_sampler_output.token_ids = torch::tensor({3, 1}, torch::kInt32).reshape({2, 1});
    draft_sampler_output.all_probs = torch::zeros({2, 4}, torch::kFloat32);
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({invalid_stream, valid_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::OUT_OF_VOCAB_RANGE);
    EXPECT_FALSE(valid_stream->hasError());
    EXPECT_EQ(invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), invalid_tokens_before);
    EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
    EXPECT_EQ(invalid_stream->iterCount(), invalid_iter_before);
    EXPECT_EQ(valid_stream->iterCount(), valid_iter_before);
    EXPECT_EQ(target_model->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 1);
    EXPECT_TRUE(invalid_stream->getSPOutputBuffer()->tensors_holder.empty());
    EXPECT_TRUE(valid_stream->getSPOutputBuffer()->tensors_holder.empty());
}

TEST_F(MtpExecutorTest, decodeAcceptedTokenOutsideProposalInputVocabIsolatesInvalidStream) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle                  = 1;
    test_config.vocab_size_override                = 4;
    test_config.input_vocab_size_override          = 4;
    test_config.proposal_input_vocab_size_override = 3;
    auto components                                = createMtpExecutorComponents(test_config);

    auto invalid_stream = createDecodeStream(components.model_config,
                                             components.runtime_config,
                                             components.resource_context,
                                             {0, 1},
                                             {torch::tensor({{2}}, torch::kInt32),
                                              1,
                                              1,
                                              torch::tensor({{0.03f, 0.04f}}),
                                              torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
    auto valid_stream = createDecodeStream(components.model_config,
                                           components.runtime_config,
                                           components.resource_context,
                                           {1, 2},
                                           {torch::tensor({{1}}, torch::kInt32),
                                            1,
                                            2,
                                            torch::tensor({{0.13f, 0.14f}}),
                                            torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f}})},
                                           123);
    const auto invalid_tokens_before = invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_tokens_before   = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto invalid_iter_before   = invalid_stream->iterCount();
    const auto valid_iter_before     = valid_stream->iterCount();
    const auto valid_generator_state_before = valid_stream->getGenerator().get_state().clone();

    auto invalid_output = invalid_stream->getSPOutputBuffer();
    auto valid_output   = valid_stream->getSPOutputBuffer();
    invalid_output->tensors_holder = {invalid_output->all_probs.cpu().clone(),
                                      torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
    valid_output->tensors_holder = {valid_output->all_probs.cpu().clone(),
                                    torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 1, 1, 2}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2, 2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({4, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::zeros({2, 2}, torch::kInt32);
    sampler_output.all_probs = torch::zeros({2, 2, 4}, torch::kFloat32);
    sampler_output.success   = torch::tensor({true, true, true, true}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output});
    components.fake_sampler->setAdvanceSeededGenerators(true);

    auto invalid_accept_tokens = torch::tensor({{3}}, torch::kInt32);
    auto valid_accept_tokens   = torch::tensor({{1}}, torch::kInt32);
    components.fake_speculative_sampler->setOutputs(
        {{{invalid_accept_tokens, valid_accept_tokens}, {1, 1}}});
    components.fake_speculative_sampler->setAdvanceSeededGenerators(true);

    auto* target_model         = components.fake_target_model.get();
    auto* draft_model          = components.fake_draft_model.get();
    auto* speculative_sampler  = components.fake_speculative_sampler.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({invalid_stream, valid_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::OUT_OF_VOCAB_RANGE);
    EXPECT_FALSE(valid_stream->hasError());
    EXPECT_EQ(invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), invalid_tokens_before);
    EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
    EXPECT_EQ(invalid_stream->iterCount(), invalid_iter_before);
    EXPECT_EQ(valid_stream->iterCount(), valid_iter_before);
    EXPECT_TRUE(torch::equal(valid_stream->getGenerator().get_state(), valid_generator_state_before));
    EXPECT_EQ(target_model->forwardCallCount(), 1);
    EXPECT_EQ(speculative_sampler->forwardCallCount(), 1);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
    EXPECT_TRUE(invalid_output->tensors_holder.empty());
    EXPECT_TRUE(valid_output->tensors_holder.empty());
}

TEST_F(MtpExecutorTest, decodeFinalDraftTokenOutsideInputVocabIsolatesInvalidStream) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle                  = 1;
    test_config.vocab_size_override                = 4;
    test_config.input_vocab_size_override          = 3;
    test_config.proposal_input_vocab_size_override = 3;
    auto components                                = createMtpExecutorComponents(test_config);

    auto invalid_stream = createDecodeStream(components.model_config,
                                             components.runtime_config,
                                             components.resource_context,
                                             {0, 1},
                                             {torch::tensor({{2}}, torch::kInt32),
                                              1,
                                              1,
                                              torch::tensor({{0.03f, 0.04f}}),
                                              torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
    auto valid_stream = createDecodeStream(components.model_config,
                                           components.runtime_config,
                                           components.resource_context,
                                           {1, 2},
                                           {torch::tensor({{1}}, torch::kInt32),
                                            1,
                                            2,
                                            torch::tensor({{0.13f, 0.14f}}),
                                            torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f}})},
                                           123);
    const auto invalid_tokens_before = invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_tokens_before   = valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto valid_generator_state_before = valid_stream->getGenerator().get_state().clone();

    auto invalid_output = invalid_stream->getSPOutputBuffer();
    auto valid_output   = valid_stream->getSPOutputBuffer();
    invalid_output->tensors_holder = {invalid_output->all_probs.cpu().clone(),
                                      torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
    valid_output->tensors_holder = {valid_output->all_probs.cpu().clone(),
                                    torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 1, 1, 2}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2, 2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({4, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    GptModelInputs draft_input;
    draft_input.combo_tokens       = torch::tensor({1, 2}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({1, 1}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({2, 2}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input.last_hidden_states = torch::zeros({2, 2}, torch::kFloat32);
    GptModelOutputs draft_output;
    draft_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    draft_output.all_hidden_states = torch::zeros({2, 2}, torch::kFloat32);
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::zeros({2, 2}, torch::kInt32);
    sampler_output.all_probs = torch::zeros({2, 2, 4}, torch::kFloat32);
    sampler_output.success   = torch::tensor({true, true, true, true}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output});
    components.fake_sampler->setAdvanceSeededGenerators(true);

    auto invalid_accept_tokens = torch::tensor({{1}}, torch::kInt32);
    auto valid_accept_tokens   = torch::tensor({{2}}, torch::kInt32);
    components.fake_speculative_sampler->setOutputs(
        {{{invalid_accept_tokens, valid_accept_tokens}, {1, 1}}});
    components.fake_speculative_sampler->setAdvanceSeededGenerators(true);

    spec::FastTopKSamplerOutput draft_sampler_output;
    draft_sampler_output.token_ids = torch::tensor({3, 1}, torch::kInt64).reshape({2, 1});
    draft_sampler_output.all_probs = torch::zeros({2, 4}, torch::kFloat32);
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({invalid_stream, valid_stream}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::OUT_OF_VOCAB_RANGE);
    EXPECT_FALSE(valid_stream->hasError());
    EXPECT_EQ(invalid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), invalid_tokens_before);
    EXPECT_EQ(valid_stream->getCompleteTokenIds()->completeTokenIdsVec(0), valid_tokens_before);
    EXPECT_EQ(invalid_stream->iterCount(), 0);
    EXPECT_EQ(valid_stream->iterCount(), 0);
    EXPECT_TRUE(torch::equal(valid_stream->getGenerator().get_state(), valid_generator_state_before));
    EXPECT_EQ(target_model->forwardCallCount(), 1);
    EXPECT_EQ(draft_model->forwardCallCount(), 1);
    EXPECT_TRUE(invalid_output->tensors_holder.empty());
    EXPECT_TRUE(valid_output->tensors_holder.empty());
}

TEST_F(MtpExecutorTest, decodeSamplerFailureKeepsSuccessfulPeerRetryable) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = 1;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);

    auto stream1 = createDecodeStream(components.model_config,
                                      components.runtime_config,
                                      components.resource_context,
                                      {0, 1},
                                      {torch::tensor({{2}}, torch::kInt32),
                                       1,
                                       3,
                                       torch::tensor({{0.03f, 0.04f}}),
                                       torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})});
    auto stream2 = createDecodeStream(components.model_config,
                                      components.runtime_config,
                                      components.resource_context,
                                      {3, 2},
                                      {torch::tensor({{1}}, torch::kInt32),
                                       1,
                                       0,
                                       torch::tensor({{0.13f, 0.14f}}),
                                       torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f}})});
    const auto stream1_tokens_before = stream1->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto stream2_tokens_before = stream2->getCompleteTokenIds()->completeTokenIdsVec(0);
    const auto stream1_iter_before   = stream1->iterCount();
    const auto stream2_iter_before   = stream2->iterCount();

    auto stream1_sp = stream1->getSPOutputBuffer();
    auto stream2_sp = stream2->getSPOutputBuffer();
    stream1_sp->tensors_holder = {stream1_sp->all_probs.cpu().clone(),
                                  torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};
    stream2_sp->tensors_holder = {stream2_sp->all_probs.cpu().clone(),
                                  torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16))};

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 3, 1, 0}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2, 2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({4, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::zeros({2, 2}, torch::kInt32);
    sampler_output.success   = torch::tensor({true, true, true, false}, torch::kBool);
    components.fake_sampler->setOutputs({sampler_output});

    auto* draft_model         = components.fake_draft_model.get();
    auto* speculative_sampler = components.fake_speculative_sampler.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    absl::Status status;
    ASSERT_NO_THROW(status = components.executor->process({stream1, stream2}));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_FALSE(stream1->hasError());
    EXPECT_TRUE(stream2->hasError());
    EXPECT_EQ(stream1->getCompleteTokenIds()->completeTokenIdsVec(0), stream1_tokens_before);
    EXPECT_EQ(stream2->getCompleteTokenIds()->completeTokenIdsVec(0), stream2_tokens_before);
    EXPECT_EQ(stream1->iterCount(), stream1_iter_before);
    EXPECT_EQ(stream2->iterCount(), stream2_iter_before);
    EXPECT_EQ(speculative_sampler->forwardCallCount(), 0);
    EXPECT_EQ(draft_model->forwardCallCount(), 0);
    EXPECT_TRUE(stream1_sp->tensors_holder.empty());
    EXPECT_TRUE(stream2_sp->tensors_holder.empty());
}

TEST_F(MtpExecutorTest, testMultiBatchDecode) {
    // test multi batch decode not accept & accept all
    // input s1:[0, 1, 2, 3] + [2] s2:[3, 2, 1] + [3]
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    // verify [3, 2, 0, 0, 0], [3, 0, 2, 2, 1]
    // accept [3], [3, 0, 2, 2, 1]
    // next draft [1], [2]
    size_t propose_step = 4;
    size_t vocab_size   = 4;
    size_t batch_size   = 2;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size          = vocab_size;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    // Create context stream
    auto stream1_new_tokens        = torch::tensor({{3}}, torch::kInt32);
    auto stream1_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto stream1_draft_token_probs = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}});

    auto stream2_new_tokens        = torch::tensor({{1}}, torch::kInt32);
    auto stream2_hidden_states     = torch::tensor({{2.1f, 2.12f}});
    auto stream2_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});

    StreamSpecUpdateInfo spec_update_info1{stream1_new_tokens, 1, 2, stream1_hidden_states, stream1_draft_token_probs};
    StreamSpecUpdateInfo spec_update_info2{stream2_new_tokens, 1, 3, stream2_hidden_states, stream2_draft_token_probs};

    GenerateStreamPtr stream1 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2}, spec_update_info1);

    GenerateStreamPtr stream2 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {3, 2}, spec_update_info2);

    // set fake model outputs
    // set 3 step draft model outputs
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    auto draft_input_1  = GptModelInputs{};
    auto draft_input_2  = GptModelInputs{};
    auto draft_input_3  = GptModelInputs{};
    auto draft_output_1 = createRandomGptModelOutputs(2, 4, 2);
    auto draft_output_2 = createRandomGptModelOutputs(2, 4, 2);
    auto draft_output_3 = createRandomGptModelOutputs(2, 4, 2);

    draft_input_1.combo_tokens       = torch::tensor({2, 3}, torch::kInt32);
    draft_input_1.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_1.sequence_lengths   = torch::tensor({3, 2}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_1.last_hidden_states = torch::tensor({0.03f, 0.04f, 2.1f, 2.12f}).reshape({2, 2});

    draft_input_2.combo_tokens       = torch::tensor({1, 0}, torch::kInt32);
    draft_input_2.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_2.sequence_lengths   = torch::tensor({4, 3}, torch::kInt32);
    draft_input_2.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = torch::tensor({2, 2}, torch::kInt32);
    draft_input_3.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_3.sequence_lengths   = torch::tensor({5, 4}, torch::kInt32);
    draft_input_3.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    // accept [3], [3, 0, 2, 2, 1]
    auto next_draft_input  = GptModelInputs{};
    auto next_draft_output = GptModelOutputs{};
    next_draft_output.logits =
        torch::tensor({1.9f, 1.10f, 1.11f, 1.12f, 2.9f, 2.10f, 2.11f, 2.12f}).reshape({(int64_t)batch_size, 4});
    next_draft_output.all_hidden_states =
        torch::tensor({0.1f, 0.11f, 1.1f, 1.11f, 1.2f, 1.22f, 1.3f, 1.33f, 1.4f, 1.44f, 1.5f, 1.55f}).reshape({6, 2});

    next_draft_input.combo_tokens      = torch::tensor({3, 3, 0, 2, 2, 1}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({1, 5}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({3, 2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({0, 5}, torch::kInt32);
    next_draft_input.last_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f})
            .reshape({6, 2});

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

    // set target model
    // verify [3, 2, 0, 0, 0], [3, 0, 2, 2, 1]
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({3, 2, 1, 2, 3, 1, 3, 0, 2, 2}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({5, 5}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({3, 2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, torch::kInt32);

    target_output.logits =
        torch::tensor({0.1f,  0.2f,  0.3f,  0.4f,  1.1f,  1.2f,  1.3f,  1.4f,  2.1f,  2.2f,  2.3f,  2.4f,  3.1f,  3.2f,
                       3.3f,  3.4f,  4.1f,  4.2f,  4.3f,  4.4f,  -0.1f, -0.2f, -0.3f, -0.4f, -1.1f, -1.2f, -1.3f, -1.4f,
                       -2.1f, -2.2f, -2.3f, -2.4f, -3.1f, -3.2f, -3.3f, -3.4f, -4.1f, -4.2f, -4.3f, -4.4f})
            .reshape({(int64_t)(batch_size * (propose_step + 1)), 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f,
                       0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f})
            .reshape({(int64_t)(batch_size * (propose_step + 1)), 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set draft sampler outputs
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    // next draft [1], [2]
    auto draft_sampler_output_1    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_2    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_3    = spec::FastTopKSamplerOutput{};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{};

    draft_sampler_output_1.token_ids = torch::tensor({1, 0}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_1.all_probs =
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_2.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_2.all_probs =
        torch::tensor({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_3.token_ids = torch::tensor({3, 2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_3.all_probs =
        torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});
    next_draft_sampler_output.token_ids = torch::tensor({1, 2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    next_draft_sampler_output.all_probs =
        torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});

    components.fake_fast_topk_sampler->setInputs(
        {draft_output_1.logits, draft_output_2.logits, draft_output_3.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3, next_draft_sampler_output});

    // set fake sampler outputs
    auto target_sample_all_probs_data = createRandomVector<float>(batch_size * (propose_step + 1) * vocab_size, 1);
    auto sampler_input                = SamplerInputs{target_output.logits};
    auto sampler_output =
        SamplerOutput{torch::tensor({3, 2, 0, 0, 0, 3, 0, 2, 2, 1}, torch::kInt32).reshape({(int64_t)batch_size, 5})};
    sampler_output.all_probs = torch::tensor(target_sample_all_probs_data)
                                   .reshape({(int64_t)batch_size, (int64_t)(propose_step + 1), (int64_t)vocab_size});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake speculative sampler outputs
    auto accept_tokens1             = torch::tensor({{3}}, torch::kInt32);
    auto accept_tokens2             = torch::tensor({{3, 0, 2, 2, 1}}, torch::kInt32);
    auto speculative_sampler_output = spec::SpeculativeSamplerOutput{{accept_tokens1, accept_tokens2}, {1, 5}};
    auto draft_spec_sample_input    = SamplerOutput{};
    auto target_spec_sample_input   = SamplerOutput{};

    vector<vector<float>> draft_all_probs_list;
    draft_all_probs_list.push_back({0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    draft_all_probs_list.push_back(toVec<float>(draft_output_1.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_2.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_3.logits));
    draft_spec_sample_input.token_ids = torch::tensor({2, 1, 2, 3, 3, 0, 2, 2}, torch::kInt32).reshape({2, 4});
    draft_spec_sample_input.all_probs = torch::tensor(catVectors(draft_all_probs_list)).reshape({4, 8});
    target_spec_sample_input.all_probs =
        torch::tensor(target_sample_all_probs_data)
            .reshape({(int64_t)batch_size, (int64_t)(propose_step + 1), (int64_t)vocab_size});

    components.fake_speculative_sampler->setInputs({draft_spec_sample_input, target_spec_sample_input});
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    // Verify executor was created successfully
    auto status = components.executor->process({stream1, stream2});
    ASSERT_TRUE(status.ok());

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 3}, {3, 1}, {0, 1, 0, 0}, {0.1, 0.11});
    checkOutput(stream2, {3, 2, 1, 3, 0, 2, 2, 1}, {1, 2}, {0.0, 1.0, 0.0, 0.0}, {1.5, 1.55});
}

TEST_F(MtpExecutorTest, validateSpeculativeSamplerInputsAcceptsMatchingInputs) {
    constexpr int64_t batch_size   = 2;
    constexpr int64_t propose_step = 3;
    constexpr int64_t vocab_size   = 5;

    SamplerOutput draft_output;
    draft_output.token_ids =
        torch::tensor({0, 1, 2, 2, 3, 4}, torch::kInt32).reshape({batch_size, propose_step}).to(torch::kCUDA);
    draft_output.all_probs =
        torch::zeros({batch_size, propose_step, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    SamplerOutput target_output;
    target_output.token_ids = torch::ones({batch_size * (propose_step + 1), 7}, torch::kInt32);
    target_output.all_probs =
        torch::zeros({batch_size, propose_step + 1, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const auto validated =
        spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output);
    EXPECT_EQ(validated.vocab_size, vocab_size);
    EXPECT_EQ(validated.token_stride, 7);
    EXPECT_FALSE(validated.draft_token_ids_cpu.defined());

    const auto force_accept_validated =
        spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output, true);
    EXPECT_TRUE(force_accept_validated.draft_token_ids_cpu.is_cpu());
}

TEST_F(MtpExecutorTest, samplerFailureValidatorMapsRowsToStreams) {
    const auto success = torch::tensor({true, true, true, false, true}, torch::kBool);
    EXPECT_EQ(spec::findFailedSamplerStreamIndices(success, {2, 3}), std::vector<size_t>({1}));

    const auto two_failures = torch::tensor({false, true, true, false, true}, torch::kBool);
    EXPECT_EQ(spec::findFailedSamplerStreamIndices(two_failures, {2, 3}), std::vector<size_t>({0, 1}));
    EXPECT_TRUE(spec::findFailedSamplerStreamIndices(torch::Tensor(), {2, 3}).empty());

    for (const size_t propose_step : {3, 4, 5, 6}) {
        const size_t score_width = propose_step + 1;
        auto score_success = torch::ones(
            {static_cast<int64_t>(2 * score_width)}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        score_success.index_put_({static_cast<int64_t>(2 * score_width - 1)}, false);
        EXPECT_EQ(spec::findFailedSamplerStreamIndices(score_success, {score_width, score_width}),
                  std::vector<size_t>({1}));
    }
}

TEST_F(MtpExecutorTest, samplerFailureValidatorRejectsMalformedSuccessTensor) {
    EXPECT_THROW(spec::findFailedSamplerStreamIndices(torch::ones({5}, torch::kInt32), {2, 3}),
                 std::invalid_argument);
    EXPECT_THROW(spec::findFailedSamplerStreamIndices(torch::ones({1, 5}, torch::kBool), {2, 3}),
                 std::invalid_argument);
    EXPECT_THROW(spec::findFailedSamplerStreamIndices(torch::ones({4}, torch::kBool), {2, 3}),
                 std::invalid_argument);

    const auto non_contiguous = torch::ones({2, 5}, torch::kBool).transpose(0, 1);
    ASSERT_FALSE(non_contiguous.is_contiguous());
    EXPECT_THROW(spec::findFailedSamplerStreamIndices(non_contiguous, {5, 5}), std::invalid_argument);
}

TEST_F(MtpExecutorTest, validateSpeculativeSamplerInputsRejectsProbabilityVocabMismatch) {
    constexpr int64_t batch_size   = 1;
    constexpr int64_t propose_step = 2;

    SamplerOutput draft_output;
    draft_output.token_ids = torch::zeros({batch_size, propose_step}, torch::kInt32);
    draft_output.all_probs = torch::zeros(
        {batch_size, propose_step, 5}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    SamplerOutput target_output;
    target_output.token_ids = torch::zeros({batch_size * (propose_step + 1), 4}, torch::kInt32);
    target_output.all_probs = torch::zeros(
        {batch_size, propose_step + 1, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_THROW(spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output),
                 RTPException);
}

TEST_F(MtpExecutorTest, validateSpeculativeSamplerInputsRejectsOutOfRangeTokenIds) {
    constexpr int64_t batch_size   = 1;
    constexpr int64_t propose_step = 2;
    constexpr int64_t vocab_size   = 5;

    SamplerOutput draft_output;
    draft_output.token_ids = torch::tensor({0, 5}, torch::kInt32).reshape({batch_size, propose_step});
    draft_output.all_probs =
        torch::zeros({batch_size, propose_step, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    SamplerOutput target_output;
    target_output.token_ids = torch::zeros({batch_size * (propose_step + 1), 4}, torch::kInt32);
    target_output.all_probs =
        torch::zeros({batch_size, propose_step + 1, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_THROW(spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output),
                 RTPException);

    draft_output.token_ids.fill_(1);
    target_output.token_ids.index_put_({0, 3}, -1);
    EXPECT_THROW(spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output),
                 RTPException);
}

TEST_F(MtpExecutorTest, validateSpeculativeSamplerInputsRejectsNonContiguousDraftTokenIds) {
    constexpr int64_t batch_size   = 2;
    constexpr int64_t propose_step = 3;
    constexpr int64_t vocab_size   = 8;

    SamplerOutput draft_output;
    draft_output.token_ids =
        torch::arange(batch_size * propose_step, torch::kInt32).reshape({propose_step, batch_size}).transpose(0, 1);
    ASSERT_FALSE(draft_output.token_ids.is_contiguous());
    draft_output.all_probs =
        torch::zeros({batch_size, propose_step, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    SamplerOutput target_output;
    target_output.token_ids = torch::zeros({batch_size * (propose_step + 1), 5}, torch::kInt32);
    target_output.all_probs =
        torch::zeros({batch_size, propose_step + 1, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_THROW(spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output),
                 RTPException);
}

TEST_F(MtpExecutorTest, validateSpeculativeSamplerInputsRejectsNonContiguousTargetTokenIds) {
    constexpr int64_t batch_size   = 2;
    constexpr int64_t propose_step = 3;
    constexpr int64_t vocab_size   = 8;
    constexpr int64_t token_stride = 5;
    constexpr int64_t target_rows  = batch_size * (propose_step + 1);

    SamplerOutput draft_output;
    draft_output.token_ids = torch::zeros({batch_size, propose_step}, torch::kInt32);
    draft_output.all_probs =
        torch::zeros({batch_size, propose_step, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    SamplerOutput target_output;
    target_output.token_ids =
        torch::ones({token_stride, target_rows}, torch::kInt32).transpose(0, 1);
    ASSERT_FALSE(target_output.token_ids.is_contiguous());
    target_output.all_probs =
        torch::zeros({batch_size, propose_step + 1, vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_THROW(spec::validateSpeculativeSamplerInputs(batch_size, propose_step, draft_output, target_output),
                 RTPException);
}

TEST_F(MtpExecutorTest, validateSpeculativeEmittedTokenCountRejectsUnsafeBounds) {
    constexpr size_t propose_step = 3;

    EXPECT_EQ(spec::validateSpeculativeEmittedTokenCount(1, propose_step), 1);
    EXPECT_EQ(spec::validateSpeculativeEmittedTokenCount(4, propose_step), 4);
    EXPECT_THROW(spec::validateSpeculativeEmittedTokenCount(0, propose_step), RTPException);
    EXPECT_THROW(spec::validateSpeculativeEmittedTokenCount(-1, propose_step), RTPException);
    EXPECT_THROW(spec::validateSpeculativeEmittedTokenCount(5, propose_step), RTPException);
}

TEST_F(MtpExecutorTest, speculativeSamplerPreservesCorrectionTokenAfterRejection) {
    constexpr int64_t propose_step = 1;
    constexpr int64_t vocab_size   = 3;

    ModelConfig model_config;
    model_config.max_seq_len = 16;
    auto stream = createContextStream(model_config, RuntimeConfig{}, ResourceContext{}, {0});

    SamplerOutput draft_output;
    draft_output.token_ids = torch::tensor({0}, torch::kInt32).reshape({1, propose_step});
    draft_output.all_probs =
        torch::tensor({0.5f, 0.0f, 0.5f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({1, propose_step, vocab_size});

    SamplerOutput target_output;
    // The draft token has zero target probability, so rejection is deterministic. The q-p
    // correction distribution selects token 1, while an independent valid target sample is 2.
    target_output.all_probs =
        torch::tensor({0.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f},
                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({1, propose_step + 1, vocab_size});
    target_output.token_ids = torch::tensor({2, 0}, torch::kInt32).reshape({propose_step + 1, 1});

    spec::SpeculativeSampler sampler(propose_step);
    auto                     output = sampler.forward({stream}, draft_output, target_output);

    ASSERT_EQ(output.accept_len, std::vector<int>({1}));
    ASSERT_EQ(output.accept_tokens.size(), 1);
    EXPECT_EQ(toVec<int>(output.accept_tokens[0]), std::vector<int>({1}));
}

TEST_F(MtpExecutorTest, speculativeSamplerUsesTargetBonusTokenWhenAllDraftsAccepted) {
    constexpr int64_t propose_step = 1;
    constexpr int64_t vocab_size   = 3;

    ModelConfig model_config;
    model_config.max_seq_len = 16;
    auto stream = createContextStream(model_config, RuntimeConfig{}, ResourceContext{}, {0});

    SamplerOutput draft_output;
    draft_output.token_ids = torch::tensor({0}, torch::kInt32).reshape({1, propose_step});
    draft_output.all_probs =
        torch::tensor({0.5f, 0.5f, 0.0f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({1, propose_step, vocab_size});

    SamplerOutput target_output;
    target_output.all_probs =
        torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({1, propose_step + 1, vocab_size});
    target_output.token_ids = torch::tensor({0, 2}, torch::kInt32).reshape({propose_step + 1, 1});

    spec::SpeculativeSampler sampler(propose_step);
    auto                     output = sampler.forward({stream}, draft_output, target_output);

    ASSERT_EQ(output.accept_len, std::vector<int>({2}));
    ASSERT_EQ(output.accept_tokens.size(), 1);
    EXPECT_EQ(toVec<int>(output.accept_tokens[0]), std::vector<int>({0, 2}));
}

}  // namespace rtp_llm
