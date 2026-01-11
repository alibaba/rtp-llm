#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {

using namespace std;

struct MtpExecutorTestConfig {
    size_t max_seq_len         = 2048;
    size_t vocab_size          = 4;
    size_t num_layers          = 1;
    size_t gen_num_per_cycle   = 4;
    size_t vocab_size_override = 0;  // 0 means use vocab_size
};

template<typename T>
struct FakeOutputHolder {
    queue<T> output;

    T get() {
        T res = output.front();
        output.pop();
        return res;
    }

    void push(const T& res) {
        output.push(res);
    }

    void push(const vector<T>& res) {
        for (const auto& r : res) {
            output.push(r);
        }
    }
};

class FakeModel: public GptModel {
public:
    FakeModel(const GptModelInitParams& params): GptModel(params) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        return output_holder.get();
    }

    void setOutputs(const vector<GptModelOutputs>& outputs) {
        output_holder.push(outputs);
    }

private:
    FakeOutputHolder<GptModelOutputs> output_holder;
};

class FakeFastTopKSampler: public speculative::FastTopKSampler {
public:
    FakeFastTopKSampler() {}

    speculative::FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1) override {
        return output_holder.get();
    }

    void setOutputs(const vector<speculative::FastTopKSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

private:
    FakeOutputHolder<speculative::FastTopKSamplerOutput> output_holder;
};

class FakeSpeculativeSampler: public speculative::SpeculativeSampler {
public:
    FakeSpeculativeSampler(rtp_llm::DeviceBase* device, size_t propose_step):
        speculative::SpeculativeSampler(device, propose_step) {}

    speculative::SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                                  SamplerOutput&                      draft_sampler_output,
                                                  SamplerOutput&                      target_sampler_output) override {
        return output_holder.get();
    }

    void setOutputs(const vector<speculative::SpeculativeSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

private:
    FakeOutputHolder<speculative::SpeculativeSamplerOutput> output_holder;
};

class FakeSampler: public Sampler {
public:
    FakeSampler(const SamplerInitParams& params): Sampler(params) {}

    SamplerOutput forward(const SamplerInputs& inputs) override {
        return output_holder.get();
    }

    void setOutputs(const vector<SamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

private:
    FakeOutputHolder<SamplerOutput> output_holder;
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
                                          const vector<int>&     input_ids) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = createBuffer<int32_t>({input_ids.size()}, input_ids, AllocationType::HOST);
        query->generate_config = make_shared<GenerateConfig>();
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        return stream;
    }

    GenerateStreamPtr createDecodeStream(const ModelConfig&          model_config,
                                         const RuntimeConfig&        runtime_config,
                                         const ResourceContext&      resource_context,
                                         const vector<int>&          input_ids,
                                         const StreamSpecUpdateInfo& spec_update_info) {
        GenerateStreamPtr stream = createContextStream(model_config, runtime_config, resource_context, input_ids);

        auto      sp_buffer   = std::make_shared<SpeculativeExecutorStreamOutput>();
        BufferPtr spec_tokens = createBuffer<int>({1, 2}, {-1, -1}, AllocationType::HOST);
        sp_buffer->tokens     = spec_tokens;

        stream->setSPOutputBuffer(sp_buffer);
        stream->specUpdate(spec_update_info);
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
        auto tokens_h         = device_->clone({*tokens, AllocationType::HOST});
        EXPECT_EQ(expect_propose_tokens, buffer2vector<int>(*tokens_h));

        auto all_probs   = sp_output_buffer->all_probs;
        auto all_probs_h = device_->clone({*all_probs, AllocationType::HOST});
        EXPECT_EQ(expect_all_probs, buffer2vector<float>(*all_probs_h));

        auto last_hidden_states   = sp_output_buffer->hidden_states;
        auto last_hidden_states_h = device_->clone({*last_hidden_states, AllocationType::HOST});
        EXPECT_EQ(expect_last_hidden_states, buffer2vector<float>(*last_hidden_states_h));
    }

    MtpExecutorComponents createMtpExecutorComponents(const MtpExecutorTestConfig& test_config) {
        CustomConfig               config;
        ModelConfig                model_config;
        RuntimeConfig              runtime_config;
        KVCacheConfig              kv_cache_config;
        ResourceContext            resource_context;
        SpeculativeExecutionConfig sp_config;
        KVCacheParam               kv_params{1, 1, 1, 64, 1, DataType::TYPE_FP16};
        CacheConfig                cache_config(kv_params);

        model_config.max_seq_len    = test_config.max_seq_len;
        model_config.vocab_size     = test_config.vocab_size;
        model_config.num_layers     = test_config.num_layers;
        sp_config.gen_num_per_cycle = test_config.gen_num_per_cycle;

        EngineInitParams params =
            createEngineInitParams(device_, config, model_config, runtime_config, kv_cache_config);
        params.sp_config = sp_config;
        if (test_config.vocab_size_override > 0) {
            params.model_config_.vocab_size = test_config.vocab_size_override;
        }

        // Create propose model engine init params
        auto mtp_model_params   = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        auto mtp_params         = std::make_unique<EngineInitParams>(params);
        mtp_params->py_sp_model = py::none();

        mtp_model_params->push_back(std::move(mtp_params));

        auto propose_params = std::make_unique<ProposeModelEngineInitParams>(
            SP_TYPE_MTP, sp_config.gen_num_per_cycle, std::move(mtp_model_params));

        // Create cache managers
        std::shared_ptr<CacheManager>              cache_manager = make_shared<CacheManager>(cache_config, device_);
        std::vector<std::shared_ptr<CacheManager>> mtp_cache_managers = {cache_manager};

        // Create lora manager (can be nullptr for simple test)
        std::shared_ptr<lora::LoraManager> lora_manager = nullptr;

        // Create MtpExecutor
        auto executor = std::make_unique<MtpExecutor>(
            params, propose_params, cache_manager, mtp_cache_managers, device_, lora_manager, false);

        // Create fake models
        GptModelInitParams target_model_params(
            {device_,
             params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id});

        GptModelInitParams draft_model_params(
            {device_,
             params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id});

        auto fake_target_model        = std::make_unique<FakeModel>(target_model_params);
        auto fake_draft_model         = std::make_unique<FakeModel>(draft_model_params);
        auto fake_fast_topk_sampler   = std::make_unique<FakeFastTopKSampler>();
        auto fake_speculative_sampler = std::make_unique<FakeSpeculativeSampler>(device_, sp_config.gen_num_per_cycle);
        auto fake_sampler             = std::make_unique<FakeSampler>(SamplerInitParams{device_, 1});

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
    auto target_output   = GptModelOutputs{};
    auto draft_output    = GptModelOutputs{};
    target_output.logits = createBuffer<float>({batch_size, 4}, {0.1, 0.2, 0.3, 0.4}, AllocationType::HOST);
    target_output.all_hidden_states =
        createBuffer<float>({4, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}, AllocationType::HOST);
    draft_output.logits = createBuffer<float>({batch_size, 4}, {0.5, 0.6, 0.7, 0.8}, AllocationType::HOST);
    draft_output.all_hidden_states =
        createBuffer<float>({4, 2}, {0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18}, AllocationType::HOST);

    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    BufferPtr target_token_ids     = createBuffer<int>({batch_size, 1}, {1}, AllocationType::HOST);
    BufferPtr fast_top_k_token_ids = createBuffer<int>({batch_size, 1}, {2}, AllocationType::HOST);
    BufferPtr fast_top_k_probs     = createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);

    auto sampler_output           = SamplerOutput{target_token_ids};
    auto fast_topk_sampler_output = speculative::FastTopKSamplerOutput{Buffer2torchTensor(fast_top_k_probs, false),
                                                                       Buffer2torchTensor(fast_top_k_token_ids, false)};

    components.fake_sampler->setOutputs({sampler_output});
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
    auto target_output = GptModelOutputs{};
    auto draft_output  = GptModelOutputs{};
    target_output.logits =
        createBuffer<float>({batch_size, 4}, {0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4}, AllocationType::HOST);
    target_output.all_hidden_states = createBuffer<float>(
        {6, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1.01, 1.02, 1.03, 1.04}, AllocationType::HOST);
    draft_output.logits =
        createBuffer<float>({batch_size, 4}, {0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8}, AllocationType::HOST);
    draft_output.all_hidden_states = createBuffer<float>(
        {6, 2}, {0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 1.11, 1.12, 1.13, 1.14}, AllocationType::HOST);

    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    BufferPtr target_token_ids     = createBuffer<int>({batch_size, 1}, {1, 0}, AllocationType::HOST);
    BufferPtr fast_top_k_token_ids = createBuffer<int>({batch_size, 1}, {2, 1}, AllocationType::HOST);
    BufferPtr fast_top_k_probs =
        createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);

    auto sampler_output           = SamplerOutput{target_token_ids};
    auto fast_topk_sampler_output = speculative::FastTopKSamplerOutput{Buffer2torchTensor(fast_top_k_probs, false),
                                                                       Buffer2torchTensor(fast_top_k_token_ids, false)};

    components.fake_sampler->setOutputs({sampler_output});
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

TEST_F(MtpExecutorTest, testSingleBatchDecode) {
    size_t                propose_step = 4;
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = 4;
    auto components                 = createMtpExecutorComponents(test_config);

    size_t batch_size = 1;

    // Create context stream
    BufferPtr stream1_new_tokens        = createBuffer<int>({1, 1}, {3}, AllocationType::HOST);
    BufferPtr stream1_hidden_states     = createBuffer<float>({1, 2}, {0.03, 0.04}, AllocationType::HOST);
    BufferPtr stream1_draft_token_probs = createBuffer<float>({1, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);

    StreamSpecUpdateInfo spec_update_info1{stream1_new_tokens, 1, 3, stream1_hidden_states, stream1_draft_token_probs};

    GenerateStreamPtr stream1 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2}, spec_update_info1);

    // set fake model outputs
    auto target_output     = GptModelOutputs{};
    auto draft_output      = GptModelOutputs{};
    auto next_draft_output = GptModelOutputs{};

    target_output.logits = createBuffer<float>(
        {batch_size * (propose_step + 1), 4},
        {0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4},
        AllocationType::HOST);
    target_output.all_hidden_states = createBuffer<float>(
        {propose_step + 1, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10}, AllocationType::HOST);

    draft_output.logits            = createBuffer<float>({batch_size, 4}, {0.5, 0.6, 0.7, 0.8}, AllocationType::HOST);
    draft_output.all_hidden_states = createBuffer<float>({1, 2}, {0.11, 0.12}, AllocationType::HOST);

    next_draft_output.logits = createBuffer<float>({batch_size, 4}, {1.9, 1.10, 1.11, 1.12}, AllocationType::HOST);
    next_draft_output.all_hidden_states =
        createBuffer<float>({4, 2}, {0.1, 0.1, 0.2, 0.22, 0.3, 0.33, 0.4, 0.44}, AllocationType::HOST);

    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setOutputs({draft_output, draft_output, draft_output, next_draft_output});

    // set fake sampler outputs
    BufferPtr target_token_ids = createBuffer<int>({batch_size, 5}, {2, 2, 2, 1, 1}, AllocationType::HOST);

    BufferPtr target_sample_all_probs = createBuffer<float>(
        {batch_size, 5, 4},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
        AllocationType::HOST);

    BufferPtr fast_top_k_token_ids  = createBuffer<int>({batch_size, 1}, {2});
    BufferPtr fast_top_k_probs      = createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);
    BufferPtr next_fast_top_k_probs = createBuffer<float>({batch_size, 4}, {0.0, 0.0, 0.0, 1.0}, AllocationType::HOST);
    BufferPtr next_fast_top_k_token_ids = createBuffer<int>({batch_size, 1}, {3});

    auto sampler_output                = SamplerOutput{target_token_ids};
    sampler_output.all_probs           = target_sample_all_probs;
    auto fast_topk_sampler_output      = speculative::FastTopKSamplerOutput{Buffer2torchTensor(fast_top_k_probs, false),
                                                                       Buffer2torchTensor(fast_top_k_token_ids, false)};
    auto next_fast_topk_sampler_output = speculative::FastTopKSamplerOutput{
        Buffer2torchTensor(next_fast_top_k_probs, false), Buffer2torchTensor(next_fast_top_k_token_ids, false)};

    BufferPtr accept_tokens              = createBuffer<int>({1, 4}, {2, 2, 2, 1}, AllocationType::HOST);
    auto      speculative_sampler_output = speculative::SpeculativeSamplerOutput{{accept_tokens}, {4}};

    components.fake_sampler->setOutputs({sampler_output});
    components.fake_fast_topk_sampler->setOutputs(
        {fast_topk_sampler_output, fast_topk_sampler_output, fast_topk_sampler_output, next_fast_topk_sampler_output});
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
    checkOutput(stream1, {0, 1, 2, 3, 2, 2, 2, 1}, {1, 3}, {0.0, 0.0, 0.0, 1.0}, {0.4, 0.44});
}

TEST_F(MtpExecutorTest, testMultiBatchDecode) {
    size_t propose_step = 4;
    size_t vocab_size   = 4;
    size_t hidden_size  = 2;
    size_t batch_size   = 2;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size          = vocab_size;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    // Create context stream
    BufferPtr stream1_new_tokens        = createBuffer<int>({1, 1}, {3}, AllocationType::HOST);
    BufferPtr stream1_hidden_states     = createBuffer<float>({1, 2}, {0.03, 0.04}, AllocationType::HOST);
    BufferPtr stream1_draft_token_probs = createBuffer<float>({1, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);

    BufferPtr stream2_new_tokens        = createBuffer<int>({1, 1}, {1}, AllocationType::HOST);
    BufferPtr stream2_hidden_states     = createBuffer<float>({1, 2}, {2.1, 2.12}, AllocationType::HOST);
    BufferPtr stream2_draft_token_probs = createBuffer<float>({1, 4}, {0.0, 0.0, 0.0, 1.0}, AllocationType::HOST);

    StreamSpecUpdateInfo spec_update_info1{stream1_new_tokens, 1, 2, stream1_hidden_states, stream1_draft_token_probs};
    StreamSpecUpdateInfo spec_update_info2{stream2_new_tokens, 1, 3, stream2_hidden_states, stream2_draft_token_probs};

    GenerateStreamPtr stream1 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2}, spec_update_info1);

    GenerateStreamPtr stream2 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {3, 2}, spec_update_info2);

    // set fake model outputs
    auto target_output     = GptModelOutputs{};
    auto draft_output      = GptModelOutputs{};
    auto next_draft_output = GptModelOutputs{};

    target_output.logits = createBuffer<float>({batch_size * (propose_step + 1), vocab_size},
                                               vector<float>(batch_size * (propose_step + 1) * vocab_size, 0.1),
                                               AllocationType::HOST);
    target_output.all_hidden_states =
        createBuffer<float>({batch_size * (propose_step + 1), hidden_size},
                            vector<float>(batch_size * (propose_step + 1) * hidden_size, 0.01),
                            AllocationType::HOST);

    draft_output.logits = createBuffer<float>(
        {batch_size, vocab_size}, vector<float>(batch_size * vocab_size, 0.5), AllocationType::HOST);
    draft_output.all_hidden_states = createBuffer<float>(
        {batch_size, hidden_size}, vector<float>(batch_size * hidden_size, 0.11), AllocationType::HOST);

    next_draft_output.logits = createBuffer<float>(
        {batch_size, vocab_size}, vector<float>(batch_size * vocab_size, vocab_size), AllocationType::HOST);
    next_draft_output.all_hidden_states = createBuffer<float>(
        {5, hidden_size}, {0.1, 0.1, 0.2, 0.22, 0.3, 0.33, 0.4, 0.44, 1.1, 1.11}, AllocationType::HOST);

    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setOutputs({draft_output, draft_output, draft_output, next_draft_output});

    // set fake sampler outputs
    BufferPtr target_token_ids =
        createBuffer<int>({batch_size, 5}, {2, 2, 2, 1, 1, 1, 1, 1, 1, 1}, AllocationType::HOST);

    BufferPtr target_sample_all_probs = createBuffer<float>(
        {batch_size, 5, 4},
        {
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
        },
        AllocationType::HOST);

    BufferPtr fast_top_k_token_ids = createBuffer<int>({batch_size, 1}, {2, 3});
    BufferPtr fast_top_k_probs =
        createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, AllocationType::HOST);
    BufferPtr next_fast_top_k_probs =
        createBuffer<float>({batch_size, 4}, {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}, AllocationType::HOST);
    BufferPtr next_fast_top_k_token_ids = createBuffer<int>({batch_size, 1}, {3, 1});

    auto sampler_output                = SamplerOutput{target_token_ids};
    sampler_output.all_probs           = target_sample_all_probs;
    auto fast_topk_sampler_output      = speculative::FastTopKSamplerOutput{Buffer2torchTensor(fast_top_k_probs, false),
                                                                       Buffer2torchTensor(fast_top_k_token_ids, false)};
    auto next_fast_topk_sampler_output = speculative::FastTopKSamplerOutput{
        Buffer2torchTensor(next_fast_top_k_probs, false), Buffer2torchTensor(next_fast_top_k_token_ids, false)};

    BufferPtr accept_tokens_1       = createBuffer<int>({1, 4}, {2, 2, 2, 1}, AllocationType::HOST);
    BufferPtr accept_tokens_2       = createBuffer<int>({1, 1}, {2}, AllocationType::HOST);
    auto speculative_sampler_output = speculative::SpeculativeSamplerOutput{{accept_tokens_1, accept_tokens_2}, {4, 1}};

    components.fake_sampler->setOutputs({sampler_output});
    components.fake_fast_topk_sampler->setOutputs(
        {fast_topk_sampler_output, fast_topk_sampler_output, fast_topk_sampler_output, next_fast_topk_sampler_output});
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
    checkOutput(stream1, {0, 1, 2, 3, 2, 2, 2, 1}, {1, 3}, {0.0, 0.0, 0.0, 1.0}, {0.4, 0.44});
    checkOutput(stream2, {3, 2, 1, 2}, {2, 1}, {0.0, 1.0, 0.0, 0.0}, {1.1, 1.11});
}

}  // namespace rtp_llm
