#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

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

#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {

using namespace std;
namespace spec = speculative;

struct MtpExecutorTestConfig {
    size_t max_seq_len         = 2048;
    size_t vocab_size          = 4;
    size_t num_layers          = 1;
    size_t gen_num_per_cycle   = 4;
    size_t vocab_size_override = 0;  // 0 means use vocab_size
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

void checkBufferEqual(DeviceBase* device, const BufferPtr& buffer1, const BufferPtr& buffer2) {
    bool buffer1_is_empty = !buffer1 || buffer1->size() == 0;
    bool buffer2_is_empty = !buffer2 || buffer2->size() == 0;

    if (buffer1_is_empty && buffer2_is_empty) {
        return;
    }
    if (buffer1_is_empty || buffer2_is_empty) {
        string buffer1_info = buffer1_is_empty ? "buffer1 is empty" : "buffer1 size: " + to_string(buffer1->size());
        string buffer2_info = buffer2_is_empty ? "buffer2 is empty" : "buffer2 size: " + to_string(buffer2->size());
        throw std::runtime_error("[test] Buffer is empty: " + buffer1_info + " " + buffer2_info);
    }

    auto buf1_h = device->clone({*buffer1, AllocationType::HOST});
    auto buf2_h = device->clone({*buffer2, AllocationType::HOST});
    EXPECT_EQ(buffer1->type(), buffer2->type());
    switch (buffer1->type()) {
        case DataType::TYPE_INT64:
            EXPECT_EQ(buffer2vector<int64_t>(*buf1_h), buffer2vector<int64_t>(*buf2_h));
            break;
        case DataType::TYPE_INT32:
            EXPECT_EQ(buffer2vector<int32_t>(*buf1_h), buffer2vector<int32_t>(*buf2_h));
            break;
        case DataType::TYPE_FP32:
            EXPECT_EQ(buffer2vector<float>(*buf1_h), buffer2vector<float>(*buf2_h));
            break;
        default:
            throw std::runtime_error("[test] Unsupported check buffer data type");
    }
}

template<typename T>
vector<T> catVectors(const vector<vector<T>>& vectors) {
    vector<T> result;
    for (const auto& vec : vectors) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

class FakeModel: public GptModel {
public:
    FakeModel(const GptModelInitParams& params, DeviceBase* device): GptModel(params), device(device) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        checkInputs(inputs);
        return output_holder.get();
    }

    void checkInputs(const GptModelInputs& inputs) {
        GptModelInputs expected_inputs = input_holder.get();
        RTP_LLM_LOG_INFO("check combo_tokens");
        checkBufferEqual(device, inputs.combo_tokens, expected_inputs.combo_tokens);
        RTP_LLM_LOG_INFO("check input_lengths");
        checkBufferEqual(device, inputs.input_lengths, expected_inputs.input_lengths);
        RTP_LLM_LOG_INFO("check sequence_lengths");
        checkBufferEqual(device, inputs.sequence_lengths, expected_inputs.sequence_lengths);
        RTP_LLM_LOG_INFO("check prefix_lengths");
        checkBufferEqual(device, inputs.prefix_lengths, expected_inputs.prefix_lengths);
        RTP_LLM_LOG_INFO("check lm_output_indexes");
        checkBufferEqual(device, inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
        RTP_LLM_LOG_INFO("check last_hidden_states");
        checkBufferEqual(device, inputs.last_hidden_states, expected_inputs.last_hidden_states);
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
    DeviceBase*                     device;
};

class FakeFastTopKSampler: public spec::FastTopKSampler {
public:
    FakeFastTopKSampler(DeviceBase* device): spec::FastTopKSampler(device, nullptr), device(device) {}

    spec::FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1) override {
        checkInputs(logits);
        return output_holder.get();
    }

    void checkInputs(const torch::Tensor& logits) {
        auto expected_logits = logits_holder.get();
        RTP_LLM_LOG_INFO("check fast_topk_sampler logits");
        checkBufferEqual(device, torchTensor2Buffer(logits), expected_logits);
    }

    void setOutputs(const vector<spec::FastTopKSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<BufferPtr>& inputs) {
        logits_holder.push(inputs);
    }

private:
    TestDataHolder<BufferPtr>                   logits_holder;
    TestDataHolder<spec::FastTopKSamplerOutput> output_holder;
    DeviceBase*                                 device;
};

class FakeSpeculativeSampler: public spec::SpeculativeSampler {
public:
    FakeSpeculativeSampler(rtp_llm::DeviceBase* device, size_t propose_step):
        spec::SpeculativeSampler(device, nullptr, propose_step), device(device) {}

    spec::SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                           SamplerOutput&                      draft_sampler_output,
                                           SamplerOutput&                      target_sampler_output) override {
        return output_holder.get();
    }

    void checkInputs(const std::list<GenerateStreamPtr>& streams,
                     SamplerOutput&                      draft_sampler_output,
                     SamplerOutput&                      target_sampler_output) {
        auto [expected_draft_sampler_input, expected_target_sampler_input] = input_holder.get();
        RTP_LLM_LOG_INFO("check draft_sampler_output.token_ids");
        checkBufferEqual(device, draft_sampler_output.token_ids, expected_draft_sampler_input.token_ids);
        RTP_LLM_LOG_INFO("check draft_sampler_output.all_probs");
        checkBufferEqual(device, draft_sampler_output.all_probs, expected_draft_sampler_input.all_probs);
        RTP_LLM_LOG_INFO("check target_sampler_output.all_probs");
        checkBufferEqual(device, target_sampler_output.all_probs, expected_target_sampler_input.all_probs);
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
    DeviceBase*                                        device;
};

class FakeSampler: public Sampler {
public:
    FakeSampler(const SamplerInitParams& params, DeviceBase* device): Sampler(params), device(device) {}

    SamplerOutput forward(const SamplerInputs& inputs) override {
        checkInputs(inputs);
        return output_holder.get();
    }

    void checkInputs(const SamplerInputs& inputs) {
        auto expected_inputs = input_holder.get();
        RTP_LLM_LOG_INFO("check sampler logits");
        checkBufferEqual(device, inputs.logits, expected_inputs.logits);
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
    DeviceBase*                   device;
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

        if (expect_last_hidden_states.size() > 0) {
            auto last_hidden_states   = sp_output_buffer->hidden_states;
            auto last_hidden_states_h = device_->clone({*last_hidden_states, AllocationType::HOST});
            EXPECT_EQ(expect_last_hidden_states, buffer2vector<float>(*last_hidden_states_h));
        } else {
            EXPECT_TRUE(sp_output_buffer->hidden_states == nullptr);
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
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
        cache_manager->init();

        // Create lora manager (can be nullptr for simple test)
        std::shared_ptr<lora::LoraManager> lora_manager = nullptr;

        // Create MtpExecutor
        auto executor =
            std::make_unique<MtpExecutor>(params, propose_params, cache_manager, device_, lora_manager, false);

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

        auto fake_target_model        = std::make_unique<FakeModel>(target_model_params, device_);
        auto fake_draft_model         = std::make_unique<FakeModel>(draft_model_params, device_);
        auto fake_fast_topk_sampler   = std::make_unique<FakeFastTopKSampler>(device_);
        auto fake_speculative_sampler = std::make_unique<FakeSpeculativeSampler>(device_, sp_config.gen_num_per_cycle);
        auto fake_sampler             = std::make_unique<FakeSampler>(SamplerInitParams{device_}, device_);

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
        auto output            = GptModelOutputs{};
        auto logits_vec        = createRandomVector<float>(token_num * vocab_size, 1.0);
        auto hidden_states_vec = createRandomVector<float>(token_num * hidden_size, 1.0);
        output.logits          = createBuffer<float>({token_num, vocab_size}, logits_vec, AllocationType::HOST);
        output.all_hidden_states =
            createBuffer<float>({token_num, hidden_size}, hidden_states_vec, AllocationType::HOST);
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
    target_input.combo_tokens      = createBuffer<int>({4}, {0, 1, 2, 3}, AllocationType::HOST);
    target_input.input_lengths     = createBuffer<int>({1}, {4}, AllocationType::HOST);
    target_input.prefix_lengths    = createBuffer<int>({1}, {0}, AllocationType::HOST);
    target_input.lm_output_indexes = createBuffer<int>({1}, {3}, AllocationType::HOST);
    target_output.logits           = createBuffer<float>({batch_size, 4}, {0.1, 0.2, 0.3, 0.4}, AllocationType::HOST);
    target_output.all_hidden_states =
        createBuffer<float>({4, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}, AllocationType::HOST);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake draft model outputs
    auto draft_input               = GptModelInputs{};
    auto draft_output              = GptModelOutputs{};
    draft_input.combo_tokens       = createBuffer<int>({4}, {1, 2, 3, 1}, AllocationType::HOST);
    draft_input.input_lengths      = createBuffer<int>({1}, {4}, AllocationType::HOST);
    draft_input.prefix_lengths     = createBuffer<int>({1}, {0}, AllocationType::HOST);
    draft_input.lm_output_indexes  = createBuffer<int>({1}, {3}, AllocationType::HOST);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits            = createBuffer<float>({batch_size, 4}, {0.5, 0.6, 0.7, 0.8}, AllocationType::HOST);
    draft_output.all_hidden_states =
        createBuffer<float>({4, 2}, {0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18}, AllocationType::HOST);

    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    BufferPtr target_token_ids = createBuffer<int>({batch_size, 1}, {1}, AllocationType::HOST);
    auto      sampler_input    = SamplerInputs{target_output.logits};
    auto      sampler_output   = SamplerOutput{target_token_ids};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake fast topk sampler outputs
    BufferPtr fast_top_k_token_ids   = createBuffer<int>({batch_size, 1}, {2}, AllocationType::HOST);
    BufferPtr fast_top_k_probs       = createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);
    auto      fast_topk_sample_input = draft_output.logits;
    auto      fast_topk_sampler_output = spec::FastTopKSamplerOutput{Buffer2torchTensor(fast_top_k_probs, false),
                                                                Buffer2torchTensor(fast_top_k_token_ids, false)};
    components.fake_fast_topk_sampler->setInputs({fast_topk_sample_input});
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

    target_input.combo_tokens      = createBuffer<int>({6}, {0, 1, 2, 3, 2, 3}, AllocationType::HOST);
    target_input.input_lengths     = createBuffer<int>({2}, {4, 2}, AllocationType::HOST);
    target_input.prefix_lengths    = createBuffer<int>({2}, {0, 0}, AllocationType::HOST);
    target_input.lm_output_indexes = createBuffer<int>({2}, {3, 5}, AllocationType::HOST);
    target_output.logits =
        createBuffer<float>({batch_size, 4}, {0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4}, AllocationType::HOST);
    target_output.all_hidden_states = createBuffer<float>(
        {6, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1.01, 1.02, 1.03, 1.04}, AllocationType::HOST);

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake draft model inputs
    auto draft_input  = GptModelInputs{};
    auto draft_output = GptModelOutputs{};

    draft_input.combo_tokens       = createBuffer<int>({6}, {1, 2, 3, 1, 3, 0}, AllocationType::HOST);
    draft_input.input_lengths      = createBuffer<int>({2}, {4, 2}, AllocationType::HOST);
    draft_input.prefix_lengths     = createBuffer<int>({2}, {0, 0}, AllocationType::HOST);
    draft_input.lm_output_indexes  = createBuffer<int>({2}, {3, 5}, AllocationType::HOST);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits =
        createBuffer<float>({batch_size, 4}, {0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8}, AllocationType::HOST);
    draft_output.all_hidden_states = createBuffer<float>(
        {6, 2}, {0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 1.11, 1.12, 1.13, 1.14}, AllocationType::HOST);

    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    BufferPtr target_token_ids = createBuffer<int>({batch_size, 1}, {1, 0}, AllocationType::HOST);
    auto      sampler_input    = SamplerInputs{target_output.logits};
    auto      sampler_output   = SamplerOutput{target_token_ids};

    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake fast topk sampler inputs
    BufferPtr fast_top_k_token_ids = createBuffer<int>({batch_size, 1}, {2, 1}, AllocationType::HOST);
    BufferPtr fast_top_k_probs =
        createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);
    auto fast_topk_sampler_output = spec::FastTopKSamplerOutput{Buffer2torchTensor(fast_top_k_probs, false),
                                                                Buffer2torchTensor(fast_top_k_token_ids, false)};
    auto fast_topk_sampler_input  = draft_output.logits;

    components.fake_fast_topk_sampler->setInputs({fast_topk_sampler_input});
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

    BufferPtr stream1_new_tokens        = createBuffer<int>({1, 1}, {2}, AllocationType::HOST);
    BufferPtr stream1_hidden_states     = createBuffer<float>({1, 2}, {0.03, 0.04}, AllocationType::HOST);
    BufferPtr stream1_draft_token_probs = createBuffer<float>({1, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);

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

    draft_input_1.combo_tokens       = createBuffer<int>({1}, {3}, AllocationType::HOST);
    draft_input_1.input_lengths      = createBuffer<int>({1}, {2}, AllocationType::HOST);
    draft_input_1.sequence_lengths   = createBuffer<int>({1}, {2}, AllocationType::HOST);
    draft_input_1.lm_output_indexes  = createBuffer<int>({1}, {0}, AllocationType::HOST);
    draft_input_1.last_hidden_states = stream1_hidden_states;

    draft_input_2.combo_tokens       = createBuffer<int>({1}, {2}, AllocationType::HOST);
    draft_input_2.input_lengths      = createBuffer<int>({1}, {2}, AllocationType::HOST);
    draft_input_2.sequence_lengths   = createBuffer<int>({1}, {3}, AllocationType::HOST);
    draft_input_2.lm_output_indexes  = createBuffer<int>({1}, {0}, AllocationType::HOST);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = createBuffer<int>({1}, {1}, AllocationType::HOST);
    draft_input_3.input_lengths      = createBuffer<int>({1}, {2}, AllocationType::HOST);
    draft_input_3.sequence_lengths   = createBuffer<int>({1}, {4}, AllocationType::HOST);
    draft_input_3.lm_output_indexes  = createBuffer<int>({1}, {0}, AllocationType::HOST);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    auto next_draft_input    = GptModelInputs{};
    auto next_draft_output   = GptModelOutputs{};
    next_draft_output.logits = createBuffer<float>({batch_size, 4}, {1.9, 1.10, 1.11, 1.12}, AllocationType::HOST);
    next_draft_output.all_hidden_states =
        createBuffer<float>({3, 2}, {0.1, 0.1, 0.2, 0.22, 0.3, 0.33}, AllocationType::HOST);

    next_draft_input.combo_tokens      = createBuffer<int>({3}, {3, 2, 0}, AllocationType::HOST);
    next_draft_input.input_lengths     = createBuffer<int>({1}, {3}, AllocationType::HOST);
    next_draft_input.prefix_lengths    = createBuffer<int>({1}, {2}, AllocationType::HOST);
    next_draft_input.lm_output_indexes = createBuffer<int>({1}, {2}, AllocationType::HOST);
    next_draft_input.last_hidden_states =
        createBuffer<float>({3, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06}, AllocationType::HOST);

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

    // set fake model outputs
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = createBuffer<int>({5}, {2, 3, 2, 1, 3}, AllocationType::HOST);
    target_input.input_lengths     = createBuffer<int>({1}, {5}, AllocationType::HOST);
    target_input.prefix_lengths    = createBuffer<int>({1}, {2}, AllocationType::HOST);
    target_input.lm_output_indexes = createBuffer<int>({5}, {0, 1, 2, 3, 4}, AllocationType::HOST);

    target_output.logits = createBuffer<float>(
        {batch_size * (propose_step + 1), 4},
        {0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4},
        AllocationType::HOST);
    target_output.all_hidden_states = createBuffer<float>(
        {propose_step + 1, 2}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10}, AllocationType::HOST);

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake sampler outputs
    BufferPtr target_token_ids = createBuffer<int>({batch_size, 5}, {3, 2, 0, 0, 0}, AllocationType::HOST);
    BufferPtr target_sample_all_probs =
        createBuffer<float>({batch_size, propose_step + 1, vocab_size},
                            createRandomVector<float>(batch_size * (propose_step + 1) * vocab_size, 1),
                            AllocationType::HOST);
    auto sampler_input       = SamplerInputs{target_output.logits};
    auto sampler_output      = SamplerOutput{target_token_ids};
    sampler_output.all_probs = target_sample_all_probs;
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // draft sampler output [2, 1, 3, 0]
    auto draft_sampler_input_1     = draft_output_1.logits;
    auto draft_sampler_input_2     = draft_output_2.logits;
    auto draft_sampler_input_3     = draft_output_3.logits;
    auto next_draft_sampler_input  = next_draft_output.logits;
    auto draft_sampler_output_1    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_2    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_3    = spec::FastTopKSamplerOutput{};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{};

    auto token_ids_1 = createBuffer<int>({batch_size, 1}, {2});
    auto all_probs_1 = createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0}, AllocationType::HOST);
    auto token_ids_2 = createBuffer<int>({batch_size, 1}, {1});
    auto all_probs_2 = createBuffer<float>({batch_size, 4}, {0.0, 0.0, 0.0, 1.0}, AllocationType::HOST);
    auto token_ids_3 = createBuffer<int>({batch_size, 1}, {3});
    auto all_probs_3 = createBuffer<float>({batch_size, 4}, {1.0, 0.0, 0.0, 0.0}, AllocationType::HOST);
    auto token_ids_4 = createBuffer<int>({batch_size, 1}, {1});
    auto all_probs_4 = createBuffer<float>({batch_size, 4}, {0.0, 1.0, 0.0, 0.0}, AllocationType::HOST);

    draft_sampler_output_1.token_ids    = Buffer2torchTensor(token_ids_1, false);
    draft_sampler_output_1.all_probs    = Buffer2torchTensor(all_probs_1, false);
    draft_sampler_output_2.token_ids    = Buffer2torchTensor(token_ids_2, false);
    draft_sampler_output_2.all_probs    = Buffer2torchTensor(all_probs_2, false);
    draft_sampler_output_3.token_ids    = Buffer2torchTensor(token_ids_3, false);
    draft_sampler_output_3.all_probs    = Buffer2torchTensor(all_probs_3, false);
    next_draft_sampler_output.token_ids = Buffer2torchTensor(token_ids_4, false);
    next_draft_sampler_output.all_probs = Buffer2torchTensor(all_probs_4, false);

    components.fake_fast_topk_sampler->setInputs(
        {draft_sampler_input_1, draft_sampler_input_2, draft_sampler_input_3, next_draft_sampler_input});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3, next_draft_sampler_output});

    // set fake speculative sampler outputs
    BufferPtr accept_tokens              = createBuffer<int>({1, 3}, {3, 2, 0}, AllocationType::HOST);
    auto      speculative_sampler_output = spec::SpeculativeSamplerOutput{{accept_tokens}, {3}};
    auto      draft_spec_sample_input    = SamplerOutput{};
    auto      target_spec_sample_input   = SamplerOutput{};

    vector<vector<float>> draft_all_probs_list;
    draft_all_probs_list.push_back(buffer2vector<float>(*stream1_draft_token_probs));
    draft_all_probs_list.push_back(buffer2vector<float>(*draft_output_1.logits));
    draft_all_probs_list.push_back(buffer2vector<float>(*draft_output_2.logits));
    draft_all_probs_list.push_back(buffer2vector<float>(*draft_output_3.logits));
    draft_spec_sample_input.token_ids = createBuffer<int>({1, 4}, {3, 2, 1, 3}, AllocationType::HOST);
    draft_spec_sample_input.all_probs =
        createBuffer<float>({4, 4}, catVectors(draft_all_probs_list), AllocationType::HOST);
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
    // set 3 step draft model outputs
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    auto draft_input_1  = GptModelInputs{};
    auto draft_input_2  = GptModelInputs{};
    auto draft_input_3  = GptModelInputs{};
    auto draft_output_1 = createRandomGptModelOutputs(2, 4, 2);
    auto draft_output_2 = createRandomGptModelOutputs(2, 4, 2);
    auto draft_output_3 = createRandomGptModelOutputs(2, 4, 2);

    draft_input_1.combo_tokens       = createBuffer<int>({2}, {2, 3}, AllocationType::HOST);
    draft_input_1.input_lengths      = createBuffer<int>({2}, {3, 2}, AllocationType::HOST);
    draft_input_1.sequence_lengths   = createBuffer<int>({2}, {3, 2}, AllocationType::HOST);
    draft_input_1.lm_output_indexes  = createBuffer<int>({2}, {0, 1}, AllocationType::HOST);
    draft_input_1.last_hidden_states = createBuffer<float>({2, 2}, {0.03, 0.04, 2.1, 2.12}, AllocationType::HOST);

    draft_input_2.combo_tokens       = createBuffer<int>({2}, {1, 0}, AllocationType::HOST);
    draft_input_2.input_lengths      = createBuffer<int>({2}, {3, 2}, AllocationType::HOST);
    draft_input_2.sequence_lengths   = createBuffer<int>({2}, {4, 3}, AllocationType::HOST);
    draft_input_2.lm_output_indexes  = createBuffer<int>({2}, {0, 1}, AllocationType::HOST);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = createBuffer<int>({2}, {2, 2}, AllocationType::HOST);
    draft_input_3.input_lengths      = createBuffer<int>({2}, {3, 2}, AllocationType::HOST);
    draft_input_3.sequence_lengths   = createBuffer<int>({2}, {5, 4}, AllocationType::HOST);
    draft_input_3.lm_output_indexes  = createBuffer<int>({2}, {0, 1}, AllocationType::HOST);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    // accept [3], [3, 0, 2, 2, 1]
    auto next_draft_input  = GptModelInputs{};
    auto next_draft_output = GptModelOutputs{};
    next_draft_output.logits =
        createBuffer<float>({batch_size, 4}, {1.9, 1.10, 1.11, 1.12, 2.9, 2.10, 2.11, 2.12}, AllocationType::HOST);
    next_draft_output.all_hidden_states = createBuffer<float>(
        {6, 2}, {0.1, 0.11, 1.1, 1.11, 1.2, 1.22, 1.3, 1.33, 1.4, 1.44, 1.5, 1.55}, AllocationType::HOST);

    next_draft_input.combo_tokens       = createBuffer<int>({6}, {3, 3, 0, 2, 2, 1}, AllocationType::HOST);
    next_draft_input.input_lengths      = createBuffer<int>({2}, {1, 5}, AllocationType::HOST);
    next_draft_input.prefix_lengths     = createBuffer<int>({2}, {3, 2}, AllocationType::HOST);
    next_draft_input.lm_output_indexes  = createBuffer<int>({2}, {0, 5}, AllocationType::HOST);
    next_draft_input.last_hidden_states = createBuffer<float>(
        {6, 2}, {0.01, 0.02, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2}, AllocationType::HOST);

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

    // set target model
    // verify [3, 2, 0, 0, 0], [3, 0, 2, 2, 1]
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = createBuffer<int>({10}, {3, 2, 1, 2, 3, 1, 3, 0, 2, 2}, AllocationType::HOST);
    target_input.input_lengths     = createBuffer<int>({2}, {5, 5}, AllocationType::HOST);
    target_input.prefix_lengths    = createBuffer<int>({2}, {3, 2}, AllocationType::HOST);
    target_input.lm_output_indexes = createBuffer<int>({10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, AllocationType::HOST);

    target_output.logits =
        createBuffer<float>({batch_size * (propose_step + 1), 4},
                            {0.1,  0.2,  0.3,  0.4,  1.1,  1.2,  1.3,  1.4,  2.1,  2.2,  2.3,  2.4,  3.1,  3.2,
                             3.3,  3.4,  4.1,  4.2,  4.3,  4.4,  -0.1, -0.2, -0.3, -0.4, -1.1, -1.2, -1.3, -1.4,
                             -2.1, -2.2, -2.3, -2.4, -3.1, -3.2, -3.3, -3.4, -4.1, -4.2, -4.3, -4.4},
                            AllocationType::HOST);
    target_output.all_hidden_states = createBuffer<float>({batch_size * (propose_step + 1), 2},
                                                          {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                                                           0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20},
                                                          AllocationType::HOST);

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set draft sampler outputs
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    // next draft [1], [2]
    auto draft_sampler_input_1     = draft_output_1.logits;
    auto draft_sampler_input_2     = draft_output_2.logits;
    auto draft_sampler_input_3     = draft_output_3.logits;
    auto next_draft_sampler_input  = next_draft_output.logits;
    auto draft_sampler_output_1    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_2    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_3    = spec::FastTopKSamplerOutput{};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{};

    auto token_ids_1 = createBuffer<int>({batch_size, 1}, {1, 0});
    auto all_probs_1 =
        createBuffer<float>({batch_size, 4}, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, AllocationType::HOST);
    auto token_ids_2 = createBuffer<int>({batch_size, 1}, {2, 2});
    auto all_probs_2 =
        createBuffer<float>({batch_size, 4}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, AllocationType::HOST);
    auto token_ids_3 = createBuffer<int>({batch_size, 1}, {3, 2});
    auto all_probs_3 =
        createBuffer<float>({batch_size, 4}, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, AllocationType::HOST);
    auto token_ids_4 = createBuffer<int>({batch_size, 1}, {1, 2});
    auto all_probs_4 =
        createBuffer<float>({batch_size, 4}, {0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, AllocationType::HOST);

    draft_sampler_output_1.token_ids    = Buffer2torchTensor(token_ids_1, false);
    draft_sampler_output_1.all_probs    = Buffer2torchTensor(all_probs_1, false);
    draft_sampler_output_2.token_ids    = Buffer2torchTensor(token_ids_2, false);
    draft_sampler_output_2.all_probs    = Buffer2torchTensor(all_probs_2, false);
    draft_sampler_output_3.token_ids    = Buffer2torchTensor(token_ids_3, false);
    draft_sampler_output_3.all_probs    = Buffer2torchTensor(all_probs_3, false);
    next_draft_sampler_output.token_ids = Buffer2torchTensor(token_ids_4, false);
    next_draft_sampler_output.all_probs = Buffer2torchTensor(all_probs_4, false);

    components.fake_fast_topk_sampler->setInputs(
        {draft_sampler_input_1, draft_sampler_input_2, draft_sampler_input_3, next_draft_sampler_input});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3, next_draft_sampler_output});

    // set fake sampler outputs
    BufferPtr target_token_ids =
        createBuffer<int>({batch_size, 5}, {3, 2, 0, 0, 0, 3, 0, 2, 2, 1}, AllocationType::HOST);
    BufferPtr target_sample_all_probs =
        createBuffer<float>({batch_size, propose_step + 1, vocab_size},
                            createRandomVector<float>(batch_size * (propose_step + 1) * vocab_size, 1),
                            AllocationType::HOST);
    auto sampler_input       = SamplerInputs{target_output.logits};
    auto sampler_output      = SamplerOutput{target_token_ids};
    sampler_output.all_probs = target_sample_all_probs;
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake speculative sampler outputs
    BufferPtr accept_tokens1             = createBuffer<int>({1, 1}, {3}, AllocationType::HOST);
    BufferPtr accept_tokens2             = createBuffer<int>({1, 5}, {3, 0, 2, 2, 1}, AllocationType::HOST);
    auto      speculative_sampler_output = spec::SpeculativeSamplerOutput{{accept_tokens1, accept_tokens2}, {1, 5}};
    auto      draft_spec_sample_input    = SamplerOutput{};
    auto      target_spec_sample_input   = SamplerOutput{};

    vector<vector<float>> draft_all_probs_list;
    draft_all_probs_list.push_back({0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    draft_all_probs_list.push_back(buffer2vector<float>(*draft_output_1.logits));
    draft_all_probs_list.push_back(buffer2vector<float>(*draft_output_2.logits));
    draft_all_probs_list.push_back(buffer2vector<float>(*draft_output_3.logits));
    draft_spec_sample_input.token_ids = createBuffer<int>({2, 4}, {2, 1, 2, 3, 3, 0, 2, 2}, AllocationType::HOST);
    draft_spec_sample_input.all_probs =
        createBuffer<float>({4, 8}, catVectors(draft_all_probs_list), AllocationType::HOST);
    target_spec_sample_input.all_probs = target_sample_all_probs;

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

}  // namespace rtp_llm
