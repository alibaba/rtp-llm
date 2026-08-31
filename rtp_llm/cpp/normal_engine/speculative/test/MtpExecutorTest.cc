#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <chrono>
#include "autil/EnvUtil.h"
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include <ATen/cuda/CUDAContext.h>
#endif

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
        checkInputs(inputs);
        return output_holder.get();
    }

    void prepareAttentionInputs(const GptModelInputs& inputs) override {
        if (prepare_input_holder.test_data.empty()) {
            return;
        }
        GptModelInputs expected_inputs = prepare_input_holder.get();
        checkTensorField("prepared input_lengths", inputs.input_lengths, expected_inputs.input_lengths);
        checkTensorField("prepared sequence_lengths", inputs.sequence_lengths, expected_inputs.sequence_lengths);
        checkTensorField("prepared prefix_lengths", inputs.prefix_lengths, expected_inputs.prefix_lengths);
        checkTensorField("prepared sequence_lengths_plus_1",
                         inputs.sequence_lengths_plus_1,
                         expected_inputs.sequence_lengths_plus_1);
        checkTensorField("prepared lm_output_indexes", inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
    }

    void abortPrefillChunkSession() noexcept override {
        ++abort_prefill_chunk_session_count;
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
        if (expected_inputs.input_lengths_host_for_log.defined()) {
            checkTensorField("input_lengths_host_for_log",
                             inputs.input_lengths_host_for_log,
                             expected_inputs.input_lengths_host_for_log);
        }
        if (expected_inputs.prefix_lengths_host_for_log.defined()) {
            checkTensorField("prefix_lengths_host_for_log",
                             inputs.prefix_lengths_host_for_log,
                             expected_inputs.prefix_lengths_host_for_log);
        }
        if (expected_inputs.cache_store_publish_plan.has_value()) {
            ASSERT_TRUE(inputs.cache_store_publish_plan.has_value());
            checkTensorField("cache_store_publish_begin_block_host",
                             inputs.cache_store_publish_plan->begin_block_host,
                             expected_inputs.cache_store_publish_plan->begin_block_host);
            checkTensorField("cache_store_publish_end_block_host",
                             inputs.cache_store_publish_plan->end_block_host,
                             expected_inputs.cache_store_publish_plan->end_block_host);
            checkTensorField("cache_store_publish_terminal_host",
                             inputs.cache_store_publish_plan->terminal_host,
                             expected_inputs.cache_store_publish_plan->terminal_host);
        }
        checkTensorField("lm_output_indexes", inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
        checkTensorField("last_hidden_states", inputs.last_hidden_states, expected_inputs.last_hidden_states);
        if (expected_inputs.kv_cache_group_types.defined()) {
            checkTensorField(
                "kv_cache_group_types", inputs.kv_cache_group_types, expected_inputs.kv_cache_group_types);
        }
        if (expected_inputs.kv_cache_group_types_host.defined()) {
            checkTensorField("kv_cache_group_types_host",
                             inputs.kv_cache_group_types_host,
                             expected_inputs.kv_cache_group_types_host);
        }
        if (expected_inputs.kv_cache_layer_to_group.defined()) {
            checkTensorField("kv_cache_layer_to_group",
                             inputs.kv_cache_layer_to_group,
                             expected_inputs.kv_cache_layer_to_group);
        }
        if (expected_inputs.kv_cache_layer_to_group_host.defined()) {
            checkTensorField("kv_cache_layer_to_group_host",
                             inputs.kv_cache_layer_to_group_host,
                             expected_inputs.kv_cache_layer_to_group_host);
        }
        EXPECT_EQ(inputs.is_prefill_chunk, expected_inputs.is_prefill_chunk);
    }

    void setOutputs(const vector<GptModelOutputs>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<GptModelInputs>& inputs) {
        input_holder.push(inputs);
    }

    void setPrepareInputs(const vector<GptModelInputs>& inputs) {
        prepare_input_holder.push(inputs);
    }

    bool hasPendingPrepareInputs() const {
        return !prepare_input_holder.test_data.empty();
    }

    int abort_prefill_chunk_session_count = 0;

private:
    TestDataHolder<GptModelInputs>  input_holder;
    TestDataHolder<GptModelInputs>  prepare_input_holder;
    TestDataHolder<GptModelOutputs> output_holder;
};

class FakeFastTopKSampler: public spec::FastTopKSampler {
public:
    FakeFastTopKSampler(): spec::FastTopKSampler(torch::Tensor()) {}

    spec::FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1) override {
        checkInputs(logits);
        return output_holder.get();
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
    FakeSpeculativeSampler(size_t propose_step): spec::SpeculativeSampler(torch::Tensor(), propose_step) {}

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
};

class FakeSampler: public Sampler {
public:
    FakeSampler(const SamplerInitParams& params): Sampler(params) {}

    SamplerOutput forward(const SamplerInputs& inputs) override {
        if (inputs.logits_processor_states_ptr) {
            inputs.logits_processor_states_ptr->batchProcess(inputs);
        }
        checkInputs(inputs);
        return output_holder.get();
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
};

class RejectDraftTokenSpecProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    explicit RejectDraftTokenSpecProcessor(int32_t rejected_token, int64_t accepted_token_len):
        rejected_token_(rejected_token), accepted_token_len_(accepted_token_len) {}

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        inputs.logits.narrow(0, start_idx, finish_idx - start_idx).fill_(BaseLogitsProcessor::neg_inf);
    }
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const torch::Tensor&, int32_t num_new_tokens) override {
        accepted_token_len_ += num_new_tokens;
    }

    bool isStateful() const override {
        return true;
    }

    int64_t acceptedTokenLen() const override {
        return accepted_token_len_;
    }

    bool isSpecVerifyEligible() const override {
        return true;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        if (request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
            return request.propose_step;
        }
        std::fill_n(request.bitmask_cpu_out,
                    static_cast<size_t>(request.propose_step + 1) * request.bitmask_size_int32,
                    SpecLogitsProcessor::kBitmaskAllowAll);
        if (request.bitmask_size_int32 > 0 && rejected_token_ >= 0
            && static_cast<size_t>(rejected_token_) < request.vocab_size) {
            request.bitmask_cpu_out[rejected_token_ / 32] &= ~(1u << (rejected_token_ % 32));
        }
        if (request.draft_tokens != nullptr && request.draft_tokens[0] == rejected_token_) {
            return 0;
        }
        return request.propose_step;
    }

private:
    int32_t rejected_token_;
    int64_t accepted_token_len_;
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
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
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

        auto sp_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_buffer->tokens = torch::tensor({-1, -1}, torch::kInt32).reshape({1, 2});

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
        // The combined cache layout appends the single MTP layer after the
        // target layer. Keep the test fixture consistent with the mapping
        // produced by CacheConfigCreator::createSpConfig.
        const int global_mtp_layer_id              = static_cast<int>(cache_config.layer_num);
        cache_config.layer_all_num                 = cache_config.layer_num + mtp_config.layer_num;
        cache_config.global_layer_ids[0].push_back(global_mtp_layer_id);
        cache_config.layer_ids[0].push_back(global_mtp_layer_id);
        mtp_config.global_layer_ids[0]             = {global_mtp_layer_id};
        mtp_config.local_to_global_layer_ids       = {global_mtp_layer_id};
        cache_config.mtp_sub_configs.push_back(std::make_shared<CacheConfig>(mtp_config));

        EngineInitParams params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
        params.sp_config        = sp_config;
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
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();

        // Create MtpExecutor
        auto executor = std::make_unique<MtpExecutor>(params, propose_params, cache_manager);

        // Create fake models
        GptModelInitParams target_model_params(
            {params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id,
             params.parallelism_config});

        GptModelInitParams draft_model_params(
            {params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id,
             params.parallelism_config});

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

TEST_F(MtpExecutorTest, testDeterministicDraftSamplerReportsPointMassProposal) {
    auto identity_map = torch::arange(4, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    spec::FastTopKSampler sampler(identity_map);
    auto                  logits =
        torch::tensor({{0.0f, 1.0f, 4.0f, 2.0f}}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto output = sampler.forward(logits);

    EXPECT_EQ(output.token_ids.item<int64_t>(), 2);
    checkTensorEqual(output.all_probs, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}}).to(torch::kCUDA));
}

TEST_F(MtpExecutorTest, testMakePrefillRoundInputPacksMultiRequestRounds) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    GptModelInputs inputs;
    inputs.combo_tokens              = torch::arange(0, 8, torch::kInt32).cuda();
    inputs.combo_tokens_host_for_log = inputs.combo_tokens.cpu().clone();
    inputs.combo_tokens_type_ids     = torch::arange(100, 108, torch::kInt32).cuda();
    inputs.combo_position_ids        = torch::arange(200, 208, torch::kInt32).cuda();
    inputs.text_tokens_mask          = torch::ones({8}, torch::kBool).cuda();
    inputs.input_lengths             = torch::tensor({4, 4}, torch::kInt32).cuda();
    inputs.prefix_lengths            = torch::tensor({0, 0}, torch::kInt32).cuda();
    inputs.sequence_lengths          = torch::empty({0}, torch::kInt32).cuda();
    inputs.last_hidden_states        = torch::ones({8, 3}, torch::kFloat32).cuda();

    // Python-planned rounds (plan_kimi_k3_chunk_rounds, chunk_budget=4,
    // page_size=2): two aligned non-terminal slices, then a terminal round.
    PrefillChunkRound first_round;
    first_round.slices = {
        {0, 0, 2, 2, 0, 2, false},
        {1, 4, 6, 2, 0, 2, false},
    };
    PrefillChunkRound last_round;
    last_round.slices = {
        {0, 2, 4, 2, 2, 4, true},
        {1, 6, 8, 2, 2, 4, true},
    };

    auto first = components.executor->makePrefillRoundInput(inputs, first_round, /*total_tokens=*/8);
    EXPECT_EQ(toVec<int32_t>(first.combo_tokens), (std::vector<int32_t>{0, 1, 4, 5}));
    EXPECT_EQ(toVec<int32_t>(first.combo_tokens_host_for_log), (std::vector<int32_t>{0, 1, 4, 5}));
    EXPECT_EQ(toVec<int32_t>(first.combo_tokens_type_ids), (std::vector<int32_t>{100, 101, 104, 105}));
    EXPECT_EQ(toVec<int32_t>(first.combo_position_ids), (std::vector<int32_t>{200, 201, 204, 205}));
    EXPECT_EQ(toVec<bool>(first.text_tokens_mask), (std::vector<bool>{true, true, true, true}));
    EXPECT_EQ(toVec<int32_t>(first.input_lengths), (std::vector<int32_t>{2, 2}));
    EXPECT_EQ(toVec<int32_t>(first.prefix_lengths), (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(first.lm_output_indexes.numel(), 0);
    EXPECT_FALSE(first.last_hidden_states.defined());
    EXPECT_TRUE(first.is_prefill_chunk);
    EXPECT_EQ(first.prefill_chunk_kv_length, 4);

    auto last = components.executor->makePrefillRoundInput(inputs, last_round, /*total_tokens=*/8);
    EXPECT_EQ(toVec<int32_t>(last.combo_tokens), (std::vector<int32_t>{2, 3, 6, 7}));
    EXPECT_EQ(toVec<int32_t>(last.input_lengths), (std::vector<int32_t>{2, 2}));
    EXPECT_EQ(toVec<int32_t>(last.prefix_lengths), (std::vector<int32_t>{2, 2}));
    EXPECT_EQ(toVec<int32_t>(last.lm_output_indexes), (std::vector<int32_t>{1, 3}));
    EXPECT_EQ(last.prefill_chunk_kv_length, 8);
}

TEST_F(MtpExecutorTest, testMakePrefillRoundInputSelectsPerRequestMetadataRows) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    GptModelInputs inputs;
    inputs.combo_tokens                  = torch::arange(0, 12, torch::kInt32).cuda();
    inputs.input_lengths                 = torch::tensor({4, 4, 4}, torch::kInt32).cuda();
    inputs.prefix_lengths                = torch::tensor({0, 0, 0}, torch::kInt32).cuda();
    inputs.sequence_lengths              = torch::empty({0}, torch::kInt32).cuda();
    inputs.kv_cache_block_id             = torch::arange(0, 9, torch::kInt32).reshape({3, 3}).cuda();
    inputs.kv_cache_block_id_host        = inputs.kv_cache_block_id.cpu().clone();
    inputs.kv_cache_kernel_block_id      = torch::arange(0, 36, torch::kInt32).reshape({4, 3, 3}).cuda();
    inputs.kv_cache_kernel_block_id_host = inputs.kv_cache_kernel_block_id.cpu().clone();
    inputs.request_id                    = torch::tensor({100, 200, 300}, torch::kInt64).cuda();

    // Python-planned rounds (chunk_budget=4, page_size=2): requests 0+1
    // advance in round 0, request 2 alone in round 1.
    PrefillChunkRound first_round;
    first_round.slices = {
        {0, 0, 2, 2, 0, 2, false},
        {1, 4, 6, 2, 0, 2, false},
    };
    PrefillChunkRound middle_round;
    middle_round.slices = {
        {2, 8, 10, 2, 0, 2, false},
    };

    // Round 0 keeps requests 0 and 1; round 1 advances only request 2.
    auto first = components.executor->makePrefillRoundInput(inputs, first_round, /*total_tokens=*/12);
    EXPECT_EQ(toVec<int64_t>(first.request_id), (std::vector<int64_t>{100, 200}));
    EXPECT_EQ(toVec<int32_t>(first.kv_cache_block_id), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(toVec<int32_t>(first.kv_cache_block_id_host), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(toVec<int32_t>(first.kv_cache_kernel_block_id),
              (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23,
                                    27, 28, 29, 30, 31, 32}));

    auto middle = components.executor->makePrefillRoundInput(inputs, middle_round, /*total_tokens=*/12);
    EXPECT_EQ(toVec<int32_t>(middle.combo_tokens), (std::vector<int32_t>{8, 9}));
    EXPECT_EQ(toVec<int64_t>(middle.request_id), (std::vector<int64_t>{300}));
    EXPECT_EQ(toVec<int32_t>(middle.kv_cache_block_id), (std::vector<int32_t>{6, 7, 8}));
}

TEST_F(MtpExecutorTest, testShiftRoundComboTokensAppliesPerSliceLookahead) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    GptModelInputs inputs;
    inputs.combo_tokens              = torch::arange(0, 8, torch::kInt32).cuda();
    inputs.combo_tokens_host_for_log = inputs.combo_tokens.cpu().clone();
    inputs.input_lengths             = torch::tensor({4, 4}, torch::kInt32).cuda();
    inputs.prefix_lengths            = torch::tensor({0, 0}, torch::kInt32).cuda();

    PrefillChunkRound first_round;
    first_round.slices = {
        {0, 0, 2, 2, 0, 2, false},
        {1, 4, 6, 2, 0, 2, false},
    };
    PrefillChunkRound last_round;
    last_round.slices = {
        {0, 2, 4, 2, 2, 4, true},
        {1, 6, 8, 2, 2, 4, true},
    };

    auto first = components.executor->makePrefillRoundInput(inputs, first_round, /*total_tokens=*/8);
    components.executor->shiftRoundComboTokens(first, inputs, first_round);
    EXPECT_EQ(toVec<int32_t>(first.combo_tokens), (std::vector<int32_t>{1, 2, 5, 6}));
    EXPECT_EQ(toVec<int32_t>(first.combo_tokens_host_for_log), (std::vector<int32_t>{1, 2, 5, 6}));

    // The terminal round reaches the end of the packed batch, so its lookahead
    // shift would overrun and must be rejected (final rounds never shift).
    EXPECT_THROW(
        components.executor->shiftRoundComboTokens(first, inputs, last_round), std::runtime_error);
}

TEST_F(MtpExecutorTest, testBuildDraftCacheGroupTypesPreservesGlobalGroupNamespace) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    CacheConfig global_cache_config;
    global_cache_config.group_types = {
        CacheGroupType::FULL,
        CacheGroupType::LINEAR,
        CacheGroupType::SWA,
    };
    CacheLayerLayout draft_layout;
    draft_layout.layer_to_groups   = {2};
    draft_layout.layer_group_types = {CacheGroupType::FULL};

    auto group_types = components.executor->buildDraftCacheGroupTypes(global_cache_config, draft_layout);
    EXPECT_TRUE(group_types.is_pinned());
    EXPECT_EQ(toVec<int32_t>(group_types),
              (std::vector<int32_t>{static_cast<int32_t>(CacheGroupType::FULL),
                                    static_cast<int32_t>(CacheGroupType::LINEAR),
                                    static_cast<int32_t>(CacheGroupType::FULL)}));

    draft_layout.layer_to_groups = {3};
    EXPECT_THROW(components.executor->buildDraftCacheGroupTypes(global_cache_config, draft_layout), std::runtime_error);
}

TEST_F(MtpExecutorTest, testRunChunkPrefillRoundInterleavesTargetShiftAndDraft) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});

    // Make the draft semantics intentionally differ from the target metadata
    // gathered by the stream processor.  This reproduces K3 (target group 0
    // LINEAR, Eagle3 group 0 FULL) without constructing an invalid synthetic
    // hybrid cache layout in this CPU unit test.
    auto draft_group_types = torch::tensor({static_cast<int32_t>(CacheGroupType::FULL),
                                            static_cast<int32_t>(CacheGroupType::LINEAR),
                                            static_cast<int32_t>(CacheGroupType::FULL)},
                                           torch::kInt32)
                                 .pin_memory();
    auto draft_layer_to_group = torch::tensor({2}, torch::kInt32).pin_memory();
    components.executor->draft_kv_cache_group_types = draft_group_types;
    components.executor->draft_kv_cache_layer_to_group = draft_layer_to_group;

    GptModelInputs full_inputs;
    full_inputs.combo_tokens     = torch::arange(0, 8, torch::kInt32);
    full_inputs.input_lengths    = torch::tensor({8}, torch::kInt32);
    full_inputs.prefix_lengths   = torch::tensor({0}, torch::kInt32);
    full_inputs.sequence_lengths = torch::empty({0}, torch::kInt32);

    // The Python planner (page_size=2) schedules three rounds:
    // [4] -> [2] -> terminal [2].
    PrefillChunkRound first_round;
    first_round.slices = {{0, 0, 4, 4, 0, 4, false}};
    PrefillChunkRound middle_round;
    middle_round.slices = {{0, 4, 6, 2, 4, 6, false}};
    PrefillChunkRound final_round;
    final_round.slices = {{0, 6, 8, 2, 6, 8, true}};

    // Draft position p consumes target token p + 1, so each non-final draft
    // input is the round slice shifted by one token and re-pointed at the
    // draft cache groups.
    auto first_draft                      = GptModelInputs{};
    first_draft.combo_tokens              = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    first_draft.input_lengths             = torch::tensor({4}, torch::kInt32);
    first_draft.prefix_lengths            = torch::tensor({0}, torch::kInt32);
    first_draft.is_prefill_chunk          = true;
    first_draft.kv_cache_group_types       = draft_group_types;
    first_draft.kv_cache_group_types_host = draft_group_types;
    first_draft.kv_cache_layer_to_group      = draft_layer_to_group;
    first_draft.kv_cache_layer_to_group_host = draft_layer_to_group;

    auto middle_draft                      = GptModelInputs{};
    middle_draft.combo_tokens              = torch::tensor({5, 6}, torch::kInt32);
    middle_draft.input_lengths             = torch::tensor({2}, torch::kInt32);
    middle_draft.prefix_lengths            = torch::tensor({4}, torch::kInt32);
    middle_draft.is_prefill_chunk          = true;
    middle_draft.kv_cache_group_types       = draft_group_types;
    middle_draft.kv_cache_group_types_host = draft_group_types;
    middle_draft.kv_cache_layer_to_group      = draft_layer_to_group;
    middle_draft.kv_cache_layer_to_group_host = draft_layer_to_group;

    auto terminal_prefix_draft                         = GptModelInputs{};
    terminal_prefix_draft.combo_tokens                 = torch::tensor({7}, torch::kInt32);
    terminal_prefix_draft.input_lengths                = torch::tensor({1}, torch::kInt32);
    terminal_prefix_draft.prefix_lengths               = torch::tensor({6}, torch::kInt32);
    terminal_prefix_draft.is_prefill_chunk             = true;
    terminal_prefix_draft.kv_cache_group_types         = draft_group_types;
    terminal_prefix_draft.kv_cache_group_types_host    = draft_group_types;
    terminal_prefix_draft.kv_cache_layer_to_group      = draft_layer_to_group;
    terminal_prefix_draft.kv_cache_layer_to_group_host = draft_layer_to_group;

    components.fake_draft_model->setInputs({first_draft, middle_draft, terminal_prefix_draft});
    components.fake_draft_model->setOutputs({GptModelOutputs{}, GptModelOutputs{}, GptModelOutputs{}});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    MtpExecutor::ChunkPrefillContext hook;
    hook.full_inputs      = full_inputs;
    hook.total_tokens     = 8;
    hook.terminal_seen.assign(1, false);
    int64_t model_forward_us = 0;
    hook.model_forward_us = &model_forward_us;

    components.executor->runChunkPrefillRound(hook, first_round, /*is_last=*/false);
    components.executor->runChunkPrefillRound(hook, middle_round, /*is_last=*/false);
    components.executor->runChunkPrefillRound(hook, final_round, /*is_last=*/true);

    ASSERT_EQ(hook.terminal_round.slices.size(), 1);
    const auto& terminal = hook.terminal_round.slices[0];
    EXPECT_TRUE(terminal.terminal);
    EXPECT_EQ(terminal.source_start, 7);
    EXPECT_EQ(terminal.source_end, 8);
    EXPECT_EQ(terminal.new_length, 1);
    EXPECT_EQ(terminal.absolute_start, 7);
    EXPECT_EQ(terminal.absolute_end, 8);

    // force_disable_sp_run skips every draft pass but still records the same
    // final round; no draft publication frontier is expected to advance.
    MtpExecutor::ChunkPrefillContext no_sp_hook;
    auto no_sp_full_inputs = full_inputs;
    no_sp_full_inputs.force_disable_sp_run = true;
    no_sp_hook.full_inputs      = no_sp_full_inputs;
    no_sp_hook.total_tokens     = 8;
    no_sp_hook.terminal_seen.assign(1, false);
    no_sp_hook.model_forward_us = &model_forward_us;
    components.executor->runChunkPrefillRound(no_sp_hook, first_round, /*is_last=*/false);
    components.executor->runChunkPrefillRound(no_sp_hook, middle_round, /*is_last=*/false);
    components.executor->runChunkPrefillRound(no_sp_hook, final_round, /*is_last=*/true);
    ASSERT_EQ(no_sp_hook.terminal_round.slices.size(), 1);
    EXPECT_EQ(no_sp_hook.terminal_round.slices[0].source_start, 7);
}

TEST_F(MtpExecutorTest, testRunChunkPrefillRoundPropagatesDraftFailure) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    GptModelInputs full_inputs;
    full_inputs.combo_tokens     = torch::arange(0, 4, torch::kInt32);
    full_inputs.input_lengths    = torch::tensor({4}, torch::kInt32);
    full_inputs.prefix_lengths   = torch::tensor({0}, torch::kInt32);
    full_inputs.sequence_lengths = torch::empty({0}, torch::kInt32);

    MtpExecutor::ChunkPrefillContext hook;
    hook.full_inputs      = full_inputs;
    hook.total_tokens     = 4;
    hook.terminal_seen.assign(1, false);
    int64_t model_forward_us = 0;
    hook.model_forward_us = &model_forward_us;

    PrefillChunkRound first_round;
    first_round.slices = {{0, 0, 4, 4, 0, 4, false}};

    // No draft inputs/outputs queued: the draft forward throws, and the hook
    // must propagate it so prefillStep's session guard aborts the Python
    // chunk session.
    EXPECT_THROW(components.executor->runChunkPrefillRound(hook, first_round, /*is_last=*/false),
                 std::runtime_error);
}

TEST_F(MtpExecutorTest, testRunChunkPrefillRoundCollectsHeterogeneousTerminalTokens) {
    auto           components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    GptModelInputs full_inputs;
    full_inputs.combo_tokens         = torch::arange(0, 5, torch::kInt32);
    full_inputs.input_lengths        = torch::tensor({1, 4}, torch::kInt32);
    full_inputs.prefix_lengths       = torch::tensor({0, 0}, torch::kInt32);
    full_inputs.sequence_lengths     = torch::empty({0}, torch::kInt32);
    full_inputs.force_disable_sp_run = true;

    PrefillChunkRound first_round;
    first_round.slices = {
        {0, 0, 1, 1, 0, 1, true},
        {1, 1, 3, 2, 0, 2, false},
    };
    PrefillChunkRound last_round;
    last_round.slices = {{1, 3, 5, 2, 2, 4, true}};

    MtpExecutor::ChunkPrefillContext hook;
    hook.full_inputs  = full_inputs;
    hook.total_tokens = 5;
    hook.terminal_seen.assign(2, false);
    int64_t model_forward_us = 0;
    hook.model_forward_us    = &model_forward_us;

    components.executor->runChunkPrefillRound(hook, first_round, /*is_last=*/false);
    components.executor->runChunkPrefillRound(hook, last_round, /*is_last=*/true);
    ASSERT_EQ(hook.terminal_round.slices.size(), 2);
    std::sort(hook.terminal_round.slices.begin(),
              hook.terminal_round.slices.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.original_batch_idx < rhs.original_batch_idx; });

    auto terminal_input =
        components.executor->makePrefillRoundInput(full_inputs, hook.terminal_round, /*total_tokens=*/5);
    EXPECT_EQ(toVec<int32_t>(terminal_input.combo_tokens), (std::vector<int32_t>{0, 4}));
    EXPECT_EQ(toVec<int32_t>(terminal_input.input_lengths), (std::vector<int32_t>{1, 1}));
    EXPECT_EQ(toVec<int32_t>(terminal_input.prefix_lengths), (std::vector<int32_t>{0, 3}));
    EXPECT_EQ(toVec<int32_t>(terminal_input.lm_output_indexes), (std::vector<int32_t>{0, 1}));
}

TEST_F(MtpExecutorTest, testPrefillChunkCacheStorePublishPlanDefersAndRewritesTerminalPartialBlock) {
    auto           components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    GptModelInputs full_inputs;
    full_inputs.combo_tokens     = torch::arange(0, 5, torch::kInt32);
    full_inputs.input_lengths    = torch::tensor({1, 4}, torch::kInt32);
    full_inputs.prefix_lengths   = torch::tensor({0, 0}, torch::kInt32);
    full_inputs.sequence_lengths = torch::empty({0}, torch::kInt32);

    // Before the sampled token is available, request 1 has produced positions
    // [0, 3). Only its complete [0, 2) block may be published.
    PrefillChunkRound terminal_prefix_round;
    terminal_prefix_round.slices = {{1, 1, 4, 3, 0, 3, false}};
    auto terminal_prefix_input =
        components.executor->makePrefillRoundInput(full_inputs, terminal_prefix_round, /*total_tokens=*/5);
    std::vector<int32_t> publish_frontier{0, 0};
    components.executor->setPrefillChunkCacheStorePublishPlan(terminal_prefix_input,
                                                              terminal_prefix_round,
                                                              /*seq_size_per_block=*/2,
                                                              /*complete_blocks_only=*/true,
                                                              publish_frontier);
    EXPECT_EQ(toVec<int32_t>(terminal_prefix_input.prefix_lengths), (std::vector<int32_t>{0}));
    EXPECT_EQ(toVec<int32_t>(terminal_prefix_input.input_lengths), (std::vector<int32_t>{3}));
    ASSERT_TRUE(terminal_prefix_input.cache_store_publish_plan.has_value());
    EXPECT_EQ(toVec<int32_t>(terminal_prefix_input.cache_store_publish_plan->begin_block_host),
              (std::vector<int32_t>{0}));
    EXPECT_EQ(toVec<int32_t>(terminal_prefix_input.cache_store_publish_plan->end_block_host),
              (std::vector<int32_t>{1}));
    EXPECT_EQ(toVec<bool>(terminal_prefix_input.cache_store_publish_plan->terminal_host),
              (std::vector<bool>{false}));
    components.executor->advanceDraftCacheStorePublishFrontier(
        terminal_prefix_input, terminal_prefix_round, publish_frontier);
    EXPECT_EQ(publish_frontier, (std::vector<int32_t>{0, 1}));

    // The final draft pass still attends at the real singleton prefixes 0 and
    // 3, while cache store publishes the remaining block ranges [0, 1) and
    // [1, 2) from the two independent request frontiers.
    PrefillChunkRound terminal_round;
    terminal_round.slices = {
        {0, 0, 1, 1, 0, 1, true},
        {1, 4, 5, 1, 3, 4, true},
    };
    auto terminal_input =
        components.executor->makePrefillRoundInput(full_inputs, terminal_round, /*total_tokens=*/5);
    components.executor->setPrefillChunkCacheStorePublishPlan(terminal_input,
                                                              terminal_round,
                                                              /*seq_size_per_block=*/2,
                                                              /*complete_blocks_only=*/false,
                                                              publish_frontier);
    EXPECT_EQ(toVec<int32_t>(terminal_input.prefix_lengths), (std::vector<int32_t>{0, 3}));
    EXPECT_EQ(toVec<int32_t>(terminal_input.input_lengths), (std::vector<int32_t>{1, 1}));
    ASSERT_TRUE(terminal_input.cache_store_publish_plan.has_value());
    EXPECT_EQ(toVec<int32_t>(terminal_input.cache_store_publish_plan->begin_block_host),
              (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(toVec<int32_t>(terminal_input.cache_store_publish_plan->end_block_host),
              (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(toVec<bool>(terminal_input.cache_store_publish_plan->terminal_host),
              (std::vector<bool>{true, true}));
    components.executor->advanceDraftCacheStorePublishFrontier(
        terminal_input, terminal_round, publish_frontier);
    EXPECT_EQ(publish_frontier, (std::vector<int32_t>{1, 2}));

    // Regression for full-model PD smoke's first chunked stage. These requests
    // enter the terminal singleton with no prior draft publication, so the
    // transfer view must cover the reused prefix from block 0 through the
    // terminal partial block.
    PrefillChunkRound smoke_terminal_round;
    smoke_terminal_round.slices = {
        {0, 0, 1, 1, 9651, 9652, true},
        {1, 1, 2, 1, 16051, 16052, true},
        {2, 2, 3, 1, 22451, 22452, true},
        {3, 3, 4, 1, 28851, 28852, true},
    };
    GptModelInputs smoke_terminal_input;
    smoke_terminal_input.input_lengths   = torch::tensor({1, 1, 1, 1}, torch::kInt32);
    smoke_terminal_input.prefix_lengths  = torch::tensor({9651, 16051, 22451, 28851}, torch::kInt32);
    std::vector<int32_t> smoke_publish_frontier(4, 0);
    components.executor->setPrefillChunkCacheStorePublishPlan(smoke_terminal_input,
                                                               smoke_terminal_round,
                                                               /*seq_size_per_block=*/4096,
                                                              /*complete_blocks_only=*/false,
                                                              smoke_publish_frontier);
    ASSERT_TRUE(smoke_terminal_input.cache_store_publish_plan.has_value());
    EXPECT_EQ(toVec<int32_t>(smoke_terminal_input.cache_store_publish_plan->begin_block_host),
              (std::vector<int32_t>{0, 0, 0, 0}));
    EXPECT_EQ(toVec<int32_t>(smoke_terminal_input.cache_store_publish_plan->end_block_host),
              (std::vector<int32_t>{3, 4, 6, 8}));
    EXPECT_EQ(toVec<bool>(smoke_terminal_input.cache_store_publish_plan->terminal_host),
              (std::vector<bool>{true, true, true, true}));
}

TEST_F(MtpExecutorTest, testPrefillChunkCacheStorePublishPlanPublishesReusedPrefixBeforeTerminal) {
    auto              components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    constexpr int32_t block_size = 4096;

    // Production regression: Prefill starts draft computation at block 25,
    // while Decode still needs block 24. The first publication must cover the
    // complete reused prefix [0, 25), even though the current draft slice lies
    // wholly inside block 25 and produces no newly complete block.
    PrefillChunkRound prefix_round;
    prefix_round.slices = {{0, 0, 1957, 1957, 25 * block_size, 104357, false}};
    GptModelInputs prefix_input;
    prefix_input.input_lengths  = torch::tensor({1957}, torch::kInt32);
    prefix_input.prefix_lengths = torch::tensor({25 * block_size}, torch::kInt32);
    std::vector<int32_t> publish_frontier{0};
    components.executor->setPrefillChunkCacheStorePublishPlan(prefix_input,
                                                              prefix_round,
                                                              block_size,
                                                              /*complete_blocks_only=*/true,
                                                              publish_frontier);
    ASSERT_TRUE(prefix_input.cache_store_publish_plan.has_value());
    EXPECT_EQ(toVec<int32_t>(prefix_input.cache_store_publish_plan->begin_block_host), (std::vector<int32_t>{0}));
    EXPECT_EQ(toVec<int32_t>(prefix_input.cache_store_publish_plan->end_block_host), (std::vector<int32_t>{25}));
    components.executor->advanceDraftCacheStorePublishFrontier(prefix_input, prefix_round, publish_frontier);
    EXPECT_EQ(publish_frontier, (std::vector<int32_t>{25}));

    PrefillChunkRound terminal_round;
    terminal_round.slices = {{0, 1957, 1958, 1, 104357, 104358, true}};
    GptModelInputs terminal_input;
    terminal_input.input_lengths  = torch::tensor({1}, torch::kInt32);
    terminal_input.prefix_lengths = torch::tensor({104357}, torch::kInt32);
    components.executor->setPrefillChunkCacheStorePublishPlan(terminal_input,
                                                              terminal_round,
                                                              block_size,
                                                              /*complete_blocks_only=*/false,
                                                              publish_frontier);
    ASSERT_TRUE(terminal_input.cache_store_publish_plan.has_value());
    EXPECT_EQ(toVec<int32_t>(terminal_input.cache_store_publish_plan->begin_block_host), (std::vector<int32_t>{25}));
    EXPECT_EQ(toVec<int32_t>(terminal_input.cache_store_publish_plan->end_block_host), (std::vector<int32_t>{26}));
    components.executor->advanceDraftCacheStorePublishFrontier(terminal_input, terminal_round, publish_frontier);
    EXPECT_EQ(publish_frontier, (std::vector<int32_t>{26}));
}

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
    draft_input_1.sequence_lengths   = torch::tensor({3}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_1.last_hidden_states = stream1_hidden_states;

    draft_input_2.combo_tokens       = torch::tensor({2}, torch::kInt32);
    draft_input_2.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_2.sequence_lengths   = torch::tensor({4}, torch::kInt32);
    draft_input_2.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = torch::tensor({1}, torch::kInt32);
    draft_input_3.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_3.sequence_lengths   = torch::tensor({5}, torch::kInt32);
    draft_input_3.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    auto next_draft_input    = GptModelInputs{};
    auto next_draft_output   = GptModelOutputs{};
    next_draft_output.logits = torch::tensor({1.9f, 1.10f, 1.11f, 1.12f}).reshape({(int64_t)batch_size, 4});
    next_draft_output.all_hidden_states =
        torch::tensor({0.1f, 0.1f, 0.2f, 0.22f, 0.3f, 0.33f, 0.0f, 0.0f, 0.0f, 0.0f}).reshape({5, 2});

    next_draft_input.combo_tokens      = torch::tensor({3, 2, 0, 0, 0}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({5}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    next_draft_input.input_lengths_host_for_log  = torch::tensor({5}, torch::kInt32);
    next_draft_input.prefix_lengths_host_for_log = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({2}, torch::kInt32);

    // set fake model outputs
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3, 2, 1, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({5}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.input_lengths_host_for_log  = torch::tensor({5}, torch::kInt32);
    target_input.prefix_lengths_host_for_log = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3, 4}, torch::kInt32);

    auto target_prepare_input                    = GptModelInputs{};
    target_prepare_input.input_lengths           = torch::tensor({5}, torch::kInt32);
    target_prepare_input.prefix_lengths          = torch::tensor({2}, torch::kInt32);
    target_prepare_input.sequence_lengths_plus_1 = torch::tensor({3}, torch::kInt32);
    target_prepare_input.lm_output_indexes       = torch::tensor({0}, torch::kInt32);
    components.fake_target_model->setPrepareInputs({target_prepare_input});

    target_output.logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f, 2.1f, 2.2f,
                                          2.3f, 2.4f, 3.1f, 3.2f, 3.3f, 3.4f, 4.1f, 4.2f, 4.3f, 4.4f})
                               .reshape({(int64_t)(batch_size * (propose_step + 1)), 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f})
            .reshape({(int64_t)(propose_step + 1), 2});

    next_draft_input.last_hidden_states = target_output.all_hidden_states;

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

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
    auto accept_tokens                           = torch::tensor({{3, 2, 0, 0, 0}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = accept_tokens;
    speculative_sampler_output.accept_tokens     = accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({3}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    auto draft_spec_sample_input                 = SamplerOutput{};
    auto target_spec_sample_input                = SamplerOutput{};

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

    auto* fake_target_model = components.fake_target_model.get();

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
    const auto* async_prepare_env       = std::getenv("RTP_LLM_MTP_ASYNC_PREPARE");
    const bool  async_prepare_requested =
        async_prepare_env != nullptr && std::strcmp(async_prepare_env, "1") == 0;
    EXPECT_EQ(components.executor->useAsyncPrepare(), async_prepare_requested);
    if (async_prepare_requested) {
        EXPECT_FALSE(fake_target_model->hasPendingPrepareInputs());
    }

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 2, 0}, {0, 1}, {0.0, 1.0, 0.0, 0.0}, {0.3, 0.33});
}

TEST_F(MtpExecutorTest, testTargetVerifyHostMetadataMatchesPackedInput) {
    GptModelInputs target;
    auto prefix_lengths_host =
        torch::tensor({126, 255}, torch::TensorOptions(torch::kInt32).pinned_memory(true));

    MtpExecutor::populateTargetVerifyHostMetadata(target, prefix_lengths_host, 2, 4);

    EXPECT_TRUE(target.input_lengths_host_for_log.defined());
    EXPECT_FALSE(target.input_lengths_host_for_log.is_cuda());
    EXPECT_TRUE(target.input_lengths_host_for_log.is_pinned());
    EXPECT_EQ(std::vector<int32_t>({4, 4}), toVec<int32_t>(target.input_lengths_host_for_log));
    EXPECT_EQ(std::vector<int32_t>({126, 255}), toVec<int32_t>(target.prefix_lengths_host_for_log));
    EXPECT_FALSE(target.sequence_lengths_host_for_log.defined());
}

// One-step MTP clears sequence_lengths_host_for_log before this helper runs, so it
// must tolerate an absent mirror instead of slicing an undefined tensor.
TEST_F(MtpExecutorTest, testTargetVerifyHostMetadataToleratesMissingMirror) {
    GptModelInputs target;

    MtpExecutor::populateTargetVerifyHostMetadata(target, torch::Tensor(), 2, 4);

    EXPECT_TRUE(target.input_lengths_host_for_log.defined());
    EXPECT_EQ(std::vector<int32_t>({4, 4}), toVec<int32_t>(target.input_lengths_host_for_log));
    EXPECT_FALSE(target.prefix_lengths_host_for_log.defined());
    EXPECT_FALSE(target.sequence_lengths_host_for_log.defined());
}

// A mirror shorter than the verify batch cannot be sliced safely either.
TEST_F(MtpExecutorTest, testTargetVerifyHostMetadataRejectsShortMirror) {
    GptModelInputs target;
    auto           prefix_lengths_host =
        torch::tensor({126}, torch::TensorOptions(torch::kInt32).pinned_memory(true));

    MtpExecutor::populateTargetVerifyHostMetadata(target, prefix_lengths_host, 2, 4);

    EXPECT_FALSE(target.prefix_lengths_host_for_log.defined());
}

TEST_F(MtpExecutorTest, testDecodeSpecLogitsCapReplacesInvalidDraftWithTargetToken) {
    size_t propose_step = 2;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    auto                 stream_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto                 stream_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto                 stream_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});
    StreamSpecUpdateInfo spec_update_info{stream_new_tokens, 1, 3, stream_hidden_states, stream_draft_token_probs};

    GenerateStreamPtr stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    stream->logits_processor_list_.push_back(
        std::make_shared<RejectDraftTokenSpecProcessor>(3, stream->outputTokenLen()));

    auto draft_input_1               = GptModelInputs{};
    auto draft_output_1              = GptModelOutputs{};
    draft_input_1.combo_tokens       = torch::tensor({3}, torch::kInt32);
    draft_input_1.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_1.sequence_lengths   = torch::tensor({3}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_1.last_hidden_states = stream_hidden_states;
    draft_output_1.logits            = torch::tensor({0.4f, 0.3f, 0.2f, 0.1f}).reshape({1, 4});
    draft_output_1.all_hidden_states = torch::tensor({0.11f, 0.12f}).reshape({1, 2});

    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3, 0}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({3}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2}, torch::kInt32);
    target_output.logits = torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.2f, 0.1f, 0.8f, 0.4f, 0.7f, 0.2f, 0.1f, 0.0f})
                               .reshape({3, 4})
                               .to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f}).reshape({3, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({1, 0, 0}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({3}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states;
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f}).reshape({3, 2});

    components.fake_draft_model->setInputs({draft_input_1, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, next_draft_output});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    auto draft_sampler_output_1 = spec::FastTopKSamplerOutput{torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({1, 4}),
                                                              torch::tensor({0}, torch::kInt32).reshape({1, 1})};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}), torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({draft_output_1.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output_1, next_draft_sampler_output});

    auto sampler_input         = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3] = BaseLogitsProcessor::neg_inf;
    auto target_sampler_output = SamplerOutput{torch::tensor({1, 2, 2}, torch::kInt32).reshape({3, 1})};
    target_sampler_output.all_probs =
        torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({3, 4});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({target_sampler_output});

    auto forced_accept_tokens                    = torch::tensor({{3, 0, 0}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = forced_accept_tokens;
    speculative_sampler_output.accept_tokens     = forced_accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({1}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    auto status = components.executor->process({stream});
    ASSERT_TRUE(status.ok());

    checkOutput(stream, {0, 1, 2, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {0.21, 0.22});
}

TEST_F(MtpExecutorTest, testDecodeOneStepSpecLogitsCapReplacesInvalidDraftWithTargetToken) {
    size_t propose_step = 1;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    auto                 stream_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto                 stream_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto                 stream_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});
    StreamSpecUpdateInfo spec_update_info{stream_new_tokens, 1, 3, stream_hidden_states, stream_draft_token_probs};

    GenerateStreamPtr stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    stream->logits_processor_list_.push_back(
        std::make_shared<RejectDraftTokenSpecProcessor>(3, stream->outputTokenLen()));

    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.7f, 0.2f, 0.1f, 0.0f}).reshape({2, 4}).to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({1, 2}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({2}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states;
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f, 0.23f, 0.24f}).reshape({2, 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setInputs({next_draft_input});
    components.fake_draft_model->setOutputs({next_draft_output});

    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}), torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({next_draft_sampler_output});

    auto sampler_input              = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3]      = BaseLogitsProcessor::neg_inf;
    auto target_sampler_output      = SamplerOutput{torch::tensor({1, 2}, torch::kInt32).reshape({2, 1})};
    target_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({2, 4});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({target_sampler_output});

    auto forced_accept_tokens                    = torch::tensor({{3, 2}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = forced_accept_tokens;
    speculative_sampler_output.accept_tokens     = forced_accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({2}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    auto status = components.executor->process({stream});
    ASSERT_TRUE(status.ok());

    checkOutput(stream, {0, 1, 2, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {});
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
    draft_input_1.sequence_lengths   = torch::tensor({4, 3}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_1.last_hidden_states = torch::tensor({0.03f, 0.04f, 2.1f, 2.12f}).reshape({2, 2});

    draft_input_2.combo_tokens       = torch::tensor({1, 0}, torch::kInt32);
    draft_input_2.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_2.sequence_lengths   = torch::tensor({5, 4}, torch::kInt32);
    draft_input_2.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = torch::tensor({2, 2}, torch::kInt32);
    draft_input_3.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_3.sequence_lengths   = torch::tensor({6, 5}, torch::kInt32);
    draft_input_3.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    // accept [3], [3, 0, 2, 2, 1]
    auto next_draft_input  = GptModelInputs{};
    auto next_draft_output = GptModelOutputs{};
    next_draft_output.logits =
        torch::tensor({1.9f, 1.10f, 1.11f, 1.12f, 2.9f, 2.10f, 2.11f, 2.12f}).reshape({(int64_t)batch_size, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.1f, 0.11f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                         0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.5f, 1.55f})
                                              .reshape({10, 2});

    next_draft_input.combo_tokens      = torch::tensor({3, 0, 0, 0, 0, 3, 0, 2, 2, 1}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({5, 5}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({3, 2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({0, 9}, torch::kInt32);

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

    next_draft_input.last_hidden_states = target_output.all_hidden_states;

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

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
    auto accept_tokens                           = torch::tensor({{3, 0, 0, 0, 0}, {3, 0, 2, 2, 1}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = accept_tokens;
    speculative_sampler_output.accept_tokens     = accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({1, 5}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    auto draft_spec_sample_input                 = SamplerOutput{};
    auto target_spec_sample_input                = SamplerOutput{};

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

TEST_F(MtpExecutorTest, testDispatchStatePrepareKernel) {
    // Test invokeMtpDispatchStatePrepare correctness
    const int64_t batch_size = 8;
    auto          cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto          cuda_i64   = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    auto accept_len   = torch::tensor({3, 1, 5, 2, 4, 1, 3, 2}, cuda_i32);
    auto prev_seq_len = torch::tensor({100, 200, 50, 300, 150, 400, 75, 250}, cuda_i32);
    auto next_seq_len = torch::empty({batch_size}, cuda_i32);
    auto hidden_idx   = torch::empty({batch_size}, cuda_i64);

#if USING_CUDA
    invokeMtpDispatchStatePrepare(
        accept_len, prev_seq_len, next_seq_len, hidden_idx, batch_size, at::cuda::getCurrentCUDAStream().stream());
    cudaDeviceSynchronize();
#endif

    // Verify next_seq_len = prev_seq_len + accept_len
    auto expected_next = (prev_seq_len + accept_len).cpu();
    auto actual_next   = next_seq_len.cpu();
    EXPECT_TRUE(torch::equal(actual_next, expected_next)) << "next_seq_len mismatch:\n"
                                                          << actual_next << "\nvs expected:\n"
                                                          << expected_next;

    // Verify hidden_idx = accept_len - 1
    auto expected_idx = (accept_len.to(torch::kInt64) - 1).cpu();
    auto actual_idx   = hidden_idx.cpu();
    EXPECT_TRUE(torch::equal(actual_idx, expected_idx)) << "hidden_idx mismatch:\n"
                                                        << actual_idx << "\nvs expected:\n"
                                                        << expected_idx;
}

TEST_F(MtpExecutorTest, testLinearKvCacheBlockPatchKernel) {
    constexpr int64_t group_num   = 4;
    constexpr int64_t batch_size  = 6;
    constexpr int64_t row_width   = 16;  // Simulates global FULL-group BPK > 1.
    constexpr int64_t patch_width = 4;
    constexpr int32_t page_size   = 4;

    auto block_ids_cpu =
        torch::arange(group_num * batch_size * row_width, torch::kInt32).reshape({group_num, batch_size, row_width});
    block_ids_cpu[3][0][1] = -1;
    block_ids_cpu[3][0][2] = -1;
    auto expected          = block_ids_cpu.clone();
    auto group_types_cpu   = torch::tensor({0, 1, 2, 0}, torch::kInt32);
    auto valid_counts_cpu  = torch::full({group_num, batch_size}, 12, torch::kInt32);
    valid_counts_cpu[0][5] = 2;
    valid_counts_cpu[3][5] = 2;
    auto prev_seq_len_cpu  = torch::tensor({3, 5, 8, 11, 4, 3}, torch::kInt32);
    auto accept_len_cpu    = torch::tensor({3, 2, 5, 4, 1, 3}, torch::kInt32);
    auto pending_cpu       = torch::tensor({1, 1, 1, 0, 1, 1}, torch::kInt32);

    for (int64_t group_id = 0; group_id < group_num; ++group_id) {
        if (group_types_cpu[group_id].item<int32_t>() != static_cast<int32_t>(CacheGroupType::LINEAR)) {
            continue;
        }
        for (int64_t batch_id = 0; batch_id < batch_size; ++batch_id) {
            const int32_t accepted = accept_len_cpu[batch_id].item<int32_t>();
            if (pending_cpu[batch_id].item<int32_t>() == 0 || accepted <= 1) {
                continue;
            }
            const int32_t cur_cached_len        = prev_seq_len_cpu[batch_id].item<int32_t>() - 1;
            const int32_t nxt_cached_len        = cur_cached_len + accepted;
            const auto [cached_src, cached_dst] = getCachedTokenBlockSwapIdx(cur_cached_len, nxt_cached_len, page_size);
            const auto [final_src, final_dst]   = getFinalTokenBlockSwapIdx(cur_cached_len, nxt_cached_len, page_size);
            const int32_t valid_count           = valid_counts_cpu[group_id][batch_id].item<int32_t>();
            if (cached_src >= valid_count || cached_dst >= valid_count || final_src >= valid_count
                || final_dst >= valid_count) {
                continue;
            }
            auto* row = expected[group_id][batch_id].data_ptr<int32_t>();
            std::swap(row[cached_src], row[cached_dst]);
            std::swap(row[final_src], row[final_dst]);
        }
    }

    auto block_ids          = block_ids_cpu.to(torch::kCUDA);
    auto group_types        = group_types_cpu.to(torch::kCUDA);
    auto valid_block_counts = valid_counts_cpu.to(torch::kCUDA);
    auto prev_seq_len       = prev_seq_len_cpu.to(torch::kCUDA);
    auto accept_len         = accept_len_cpu.to(torch::kCUDA);
    auto pending            = pending_cpu.to(torch::kCUDA);
    auto cuda_i32           = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto positions          = torch::empty({batch_size, patch_width}, cuda_i32);
    auto source_slots       = torch::empty({batch_size, patch_width}, cuda_i32);
    auto before_values      = torch::empty({batch_size, group_num, patch_width}, cuda_i32);
    auto after_values       = torch::empty({batch_size, group_num, patch_width}, cuda_i32);
    auto patch_valid        = torch::empty({batch_size, group_num}, cuda_i32);

#if USING_CUDA
    invokeMtpLinearKvCacheBlockPatchBuild(block_ids,
                                          group_types,
                                          valid_block_counts,
                                          prev_seq_len,
                                          accept_len,
                                          positions,
                                          source_slots,
                                          before_values,
                                          after_values,
                                          patch_valid,
                                          page_size,
                                          at::cuda::getCurrentCUDAStream().stream());
    invokeMtpLinearKvCacheBlockPatchApply(block_ids,
                                          group_types,
                                          valid_block_counts,
                                          positions,
                                          source_slots,
                                          before_values,
                                          after_values,
                                          patch_valid,
                                          pending,
                                          at::cuda::getCurrentCUDAStream().stream());
#endif

    EXPECT_TRUE(torch::equal(block_ids.cpu(), expected));

    // The overlapping pairs (1,0) then (2,1) form a 3-cycle. The patch must
    // store all three final values, not two independent destination values.
    EXPECT_EQ(toVec<int32_t>(positions.cpu()[0]), (std::vector<int32_t>{1, 0, 2, -1}));
    EXPECT_EQ(toVec<int32_t>(source_slots.cpu()[0]), (std::vector<int32_t>{2, 0, 1, -1}));
    EXPECT_EQ(toVec<int32_t>(before_values.cpu()[0][0]), (std::vector<int32_t>{1, 0, 2, -1}));
    EXPECT_EQ(toVec<int32_t>(after_values.cpu()[0][0]), (std::vector<int32_t>{2, 1, 0, -1}));
    EXPECT_EQ(patch_valid.cpu()[0][3].item<int32_t>(), 1);

#if USING_CUDA
    // Applying the same final-value patch again, or applying it to a host table
    // where the worker already committed the swap, is a no-op.
    invokeMtpLinearKvCacheBlockPatchApply(block_ids,
                                          group_types,
                                          valid_block_counts,
                                          positions,
                                          source_slots,
                                          before_values,
                                          after_values,
                                          patch_valid,
                                          pending,
                                          at::cuda::getCurrentCUDAStream().stream());
#endif
    EXPECT_TRUE(torch::equal(block_ids.cpu(), expected));

    auto already_committed = expected.to(torch::kCUDA);
#if USING_CUDA
    invokeMtpLinearKvCacheBlockPatchApply(already_committed,
                                          group_types,
                                          valid_block_counts,
                                          positions,
                                          source_slots,
                                          before_values,
                                          after_values,
                                          patch_valid,
                                          pending,
                                          at::cuda::getCurrentCUDAStream().stream());
#endif
    EXPECT_TRUE(torch::equal(already_committed.cpu(), expected));

    // A fresh allocator value in a touched slot must be permuted with the
    // current tuple, not overwritten by a stale saved block ID.
    auto allocator_edited_cpu      = block_ids_cpu.clone();
    allocator_edited_cpu[0][0][2]  = 9999;
    auto allocator_edited_expected = expected.clone();
    allocator_edited_expected[0][0].copy_(allocator_edited_cpu[0][0]);
    auto* allocator_edited_row = allocator_edited_expected[0][0].data_ptr<int32_t>();
    std::swap(allocator_edited_row[1], allocator_edited_row[0]);
    std::swap(allocator_edited_row[2], allocator_edited_row[1]);
    auto allocator_edited = allocator_edited_cpu.to(torch::kCUDA);
#if USING_CUDA
    invokeMtpLinearKvCacheBlockPatchApply(allocator_edited,
                                          group_types,
                                          valid_block_counts,
                                          positions,
                                          source_slots,
                                          before_values,
                                          after_values,
                                          patch_valid,
                                          pending,
                                          at::cuda::getCurrentCUDAStream().stream());
#endif
    EXPECT_TRUE(torch::equal(allocator_edited.cpu(), allocator_edited_expected));
}

TEST_F(MtpExecutorTest, testLinearKvCacheSnapshotEpochPreventsDoubleSwap) {
    auto cache_config = test::makeSimpleHybridMhaCacheConfig(
        /*layer_num=*/4, /*block_num=*/64, /*tokens_per_block=*/4, TYPE_INT8, /*group_layer_num=*/2);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ModelConfig model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 32;
    model_config.num_layers  = 4;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    auto                 stream = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3});
    BatchKVCacheResource kv_cache;
    kv_cache.resetBatchSize(1);
    kv_cache.initGroups(cache_config.groupNums(),
                        model_config.num_layers,
                        cache_config.layer_to_group_id,
                        cache_config.kernelBlocksPerKvBlock(),
                        cache_config.group_types);
    kv_cache.setBatchBlocks(0, 0, {10, 11, 12, 13});
    kv_cache.setBatchBlocks(0, 1, {20, 21, 22, 23});
    stream->setKVCache(kv_cache);
    stream->setNeedReleaseResource(false);

    auto sp_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_buffer->tokens = torch::full({1, 2}, -1, torch::kInt32);
    stream->setSPOutputBuffer(sp_buffer);

    GenerateStream::MtpAsyncDeviceState state;
    state.accept_len_gpu   = torch::tensor({3}, torch::kInt32).to(torch::kCUDA);
    state.prev_seq_len_gpu = torch::tensor({3}, torch::kInt32).to(torch::kCUDA);
    const auto epoch       = stream->setMtpAsyncDeviceState(std::move(state), true);

    auto before = stream->snapshotKVCacheBlocks();
    EXPECT_TRUE(before.needs_mtp_linear_patch);
    EXPECT_EQ(before.kernel_blocks[0][0], (BlockIndicesType{10, 11, 12, 13}));

    StreamSpecUpdateInfo update_info{torch::tensor({{4, 5, 6}}, torch::kInt32),
                                     3,
                                     7,
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     true,
                                     false,
                                     epoch};
    stream->specUpdate(update_info);

    auto after = stream->snapshotKVCacheBlocks();
    EXPECT_FALSE(after.needs_mtp_linear_patch);
    EXPECT_EQ(after.kernel_blocks[0][0], (BlockIndicesType{11, 12, 10, 13}));
    EXPECT_EQ(after.kernel_blocks[0][1], before.kernel_blocks[0][1]);
}

TEST_F(MtpExecutorTest, testDispatchStatePrepareBenchmark) {
    // Micro-benchmark: compare per-stream scalar ops vs batched approach
    const int64_t batch_size = 128;
    const int     iterations = 1000;
    auto          cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto          cuda_i64   = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    auto accept_len   = torch::randint(1, 5, {batch_size}, cuda_i32);
    auto prev_seq_len = torch::randint(10, 1000, {batch_size}, cuda_i32);

    // Pre-allocate output buffers
    auto next_seq_len = torch::empty({batch_size}, cuda_i32);
    auto hidden_idx   = torch::empty({batch_size}, cuda_i64);

    // Warm up
    for (int i = 0; i < 10; i++) {
#if USING_CUDA
        invokeMtpDispatchStatePrepare(
            accept_len, prev_seq_len, next_seq_len, hidden_idx, batch_size, at::cuda::getCurrentCUDAStream().stream());
#endif
    }
    cudaDeviceSynchronize();

    // Benchmark batched approach (fused kernel)
    auto start_batched = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
#if USING_CUDA
        invokeMtpDispatchStatePrepare(
            accept_len, prev_seq_len, next_seq_len, hidden_idx, batch_size, at::cuda::getCurrentCUDAStream().stream());
#endif
    }
    cudaDeviceSynchronize();
    auto end_batched = std::chrono::high_resolution_clock::now();
    auto us_batched  = std::chrono::duration_cast<std::chrono::microseconds>(end_batched - start_batched).count();

    // Benchmark per-stream scalar approach (old way)
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        for (int64_t j = 0; j < batch_size; j++) {
            auto al_slice     = accept_len.narrow(0, j, 1);
            auto prev_slice   = prev_seq_len.narrow(0, j, 1);
            auto next_slice   = (prev_slice + al_slice).to(torch::kInt32);
            auto hidden_slice = (al_slice - 1).to(torch::kLong);
            (void)next_slice;
            (void)hidden_slice;
        }
    }
    cudaDeviceSynchronize();
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto us_scalar  = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count();

    double speedup = static_cast<double>(us_scalar) / static_cast<double>(us_batched);
    RTP_LLM_LOG_INFO("[dispatch-bench] batch_size=%ld iterations=%d", batch_size, iterations);
    RTP_LLM_LOG_INFO(
        "[dispatch-bench] batched: %ld us total, %.2f us/iter", us_batched, (double)us_batched / iterations);
    RTP_LLM_LOG_INFO("[dispatch-bench] scalar:  %ld us total, %.2f us/iter", us_scalar, (double)us_scalar / iterations);
    RTP_LLM_LOG_INFO("[dispatch-bench] speedup: %.1fx", speedup);
}

}  // namespace rtp_llm
