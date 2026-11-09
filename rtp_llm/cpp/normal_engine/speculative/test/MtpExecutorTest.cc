#include <algorithm>
#include <memory>
#include <limits>
#include <chrono>
#include "torch/all.h"
#include "autil/EnvUtil.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/MtpFusion.h"

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
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_token_gather.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
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
    size_t cache_block_num     = 10;
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
        if (forward_callback_) {
            forward_callback_();
        }
        return output_holder.get();
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

    void setForwardCallback(std::function<void()> callback) {
        forward_callback_ = std::move(callback);
    }

private:
    TestDataHolder<GptModelInputs>  input_holder;
    TestDataHolder<GptModelOutputs> output_holder;
    std::function<void()>           forward_callback_;
};

class FakeFastTopKSampler: public spec::FastTopKSampler {
public:
    FakeFastTopKSampler(): spec::FastTopKSampler(torch::Tensor()) {}

    spec::FastTopKSamplerOutput
    forward(const torch::Tensor& logits, int top_k = 1, const spec::MtpDraftSampleContext* = nullptr) override {
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
        if (forward_callback_) {
            forward_callback_();
        }
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

    void setForwardCallback(std::function<void()> callback) {
        forward_callback_ = std::move(callback);
    }

private:
    TestDataHolder<SamplerInputs> input_holder;
    TestDataHolder<SamplerOutput> output_holder;
    std::function<void()>         forward_callback_;
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
        auto update = spec_update_info;
        if (update.num_new_tokens > 0) {
            const auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            update.draft_token_gpu = torch::tensor({update.draft_token}, cuda_i32);
            update.target_token_gpu =
                update.new_tokens.reshape({-1}).narrow(0, update.num_new_tokens - 1, 1).to(cuda_i32);
        }
        stream->specUpdate(update);
        if (update.num_new_tokens > 0) {
            EXPECT_TRUE(sp_buffer->target_token_gpu.is_cuda());
            EXPECT_TRUE(sp_buffer->propose_tokens_gpu.is_cuda());
            checkTensorEqual(sp_buffer->target_token_gpu.reshape({-1}),
                             update.new_tokens.reshape({-1}).narrow(0, update.num_new_tokens - 1, 1));
            checkTensorEqual(sp_buffer->propose_tokens_gpu.reshape({-1}),
                             torch::tensor({update.draft_token}, torch::kInt32));
            checkTensorEqual(sp_buffer->target_token_gpu.reshape({-1}), sp_buffer->tokens.select(1, 0));
            checkTensorEqual(sp_buffer->propose_tokens_gpu.reshape({-1}), sp_buffer->tokens.select(1, 1));
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

    MtpExecutorComponents createMtpExecutorComponents(
        const MtpExecutorTestConfig&                    test_config,
        MtpExecutor::CacheStatusSnapshotRefreshCallback cache_status_snapshot_refresh_callback = {},
        bool                                            defer_connector_start                  = false) {
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
                                                                            /*block_num=*/test_config.cache_block_num,
                                                                            /*tokens_per_block=*/2,
                                                                            rtp_llm::TYPE_INT8,
                                                                            /*local_head_num_kv=*/128,
                                                                            /*size_per_head=*/256));

        auto cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                           /*block_num=*/test_config.cache_block_num,
                                                           /*tokens_per_block=*/2,
                                                           rtp_llm::TYPE_INT8,
                                                           /*local_head_num_kv=*/128,
                                                           /*size_per_head=*/256);

        auto mtp_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                         /*block_num=*/test_config.cache_block_num,
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

        // Create propose model engine init params
        auto mtp_model_params   = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        auto mtp_params         = std::make_unique<EngineInitParams>(params);
        mtp_params->py_sp_model = py::none();

        mtp_model_params->push_back(std::move(mtp_params));

        auto propose_params = std::make_unique<ProposeModelEngineInitParams>(
            SP_TYPE_MTP, sp_config.gen_num_per_cycle, std::move(mtp_model_params));

        // Create cache managers
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init(defer_connector_start);

        // Create MtpExecutor
        auto executor = std::make_unique<MtpExecutor>(params,
                                                      propose_params,
                                                      cache_manager,
                                                      MlaOpsType::AUTO,
                                                      /*kv_cache_group_num=*/1,
                                                      /*kv_cache_layer_to_group=*/std::vector<int32_t>{},
                                                      /*warm_up=*/false,
                                                      std::move(cache_status_snapshot_refresh_callback));

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

TEST_F(MtpExecutorTest, testForwardOverlapHiddenStateContract) {
    ModelConfig target;
    ModelConfig draft;
    target.hidden_size = draft.hidden_size = 8;
    target.hc_mult = draft.hc_mult = 1;
    EXPECT_TRUE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
    target.hc_mult = 4;
    EXPECT_FALSE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
    target.model_type = "hy_v4";
    EXPECT_FALSE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
    draft.model_type = "hy_v4_mtp";
    EXPECT_TRUE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
    target.attn_config.layer_compress_ratios = {1, 4};
    EXPECT_FALSE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
    target.attn_config.layer_compress_ratios.clear();
    draft.hc_mult = 4;
    EXPECT_FALSE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
    draft.hc_mult = 1;
    draft.hidden_size = 16;
    EXPECT_FALSE(MtpExecutor::supportsForwardOverlapHiddenState(target, draft));
}

TEST_F(MtpExecutorTest, testDraftRawLogitsOptimizationIsOptIn) {
    Sampler glm_sampler(SamplerInitParams{"glm5"});
    Sampler hy4_sampler(SamplerInitParams{"hy_v4"});
    EXPECT_FALSE(glm_sampler.optimizedSamplingEnabled());
    EXPECT_EQ(glm_sampler.force_topk_, 0);
    EXPECT_TRUE(hy4_sampler.optimizedSamplingEnabled());
    EXPECT_EQ(hy4_sampler.force_topk_, 4096);
    auto identity_map = torch::arange(2, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto logits = torch::tensor({{0.0f, 0.0001f}}, torch::TensorOptions().device(torch::kCUDA)).to(torch::kBFloat16);
    spec::FastTopKSampler legacy(identity_map);
    spec::FastTopKSampler optimized(identity_map, false, true);
    ASSERT_EQ(torch::softmax(logits, -1).argmax(-1).item<int64_t>(), 0);
    EXPECT_EQ(legacy.forward(logits).token_ids.item<int64_t>(), 0);
    EXPECT_EQ(optimized.forward(logits).token_ids.item<int64_t>(), 1);
}

TEST_F(MtpExecutorTest, testDeterministicDraftSamplerReportsPointMassProposal) {
    auto identity_map = torch::arange(4, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    spec::FastTopKSampler sampler(identity_map);
    auto                  logits =
        torch::tensor({{0.0f, 1.0f, 4.0f, 2.0f}}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto output = sampler.forward(logits);

    EXPECT_EQ(output.token_ids.item<int64_t>(), 2);
    checkTensorEqual(output.all_probs, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}}).to(torch::kCUDA));
}

#if USING_CUDA
TEST_F(MtpExecutorTest, testRejectionSamplingOverwritesPoisonedOutputs) {
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto f32 = i32.dtype(torch::kFloat32);
    for (int batch : {1, 4, 7}) {
        for (int steps : {1, 3, 7}) {
            for (bool sample : {false, true}) {
                for (int reject : {0, steps / 2, steps}) {
                    auto draft  = torch::full({batch, steps}, 2, i32);
                    auto target = torch::full({batch, steps + 1}, 2, i32);
                    target.select(1, steps).fill_(3);
                    if (reject < steps)
                        target.select(1, reject).fill_(3);
                    auto dp = torch::zeros({batch, steps, 32}, f32);
                    dp.select(2, 2).fill_(1.f);
                    auto tp = torch::zeros({batch, steps + 1, 32}, f32);
                    tp.scatter_(2, target.unsqueeze(2).to(torch::kInt64), 1.f);
                    auto uniform     = torch::full({batch, steps + 1}, 0.5f, f32);
                    auto do_sample   = torch::full({batch}, sample, i32.dtype(torch::kBool));
                    auto old_tokens  = torch::zeros_like(target);
                    auto old_lengths = torch::zeros({batch}, i32);
                    auto new_tokens  = torch::full_like(target, -777);
                    auto new_lengths = torch::full({batch}, -777, i32);
                    execRejectionSampling({dp,
                                           draft,
                                           uniform,
                                           tp,
                                           target.reshape({batch * (steps + 1), 1}),
                                           old_tokens,
                                           old_lengths,
                                           do_sample});
                    execRejectionSampling({dp,
                                           draft,
                                           uniform,
                                           tp,
                                           target.reshape({batch * (steps + 1), 1}),
                                           new_tokens,
                                           new_lengths,
                                           do_sample});
                    EXPECT_TRUE(torch::equal(old_tokens, new_tokens));
                    EXPECT_TRUE(torch::equal(old_lengths, new_lengths));
                    EXPECT_FALSE(new_tokens.eq(-777).any().item<bool>());
                    EXPECT_TRUE(new_lengths.eq(reject + 1).all().item<bool>());
                }
            }
        }
    }
}

TEST_F(MtpExecutorTest, testCompactPointMassProposalAndRejection) {
    const auto            i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto            f32 = i32.dtype(torch::kFloat32);
    spec::FastTopKSampler dense(torch::Tensor(), false, true);
    spec::FastTopKSampler compact(torch::Tensor(), true, true);
    for (int batch : {1, 4, 17}) {
        for (int width : {7, 2048, 154880}) {
            for (int pattern : {0, 1, 2, 3, 4, 5}) {
                auto logits = torch::randn({batch, width}, f32);
                if (pattern == 1)
                    logits.zero_();
                if (pattern == 2)
                    logits.select(1, width - 1).fill_(std::numeric_limits<float>::quiet_NaN());
                if (pattern == 3)
                    logits.fill_(-std::numeric_limits<float>::infinity());
                if (pattern == 4) {
                    logits.select(1, 0).fill_(std::numeric_limits<float>::infinity());
                    logits.select(1, width - 1).fill_(std::numeric_limits<float>::infinity());
                }
                if (pattern == 5) {
                    logits.zero_();
                    logits.select(1, width - 1).fill_(1e-8f);
                }
                auto a = dense.forward(logits);
                auto b = compact.forward(logits);
                EXPECT_TRUE(torch::equal(a.token_ids, torch::argmax(logits, -1, true)));
                EXPECT_TRUE(torch::equal(a.token_ids, b.token_ids));
                if (compact.compactPointMassEnabled())
                    EXPECT_FALSE(b.all_probs.defined());
                else
                    EXPECT_TRUE(torch::equal(a.all_probs, b.all_probs));
            }
            for (bool sample : {false, true}) {
                constexpr int steps = 3;
                for (int reject : {0, 1, steps}) {
                    auto ids = torch::randint(width, {batch, steps}, i32);
                    auto dp  = torch::zeros({batch, steps, width}, f32);
                    dp.scatter_(2, ids.to(torch::kInt64).unsqueeze(2), 1.f);
                    auto tp     = torch::softmax(torch::randn({batch, steps + 1, width}, f32), -1);
                    auto target = torch::randint(width, {batch, steps + 1}, i32);
                    target.narrow(1, 0, reject).copy_(ids.narrow(1, 0, reject));
                    if (reject < steps)
                        target.select(1, reject).copy_((ids.select(1, reject) + 1).remainder(width));
                    auto uniform = torch::full({batch, steps + 1}, 0.5f, f32);
                    uniform.narrow(1, 0, reject).zero_();
                    auto do_sample = torch::full({batch}, sample, i32.dtype(torch::kBool));
                    auto a         = torch::full({batch, steps + 1}, -777, i32);
                    auto al        = torch::full({batch}, -777, i32);
                    auto b         = torch::empty_like(a);
                    auto bl        = torch::empty_like(al);
                    execRejectionSampling(
                        {dp, ids, uniform, tp, target.reshape({batch * (steps + 1), 1}), a, al, do_sample});
                    ASSERT_EQ(invokePointMassRejectionSampling(ids.data_ptr<int32_t>(),
                                                               uniform.data_ptr<float>(),
                                                               tp.data_ptr<float>(),
                                                               target.data_ptr<int32_t>(),
                                                               1,
                                                               b.data_ptr<int32_t>(),
                                                               bl.data_ptr<int32_t>(),
                                                               do_sample.data_ptr<bool>(),
                                                               batch,
                                                               steps,
                                                               width,
                                                               at::cuda::getCurrentCUDAStream().stream()),
                              cudaSuccess);
                    EXPECT_TRUE(torch::equal(a, b));
                    EXPECT_TRUE(torch::equal(al, bl));
                    EXPECT_FALSE(b.eq(-777).any().item<bool>());
                }
            }
        }
    }
    // Mapped vocabularies retain the established dense representation.
    spec::FastTopKSampler mapped(torch::arange(7, i32.dtype(torch::kInt64)), true);
    EXPECT_FALSE(mapped.compactPointMassEnabled());
    auto                  mapping = torch::arange(7, i32.dtype(torch::kInt64)).flip({0});
    spec::FastTopKSampler remapped(mapping, true);
    auto                  logits = torch::arange(7, f32).unsqueeze(0);
    EXPECT_EQ(remapped.forward(logits).token_ids.item<int64_t>(), 0);
    auto multiple = dense.forward(logits, 3);
    EXPECT_TRUE(torch::equal(multiple.all_probs, torch::softmax(logits, -1)));
    EXPECT_TRUE(torch::equal(multiple.token_ids, std::get<1>(torch::topk(multiple.all_probs, 3, -1))));
}

TEST_F(MtpExecutorTest, testCompactTargetPointMassRejection) {
    const auto    i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto    f32   = i32.dtype(torch::kFloat32);
    constexpr int batch = 5;
    constexpr int steps = 3;
    for (int width : {7, 4096}) {
        const int vocab = width * 2;
        auto      candidate_ids =
            (torch::arange(width, i32) * 2).reshape({1, 1, width}).expand({batch, steps + 1, width}).contiguous();
        candidate_ids.select(2, width - 1).fill_(-1);
        auto compact_q = torch::zeros({batch, steps + 1, width}, f32);
        compact_q.select(2, 0).fill_(0.25f);
        compact_q.select(2, 1).fill_(0.5f);
        compact_q.select(2, 2).fill_(0.25f);
        auto dense_q = torch::zeros({batch, steps + 1, vocab}, f32);
        dense_q.scatter_add_(2, candidate_ids.clamp_min(0).to(torch::kLong), compact_q);
        for (bool sample : {false, true}) {
            for (int reject : {0, 1, steps}) {
                for (bool missing : {false, true}) {
                    auto draft   = torch::full({batch, steps}, 2, i32);
                    auto target  = torch::full({batch, steps + 1}, 2, i32);
                    auto uniform = torch::full({batch, steps + 1}, 0.125f, f32);
                    if (reject < steps) {
                        uniform.select(1, reject).fill_(0.875f);
                        target.select(1, reject).fill_(4);
                        if (missing)
                            draft.select(1, reject).fill_(3);
                    }
                    auto do_sample       = torch::full({batch}, sample, i32.dtype(torch::kBool));
                    auto dense_ids       = torch::full({batch, steps + 1}, -777, i32);
                    auto compact_ids     = torch::empty_like(dense_ids);
                    auto dense_lengths   = torch::empty({batch}, i32);
                    auto compact_lengths = torch::empty_like(dense_lengths);
                    ASSERT_EQ(invokePointMassRejectionSampling(draft.data_ptr<int32_t>(),
                                                               uniform.data_ptr<float>(),
                                                               dense_q.data_ptr<float>(),
                                                               target.data_ptr<int32_t>(),
                                                               1,
                                                               dense_ids.data_ptr<int32_t>(),
                                                               dense_lengths.data_ptr<int32_t>(),
                                                               do_sample.data_ptr<bool>(),
                                                               batch,
                                                               steps,
                                                               vocab,
                                                               at::cuda::getCurrentCUDAStream().stream()),
                              cudaSuccess);
                    ASSERT_EQ(invokePointMassRejectionSampling(draft.data_ptr<int32_t>(),
                                                               uniform.data_ptr<float>(),
                                                               compact_q.data_ptr<float>(),
                                                               target.data_ptr<int32_t>(),
                                                               1,
                                                               compact_ids.data_ptr<int32_t>(),
                                                               compact_lengths.data_ptr<int32_t>(),
                                                               do_sample.data_ptr<bool>(),
                                                               batch,
                                                               steps,
                                                               width,
                                                               at::cuda::getCurrentCUDAStream().stream(),
                                                               candidate_ids.data_ptr<int32_t>()),
                              cudaSuccess);
                    EXPECT_TRUE(torch::equal(dense_ids, compact_ids));
                    EXPECT_TRUE(torch::equal(dense_lengths, compact_lengths));
                    EXPECT_TRUE(compact_lengths.eq(reject + 1).all().item<bool>());
                    EXPECT_FALSE(compact_ids.eq(-777).any().item<bool>());
                }
            }
        }
    }
}

TEST_F(MtpExecutorTest, testHy4SamplerConfig) {
    {
        autil::EnvGuard env("RTP_LLM_HY4_FORCE_TOPK", "0");
        ASSERT_EQ(::unsetenv("RTP_LLM_HY4_FORCE_TOPK"), 0);
        Sampler sampler(SamplerInitParams{"hy_v4"});
        EXPECT_EQ(sampler.force_topk_, 4096);
    }
    {
        autil::EnvGuard env("RTP_LLM_HY4_FORCE_TOPK", "0");
        Sampler         sampler(SamplerInitParams{"hy_v4"});
        EXPECT_EQ(sampler.force_topk_, 0);
    }
    {
        autil::EnvGuard env("RTP_LLM_HY4_FORCE_TOPK", "4096");
        Sampler         sampler(SamplerInitParams{"hy_v4"});
        Sampler         other(SamplerInitParams{"glm_5"});
        EXPECT_EQ(sampler.force_topk_, 4096);
        EXPECT_EQ(other.force_topk_, 0);
    }
    for (const std::string value : {"", "-1", "2147483648", "4096x"}) {
        autil::EnvGuard env("RTP_LLM_HY4_FORCE_TOPK", value);
        EXPECT_THROW(Sampler(SamplerInitParams{"hy_v4"}), c10::Error);
        EXPECT_NO_THROW(Sampler(SamplerInitParams{"glm_5"}));
    }
}

TEST_F(MtpExecutorTest, testDraftRawLogitsArgmax) {
    auto logits = torch::zeros({1, 8192}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    logits[0][8191].fill_(1e-8f);
    spec::FastTopKSampler draft(torch::Tensor(), false, true);
    auto                  result = draft.forward(logits);
    EXPECT_EQ(result.token_ids.item<int64_t>(), 8191);
    EXPECT_TRUE(torch::equal(result.all_probs, torch::zeros_like(logits).scatter_(1, result.token_ids, 1.f)));
}

TEST_F(MtpExecutorTest, testHy4CompactTargetSamplerForward) {
    constexpr int batch      = 4;
    constexpr int vocab      = 8192;
    constexpr int cap        = 4096;
    const auto    host_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    const auto    device_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (int mode = 0; mode < 6; ++mode) {
        SCOPED_TRACE(mode);
        autil::EnvGuard env("RTP_LLM_HY4_FORCE_TOPK", mode == 2 ? "0" : "4096");
        Sampler         sampler(SamplerInitParams{mode == 5 ? "glm_5" : "hy_v4"});
        SamplerInputs   inputs{};
        inputs.logits                     = torch::randn({batch, vocab}, device_f32);
        const auto expected_greedy        = torch::argmax(inputs.logits, -1);
        inputs.token_ids                  = torch::zeros({batch, 1}, host_i32);
        inputs.input_lengths              = torch::zeros({batch}, host_i32);
        inputs.sequence_lengths           = torch::zeros({batch}, host_i32);
        inputs.vocab_size                 = vocab;
        inputs.step                       = 0;
        inputs.batch_size                 = batch;
        inputs.batch_size_out             = batch;
        inputs.num_beams_in               = torch::ones({batch}, host_i32.dtype(torch::kInt64));
        inputs.num_beams_out              = torch::ones_like(inputs.num_beams_in);
        inputs.top_k                      = torch::zeros({batch}, host_i32);
        inputs.top_p                      = torch::full({batch}, 0.9f, host_i32.dtype(torch::kFloat32));
        inputs.temperature                = torch::ones({batch}, host_i32.dtype(torch::kFloat32));
        inputs.allow_compact_target_probs = true;
        inputs.skip_logits_probs_restore  = true;
        inputs.phase                      = LogitsProcessorPhase::MTP_VERIFY;
        if (mode == 1) {
            inputs.top_k.data_ptr<int32_t>()[0] = 1;
            inputs.top_k.data_ptr<int32_t>()[2] = 32;
            inputs.top_p.data_ptr<float>()[3]   = 0;
        }
        if (mode == 3)
            inputs.cum_log_probs = torch::zeros({batch}, host_i32.dtype(torch::kFloat32));
        if (mode == 4)
            inputs.top_k.fill_(1);
        for (int row = 0; row < batch; ++row) {
            inputs.generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            inputs.generator.back().set_current_seed(1234 + row);
        }
        auto       output  = sampler.forward(inputs);
        const bool compact = mode < 2;
        ASSERT_EQ(output.all_probs.dim(), 2);
        ASSERT_EQ(output.all_probs.size(0), batch);
        ASSERT_EQ(output.all_probs.size(1), compact ? cap : vocab);
        ASSERT_EQ(output.all_probs_indices.defined(), compact);
        EXPECT_TRUE(output.success.all().item<bool>());
        EXPECT_TRUE(torch::allclose(output.all_probs.sum(-1), torch::ones({batch}, device_f32), 1e-5, 1e-5));
        const auto tokens = output.token_ids.select(1, 0);
        if (compact) {
            ASSERT_EQ(output.all_probs_indices.scalar_type(), torch::kInt32);
            ASSERT_EQ(output.all_probs_indices.sizes(), output.all_probs.sizes());
            const auto matches = output.all_probs_indices.eq(tokens.unsqueeze(1));
            EXPECT_TRUE(matches.sum(-1).eq(1).all().item<bool>());
            EXPECT_TRUE((output.all_probs * matches).sum(-1).gt(0).all().item<bool>());
        } else {
            EXPECT_TRUE(output.all_probs.gather(1, tokens.to(torch::kLong).unsqueeze(1)).gt(0).all().item<bool>());
        }
        if (mode == 1) {
            EXPECT_EQ(tokens[0].item<int32_t>(), expected_greedy[0].item<int64_t>());
        }
        if (mode == 4) {
            EXPECT_TRUE(torch::equal(tokens.to(torch::kLong), expected_greedy));
        }
        if (mode == 3) {
            EXPECT_TRUE(torch::isfinite(output.cum_log_probs).all().item<bool>());
        }
    }
}

TEST_F(MtpExecutorTest, testDraftArgmaxGraphAndBenchmark) {
    const auto                 options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto                       stream  = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    cudaDeviceProp             device;
    ASSERT_EQ(cudaGetDeviceProperties(&device, stream.device_index()), cudaSuccess);
    std::cout << "HY4_DRAFT_DEVICE name=" << device.name << " sm=" << device.major << device.minor << std::endl;
    for (int batch : {1, 16, 128}) {
        auto          logits = torch::randn({batch, 154880}, options);
        torch::Tensor probabilities, ids, ids32, lengths;
        auto          run = [&](const torch::Tensor& scores) {
            ASSERT_TRUE(
                tryMtpGreedyPack(scores, {}, probabilities, ids, ids32, {}, {}, false, lengths, stream.stream(), true));
        };
        run(logits);
        ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        run(logits);
        graph.capture_end();
        for (int replay = 0; replay < 3; ++replay) {
            logits.normal_();
            logits.select(1, replay * 2048 + 17).fill_(100.0f);
            graph.replay();
            EXPECT_TRUE(torch::equal(ids, torch::argmax(logits, -1, true)));
            EXPECT_TRUE(torch::equal(ids32, ids.to(torch::kInt32)));
        }
        graph.reset();
        logits.normal_();
        auto original_ids = torch::argmax(torch::softmax(logits, -1), -1, true);
        run(logits);
        EXPECT_TRUE(torch::equal(ids, original_ids));
        std::cout << "HY4_DRAFT_ACCURACY batch=" << batch
                  << " random_mismatches=" << ids.ne(original_ids).sum().item<int64_t>() << std::endl;
        cudaEvent_t start, stop;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
        for (int trial = 0; trial < 3; ++trial) {
            for (int side = 0; side < 2; ++side) {
                const bool softmax = (trial + side) % 2 == 0;
                for (int i = 0; i < 20; ++i)
                    run(softmax ? torch::softmax(logits, -1) : logits);
                ASSERT_EQ(cudaEventRecord(start, stream.stream()), cudaSuccess);
                for (int i = 0; i < 100; ++i)
                    run(softmax ? torch::softmax(logits, -1) : logits);
                ASSERT_EQ(cudaEventRecord(stop, stream.stream()), cudaSuccess);
                ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
                float ms = 0;
                ASSERT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
                std::cout << "HY4_DRAFT_ARGMAX batch=" << batch << " vocab=154880 softmax=" << softmax
                          << " trial=" << trial << " us=" << ms * 1000.0f / 100 << std::endl;
            }
        }
        ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
    }
}

TEST_F(MtpExecutorTest, testCompactPointMassPreservesSamplerRngState) {
    const auto i32                   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto f32                   = i32.dtype(torch::kFloat32);
    auto       global_generator      = at::cuda::detail::getDefaultCUDAGenerator();
    auto       original_global_state = global_generator.get_state();
    for (int width : {7, 2048, 154880}) {
        ModelConfig model_config;
        model_config.max_seq_len = 64;
        model_config.vocab_size  = width;
        model_config.num_layers  = 1;
        RuntimeConfig                runtime_config;
        ResourceContext              resource_context;
        std::list<GenerateStreamPtr> streams;
        for (int row = 0; row < 4; ++row) {
            auto query                    = std::make_shared<GenerateInput>();
            query->input_ids              = torch::tensor({1}, torch::kInt32);
            query->generate_config        = std::make_shared<GenerateConfig>();
            query->generate_config->top_k = row % 2 == 0 ? 1 : 0;
            if (row >= 2)
                query->generate_config->random_seed = 1234 + row;
            streams.push_back(
                std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr));
        }
        for (int steps : {1, 3, 7}) {
            spec::SpeculativeSampler sampler(torch::Tensor(), steps);
            SamplerOutput            dense, compact, target;
            dense.token_ids = torch::randint(width, {4, steps}, i32);
            dense.all_probs = torch::zeros({4, steps, width}, f32);
            dense.all_probs.scatter_(2, dense.token_ids.to(torch::kInt64).unsqueeze(2), 1.f);
            compact.token_ids        = dense.token_ids;
            compact.draft_point_mass = true;
            target.token_ids         = torch::randint(width, {4 * (steps + 1), 1}, i32);
            target.all_probs         = torch::softmax(torch::randn({4, steps + 1, width}, f32), -1);
            for (bool force_accept : {false, true}) {
                streams.back()->generateConfig()->force_sp_accept = force_accept;
                auto                       before_global          = global_generator.get_state();
                std::vector<torch::Tensor> before_seeded;
                for (auto& stream : streams) {
                    auto generator = stream->getGenerator();
                    if (generator.defined())
                        before_seeded.push_back(generator.get_state());
                }
                auto                       a            = sampler.forward(streams, dense, target);
                auto                       after_global = global_generator.get_state();
                std::vector<torch::Tensor> after_seeded;
                size_t                     idx = 0;
                for (auto& stream : streams) {
                    auto generator = stream->getGenerator();
                    if (generator.defined()) {
                        after_seeded.push_back(generator.get_state());
                        generator.set_state(before_seeded[idx++]);
                    }
                }
                global_generator.set_state(before_global);
                auto b = sampler.forward(streams, compact, target);
                EXPECT_TRUE(torch::equal(a.accept_tokens, b.accept_tokens));
                EXPECT_TRUE(torch::equal(a.accept_len, b.accept_len));
                EXPECT_TRUE(torch::equal(after_global, global_generator.get_state()));
                idx = 0;
                for (auto& stream : streams) {
                    auto generator = stream->getGenerator();
                    if (generator.defined()) {
                        EXPECT_TRUE(torch::equal(after_seeded[idx++], generator.get_state()));
                    }
                }
            }
        }
    }
    global_generator.set_state(original_global_state);
}

TEST_F(MtpExecutorTest, testMtpGreedyFinalizeBatchAndStride) {
    const auto cpu    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    const auto gpu    = cpu.device(torch::kCUDA);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (int batch : {1, 4, 7, 16, 32, 67, 257}) {
        for (int stride : {1, 3}) {
            auto  draft_cpu   = torch::empty({batch, 3}, cpu);
            auto  target_cpu  = torch::full({batch * 4, stride}, -9, cpu);
            auto  expected    = torch::zeros({batch, 4}, cpu);
            auto  lengths_cpu = torch::empty({batch}, cpu);
            auto* d           = draft_cpu.data_ptr<int32_t>();
            auto* t           = target_cpu.data_ptr<int32_t>();
            auto* e           = expected.data_ptr<int32_t>();
            for (int b = 0; b < batch; ++b) {
                const int accepted                 = b % 4;
                lengths_cpu.data_ptr<int32_t>()[b] = accepted + 1;
                for (int j = 0; j < 4; ++j) {
                    const int token                      = j < accepted ? b * 4 + j : b * 4 + j + 1;
                    t[(b * 4 + j) * stride + stride - 1] = token;
                    if (j < 3)
                        d[b * 3 + j] = b * 4 + j;
                    if (j <= accepted)
                        e[b * 4 + j] = token;
                }
            }
            auto          draft   = draft_cpu.to(torch::kCUDA);
            auto          target  = target_cpu.to(torch::kCUDA);
            auto          tokens  = torch::full({batch, 4}, -999, gpu);
            auto          lengths = torch::empty({batch}, gpu);
            torch::Tensor rows, total;
            ASSERT_TRUE(tryMtpGreedyFinalize(draft, target, tokens, lengths, rows, total, stream));
            EXPECT_TRUE(torch::equal(tokens.cpu(), expected));
            EXPECT_TRUE(torch::equal(lengths.cpu(), lengths_cpu));
            EXPECT_TRUE(torch::equal(rows.cpu(), lengths_cpu.to(torch::kInt64)));
            EXPECT_EQ(total.item<int64_t>(), lengths_cpu.sum().item<int64_t>());
            // Unsupported layouts must return without modifying caller outputs.
            auto invalid = draft.to(torch::kInt64);
            tokens.fill_(-99);
            EXPECT_FALSE(tryMtpGreedyFinalize(invalid, target, tokens, lengths, rows, total, stream));
            EXPECT_TRUE(torch::equal(tokens, torch::full_like(tokens, -99)));
        }
    }
}

TEST_F(MtpExecutorTest, testMtpGraphCopyFillOverlapAndFallback) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto stream  = at::cuda::getCurrentCUDAStream().stream();
    for (int batch : {1, 4, 7, 16, 32, 67}) {
        auto src      = torch::arange(batch, options);
        auto dst      = torch::full({batch + 9}, -99, options);
        auto expected = torch::full_like(dst, 17);
        expected.narrow(0, 2, batch).copy_(src);
        FusedD2DCopyParams copies;
        copies.add(src.data_ptr(), dst.data_ptr<int32_t>() + 2, batch * sizeof(int32_t));
        CudaGraphPrepareFillParams fills;
        fills.region_count = 1;
        fills.regions[0]   = {dst.data_ptr<int32_t>(), dst.numel(), 17};
        ASSERT_TRUE(tryInvokeMtpGraphCopyFill(copies, fills, stream));
        EXPECT_TRUE(torch::equal(dst, expected));
        dst.fill_(-99);
        // A source inside the fill region needs the original fill-before-copy order.
        copies.src[0] = dst.data_ptr();
        EXPECT_FALSE(tryInvokeMtpGraphCopyFill(copies, fills, stream));
        EXPECT_TRUE(torch::equal(dst, torch::full_like(dst, -99)));
        copies.src[0]  = src.data_ptr();
        copies.size[0] = 1;  // Partial-word overlap is deliberately unsupported.
        EXPECT_FALSE(tryInvokeMtpGraphCopyFill(copies, fills, stream));
        EXPECT_TRUE(torch::equal(dst, torch::full_like(dst, -99)));
    }
}

TEST_F(MtpExecutorTest, testMtpBatchRowsPrepareAndPublish) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto stream  = at::cuda::getCurrentCUDAStream().stream();
    for (int batch : {1, 4, 7, 16, 32, 67, 129, 257}) {
        // Two prior batch owners, reversed/subselected rows, duplicate reads.
        auto                         first  = torch::arange(batch * 2, options) + 10;
        auto                         second = torch::arange(batch * 2, options) + 1000;
        auto                         tokens = torch::arange(batch * 6, options).reshape({batch * 2, 3});
        std::vector<MtpNextInputRow> rows;
        std::vector<const int32_t*>  lengths;
        auto                         expected_lengths = torch::empty({batch}, options.device(torch::kCPU));
        auto                         expected_tokens  = torch::empty_like(expected_lengths);
        for (int i = 0; i < batch; ++i) {
            const int   row    = (batch * 2 - 1 - i / 2);
            const auto& source = i % 2 ? first : second;
            rows.push_back({tokens.data_ptr<int32_t>() + row * 3 + 2, source.data_ptr<int32_t>() + row});
            lengths.push_back(source.data_ptr<int32_t>() + row);
            expected_lengths.data_ptr<int32_t>()[i] = row + (i % 2 ? 10 : 1000);
            expected_tokens.data_ptr<int32_t>()[i]  = row * 3 + 2;
        }
        torch::Tensor packed;
        ASSERT_TRUE(tryMtpPrepareNextInputs(rows, options, packed, stream));
        EXPECT_TRUE(torch::equal(packed.select(0, 0).cpu(), expected_tokens));
        EXPECT_TRUE(torch::equal(packed.select(0, 1).cpu(), expected_lengths));
        EXPECT_TRUE(torch::equal(packed.select(0, 2).cpu(), expected_lengths - 1));
        EXPECT_TRUE(torch::equal(packed.select(0, 3), torch::arange(batch, options)));
        auto          accepted      = torch::arange(batch, options).remainder(4) + 1;
        auto          output_tokens = torch::arange(batch * 4, options).reshape({batch, 4});
        auto          hidden        = torch::randn({batch * 4, 17}, options.dtype(torch::kBFloat16));
        torch::Tensor next_a, selected_a, hidden_a, next_b, selected_b, hidden_b;
        auto          contiguous_previous = expected_lengths.to(torch::kCUDA);
        ASSERT_TRUE(tryMtpPublishState(
            accepted, output_tokens, contiguous_previous, hidden, 4, next_a, selected_a, hidden_a, stream));
        ASSERT_TRUE(tryMtpPublishState(
            accepted, output_tokens, torch::Tensor(), hidden, 4, next_b, selected_b, hidden_b, stream, lengths));
        EXPECT_TRUE(torch::equal(next_a, next_b));
        EXPECT_TRUE(torch::equal(selected_a, selected_b));
        EXPECT_TRUE(torch::equal(hidden_a, hidden_b));
        // Invalid descriptor must return before replacing or writing outputs.
        auto previous_output = packed;
        rows.back().token    = nullptr;
        EXPECT_FALSE(tryMtpPrepareNextInputs(rows, options, packed, stream));
        EXPECT_TRUE(packed.is_same(previous_output));
    }
}

TEST_F(MtpExecutorTest, testMtpIntegerCleanupMetricsAndDecodeMetadata) {
    const auto i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       stream = at::cuda::getCurrentCUDAStream().stream();
    for (int batch : {0, 1, 7, 257}) {
        auto          lengths = torch::arange(batch, i32) - 3;
        torch::Tensor rows, sum, cu, kv, dc, plus;
        ASSERT_TRUE(tryMtpAcceptMetrics(lengths, rows, sum, stream));
        EXPECT_TRUE(torch::equal(rows, lengths.to(torch::kInt64)));
        EXPECT_TRUE(torch::equal(sum, lengths.to(torch::kInt64).sum().reshape({1})));
        ASSERT_TRUE(tryMtpDecodeAttentionMetadata(lengths, cu, kv, dc, plus, stream));
        EXPECT_TRUE(torch::equal(cu, torch::zeros({batch + 1}, i32)));
        EXPECT_TRUE(torch::equal(kv, torch::zeros_like(cu)));
        EXPECT_TRUE(torch::equal(dc, torch::arange(batch + 1, i32)));
        EXPECT_TRUE(torch::equal(plus, lengths + 1));
        auto tokens    = torch::arange(batch * 4, i32).reshape({batch, 4}) - 2;
        auto reference = tokens.clone();
        reference.index_put_({reference == -1}, 0);
        ASSERT_TRUE(tryMtpCleanRejectedTokens(tokens, stream));
        EXPECT_TRUE(torch::equal(tokens, reference));
    }
}
#endif

TEST_F(MtpExecutorTest, testForwardOverlapConsumesGrpcPayloadBeforeFirstForwardAndFallback) {
    MtpExecutorTestConfig config;
    config.gen_num_per_cycle = 1;
    auto  components         = createMtpExecutorComponents(config);
    auto& executor           = *components.executor;
    // Exercise the real decodeStep gate with a probe model instead of weights.
    executor.forward_overlap_supported_ = true;
    executor.role_type_                 = RoleType::DECODE;
    // The legacy mock cache omits the explicit FULL group metadata.
    executor.batch_stream_processor_->model_input_gatherer_config_.kv_cache_group_types = {CacheGroupType::FULL};
    if (!executor.useForwardOverlap()) {
        GTEST_SKIP() << "Requires RTP_LLM_MTP_FORWARD_OVERLAP=1, RTP_LLM_STREAM_ASYNC=1, "
                        "RTP_LLM_DEVICE_INPUT=1 and RTP_LLM_MTP_ASYNC_PREPARE=0";
    }
    auto context          = components.resource_context;
    context.cache_manager = executor.cache_manager_;
    auto stream           = createContextStream(components.model_config, components.runtime_config, context, {1, 2, 3});
    stream->setReserveStep(4);
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->setIsContextStream(false);
    stream->setScoreLen(2);
    stream->setReturnAllProbs(true);
    stream->generate_status_->status = StreamState::RUNNING;

    const auto pinned_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    const auto cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       sp         = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp->tokens            = torch::tensor({1, 2}, torch::kInt32).reshape({1, 2});
    sp->tensors_holder    = {torch::full({1, 4}, 0.25, pinned_f32), torch::full({1, 2}, 10.0, pinned_f32)};
    sp->all_probs         = sp->tensors_holder[0].to(torch::kCUDA);
    sp->hidden_states     = sp->tensors_holder[1].to(torch::kCUDA);
    stream->setSPOutputBuffer(sp);
    GenerateStream::MtpAsyncDeviceState state;
    state.accept_len_gpu         = torch::ones({1}, cuda_i32);
    state.accept_tokens_gpu      = torch::tensor({1, 0}, torch::kInt32).reshape({1, 2}).to(torch::kCUDA);
    state.next_seq_len_gpu       = torch::full({1}, stream->seqLength(), cuda_i32);
    state.propose_tokens_gpu     = torch::full({1}, 2, cuda_i32);
    state.last_hidden_states_gpu = sp->hidden_states;
    state.draft_all_probs_gpu    = sp->all_probs;
    state.next_real_seq_len      = stream->seqLength();
    stream->setMtpAsyncDeviceState(state);
    ASSERT_TRUE(stream->tryAcquireMtpForwardClaim());
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    ASSERT_TRUE(
        executor.batch_stream_processor_->tryGatherDecodeForwardInput({stream}, executor.buffer_holder_).has_value());

    struct ForwardReached {};
    struct ProbeModel final: ModelBase {
        std::function<void()> probe;
        GptModelOutputs       forward(const GptModelInputs&) override {
            probe();
            throw ForwardReached{};  // Stop before sampling; no fabricated model outputs.
        }
    };
    int  forwards        = 0;
    auto expected_probs  = state.draft_all_probs_gpu;
    auto expected_hidden = state.last_hidden_states_gpu;
    auto model           = std::make_unique<ProbeModel>();
    model->probe         = [&] {
        ++forwards;
        EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), forwards == 1);  // Early forward owns a KV claim.
        EXPECT_TRUE(sp->tensors_holder.empty());
        EXPECT_TRUE(torch::equal(stream->getMtpAsyncDeviceState().draft_all_probs_gpu, expected_probs));
        EXPECT_TRUE(torch::equal(stream->getMtpAsyncDeviceState().last_hidden_states_gpu, expected_hidden));
    };
    executor.setTargetModel(std::move(model));
    executor.setDraftModel(std::move(components.fake_draft_model));
    MtpMetricsCollector metrics;
    EXPECT_THROW(executor.decodeStep({stream}, metrics), ForwardReached);
    ASSERT_EQ(forwards, 1);
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());

    // A later ordinary/fallback round must not restore the initial RPC payload
    // over the new GPU state produced by decode bookkeeping.
    state                        = stream->getMtpAsyncDeviceState();
    expected_probs               = torch::full_like(state.draft_all_probs_gpu, 0.5);
    expected_hidden              = torch::full_like(state.last_hidden_states_gpu, 20.0);
    state.draft_all_probs_gpu    = expected_probs;
    state.last_hidden_states_gpu = expected_hidden;
    stream->setMtpAsyncDeviceState(std::move(state));
    sp->all_probs                       = expected_probs;
    sp->hidden_states                   = expected_hidden;
    executor.forward_overlap_supported_ = false;
    EXPECT_THROW(executor.decodeStep({stream}, metrics), ForwardReached);
    EXPECT_EQ(forwards, 2);
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
}

TEST_F(MtpExecutorTest, testForwardOverlapMultiStepFusionMatchesBarrierPath) {
    if (!autil::EnvUtil::getEnv("RTP_LLM_MTP_FORWARD_OVERLAP", false)
        || !autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false)
        || !autil::EnvUtil::getEnv("RTP_LLM_DEVICE_INPUT", false)
        || autil::EnvUtil::getEnv("RTP_LLM_MTP_ASYNC_PREPARE", false)) {
        GTEST_SKIP() << "Run the dedicated overlap test targets";
    }
    struct TokenModel final: ModelBase {
        torch::ScalarType     hidden_dtype;
        std::function<void()> before_forward;
        explicit TokenModel(torch::ScalarType dtype): hidden_dtype(dtype) {}
        GptModelOutputs forward(const GptModelInputs& input) override {
            if (before_forward) {
                before_forward();
            }
            auto            tokens   = input.combo_tokens.to(torch::kLong);
            auto            selected = tokens.index_select(0, input.lm_output_indexes.to(torch::kLong));
            GptModelOutputs out;
            out.logits = torch::full({selected.numel(), 4}, -20.0, tokens.options().dtype(torch::kFloat32));
            out.logits.scatter_(1, (selected + 1).remainder(4).reshape({-1, 1}), 20.0);
            // BF16 exercises publish fusion; FP32 exercises its fallback.
            auto hidden           = tokens.to(hidden_dtype);
            out.all_hidden_states = torch::stack({hidden, hidden + 0.25}, 1);
            return out;
        }
    };
    for (const int steps : {2, 3}) {
        for (const auto dtype : {torch::kBFloat16, torch::kFloat32}) {
            SCOPED_TRACE(std::to_string(steps) + "/" + std::to_string(static_cast<int>(dtype)));
            auto run = [&](bool overlap, std::vector<torch::Tensor>& results) {
                MtpExecutorTestConfig config;
                config.gen_num_per_cycle            = steps;
                config.vocab_size_override          = 4;
                config.cache_block_num              = 64;
                auto  components                    = createMtpExecutorComponents(config);
                auto& executor                      = *components.executor;
                executor.forward_overlap_supported_ = overlap;
                executor.role_type_                 = RoleType::DECODE;
                executor.batch_stream_processor_->model_input_gatherer_config_.kv_cache_group_types = {
                    CacheGroupType::FULL};
                ASSERT_EQ(executor.useForwardOverlap(), overlap);
                executor.setDraftModel(std::make_unique<TokenModel>(dtype));
                auto context          = components.resource_context;
                context.cache_manager = executor.cache_manager_;
                context.role_type     = RoleType::DECODE;
                auto options          = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
                std::vector<GenerateStreamPtr> streams;
                for (int row = 0; row < 2; ++row) {
                    auto input             = std::make_shared<GenerateInput>();
                    input->input_ids       = torch::tensor({row + 1}, torch::kInt32);
                    input->generate_config = std::make_shared<GenerateConfig>();
                    auto& gc               = *input->generate_config;
                    gc.top_k               = 1;
                    gc.do_sample           = false;
                    gc.max_new_tokens      = 48;
                    gc.force_sp_accept     = false;
                    gc.ignore_eos          = true;
                    gc.return_logprobs     = true;
                    gc.top_logprobs        = 1;
                    auto stream            = std::make_shared<NormalGenerateStream>(
                        input, components.model_config, components.runtime_config, context, nullptr);
                    stream->setReserveStep(24);
                    ASSERT_TRUE(stream->initKVBlock().ok());
                    stream->setIsContextStream(false);
                    stream->setScoreLen(steps + 1);
                    stream->setReturnAllProbs(true);
                    stream->generate_status_->status = StreamState::RUNNING;
                    auto sp                          = std::make_shared<SpeculativeExecutorStreamOutput>();
                    // First request accepts its draft; second rejects its seed.
                    const int draft         = row == 0 ? 2 : 0;
                    sp->tokens              = torch::tensor({row + 1, draft}, torch::kInt32).reshape({1, 2});
                    sp->target_token_gpu    = torch::full({1}, row + 1, options);
                    sp->propose_tokens_gpu  = torch::full({1}, draft, options);
                    sp->all_probs           = torch::zeros({1, 4}, options.dtype(torch::kFloat32));
                    sp->all_probs[0][draft] = 1.0;
                    sp->hidden_states       = torch::zeros({1, 2}, options.dtype(dtype));
                    stream->setSPOutputBuffer(sp);
                    GenerateStream::MtpAsyncDeviceState state;
                    state.accept_len_gpu          = torch::ones({1}, options);
                    state.accept_tokens_gpu       = torch::zeros({1, steps + 1}, options);
                    state.accept_tokens_gpu[0][0] = row + 1;
                    state.propose_tokens_gpu      = sp->propose_tokens_gpu;
                    state.next_seq_len_gpu        = torch::ones({1}, options);
                    state.draft_all_probs_gpu     = sp->all_probs;
                    state.last_hidden_states_gpu  = sp->hidden_states;
                    state.last_real_seq_len = state.next_real_seq_len = 1;
                    stream->setMtpAsyncDeviceState(std::move(state));
                    streams.push_back(stream);
                }
                MtpMetricsCollector          metrics;
                std::list<GenerateStreamPtr> active;
                int                          target_forwards = 0;
                auto                         target          = std::make_unique<TokenModel>(dtype);
                target->before_forward                       = [&]() {
                    ++target_forwards;
                    for (const auto& stream : active) {
                        // A claim increments pending bookkeeping before forward;
                        // the barrier path has drained it. Detect silent fallback.
                        EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), overlap);
                    }
                };
                executor.setTargetModel(std::move(target));
                // No worker drain between rounds. Reorder published batch rows,
                // then shrink while the removed request still has bookkeeping.
                active = {streams[0], streams[1]};
                ASSERT_TRUE(executor.decodeStep(active, metrics).ok());
                active = {streams[1], streams[0]};
                ASSERT_TRUE(executor.decodeStep(active, metrics).ok());
                active = {streams[0]};
                ASSERT_TRUE(executor.decodeStep(active, metrics).ok());
                executor.waitPreviousBookkeepingAndKvSwaps({streams[0], streams[1]});
                EXPECT_EQ(target_forwards, 3);
                int result_row = 0;
                for (const auto& stream : streams) {
                    EXPECT_FALSE(stream->hasError());
                    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
                    const auto& state = stream->getMtpAsyncDeviceState();
                    EXPECT_EQ(state.draft_all_probs_gpu.defined(), !mtpFusionEnabled());
                    const int        expected_length = result_row == 0 ? 1 + 3 * (steps + 1) : 2 + steps + 1;
                    std::vector<int> expected_tokens;
                    for (int i = 0; i < expected_length; ++i) {
                        expected_tokens.push_back((result_row + 1 + i) % 4);
                    }
                    EXPECT_EQ(stream->getCompleteTokenIds()->completeTokenIdsVec(0), expected_tokens);
                    ++result_row;
                    if (mtpFusionEnabled() && dtype == torch::kBFloat16) {
                        ASSERT_TRUE(state.batch);
                        // Defined only when tryMtpPublishState actually succeeds.
                        ASSERT_TRUE(state.batch->selected_tokens.defined());
                    }
                    results.push_back(
                        torch::tensor(stream->getCompleteTokenIds()->completeTokenIdsVec(0), torch::kInt32));
                    for (const auto& value : {state.accept_len_gpu,
                                              state.accept_tokens_gpu,
                                              state.next_seq_len_gpu,
                                              state.propose_tokens_gpu,
                                              state.last_hidden_states_gpu,
                                              stream->getSPOutputBuffer()->target_token_gpu}) {
                        ASSERT_TRUE(value.defined());
                        results.push_back(value.cpu().clone());
                    }
                    results.push_back(state.draft_all_probs_gpu.defined() ? state.draft_all_probs_gpu.cpu().clone() :
                                                                            torch::Tensor());
                    results.push_back(
                        stream->token_logprobs_.narrow(1, 0, stream->logprobs_history_size_).cpu().clone());
                    results.push_back(
                        stream->top_logprob_token_ids_.narrow(1, 0, stream->logprobs_history_size_).cpu().clone());
                    results.push_back(stream->top_logprobs_.narrow(1, 0, stream->logprobs_history_size_).cpu().clone());
                    stream->releaseResource();
                }
            };
            std::vector<torch::Tensor> barrier, overlap;
            run(false, barrier);
            ASSERT_FALSE(HasFatalFailure());
            run(true, overlap);
            ASSERT_FALSE(HasFatalFailure());
            ASSERT_EQ(barrier.size(), overlap.size());
            for (size_t i = 0; i < barrier.size(); ++i) {
                ASSERT_EQ(barrier[i].defined(), overlap[i].defined()) << "result " << i;
                if (barrier[i].defined()) {
                    EXPECT_TRUE(torch::equal(barrier[i], overlap[i])) << "result " << i;
                }
            }
        }
    }
}

TEST_F(MtpExecutorTest, testForwardOverlapDynamicRoundsMatchBarrierPath) {
    if (!autil::EnvUtil::getEnv("RTP_LLM_MTP_FORWARD_OVERLAP", false)
        || !autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false)
        || !autil::EnvUtil::getEnv("RTP_LLM_DEVICE_INPUT", false)
        || autil::EnvUtil::getEnv("RTP_LLM_MTP_ASYNC_PREPARE", false)) {
        GTEST_SKIP() << "Requires RTP_LLM_MTP_FORWARD_OVERLAP=1, RTP_LLM_STREAM_ASYNC=1, "
                        "RTP_LLM_DEVICE_INPUT=1 and RTP_LLM_MTP_ASYNC_PREPARE=0";
    }
    // Replace only the model: exercise the real target/draft samplers,
    // rejection sampling, dispatch worker and published state across rounds.
    // This is execution/state coverage, not a real-model accuracy or graph test.
    struct TokenSuccessorModel final: ModelBase {
        std::function<void(const GptModelInputs&)> before_forward;
        GptModelOutputs                            forward(const GptModelInputs& input) override {
            if (before_forward) {
                before_forward(input);
            }
            auto            tokens = input.combo_tokens.to(torch::kLong);
            auto            selected = tokens.index_select(0, input.lm_output_indexes.to(torch::kLong));
            auto            next = (selected + 1).remainder(4).reshape({-1, 1});
            GptModelOutputs output;
            output.logits = torch::full({selected.numel(), 4}, -20.0, tokens.options().dtype(torch::kFloat32));
            output.logits.scatter_(1, next, 20.0);
            auto hidden              = tokens.to(torch::kFloat32);
            output.all_hidden_states = torch::stack({hidden, hidden + 0.25}, 1);
            return output;
        }
    };
    struct Result {
        std::vector<std::vector<int>> tokens;
        std::vector<torch::Tensor>    target_inputs;
        std::vector<torch::Tensor>    states;
        torch::Tensor                 logprobs;
        torch::Tensor                 top_ids;
        torch::Tensor                 top_logprobs;
        int                           grammar_calls = 0;
    };
    struct CountingSpecProcessor final: RejectDraftTokenSpecProcessor {
        CountingSpecProcessor(): RejectDraftTokenSpecProcessor(3, 0) {}
        int calls = 0;
        int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
            ++calls;
            return RejectDraftTokenSpecProcessor::tryAcceptAndFillBitmask(request);
        }
    };

    auto run = [&](bool overlap, Result& result) {
        MtpExecutorTestConfig config;
        config.gen_num_per_cycle   = 1;
        config.vocab_size_override = 4;
        auto  components           = createMtpExecutorComponents(config);
        auto& executor             = *components.executor;
        // The tiny C++ model has no Python graph capability declaration.
        // Override only that capability; all runtime overlap gates stay real.
        executor.forward_overlap_supported_                                                 = overlap;
        executor.role_type_                                                                 = RoleType::DECODE;
        executor.batch_stream_processor_->model_input_gatherer_config_.kv_cache_group_types = {CacheGroupType::FULL};
        ASSERT_EQ(executor.useForwardOverlap(), overlap)
            << "Requires RTP_LLM_MTP_FORWARD_OVERLAP=1, RTP_LLM_STREAM_ASYNC=1, "
               "RTP_LLM_DEVICE_INPUT=1 and RTP_LLM_MTP_ASYNC_PREPARE=0";
        ASSERT_TRUE(executor.useStreamAsync());
        const auto cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto       context    = components.resource_context;
        context.cache_manager = executor.cache_manager_;
        context.role_type     = RoleType::DECODE;

        auto make_stream = [&](int token, int draft_token, int budget, int reserve, bool logprobs) {
            auto input             = std::make_shared<GenerateInput>();
            input->input_ids       = torch::tensor({token}, torch::kInt32);
            input->generate_config = std::make_shared<GenerateConfig>();
            auto& gc               = *input->generate_config;
            gc.top_k               = 1;
            gc.do_sample           = false;
            gc.max_new_tokens      = budget;
            gc.force_sp_accept     = false;
            gc.ignore_eos          = true;  // Token 0 is ordinary test data; finish only at the budget.
            gc.return_logprobs     = logprobs;
            gc.top_logprobs        = logprobs ? 1 : 0;
            auto stream            = std::make_shared<NormalGenerateStream>(
                input, components.model_config, components.runtime_config, context, nullptr);
            stream->setReserveStep(reserve);
            auto status = stream->initKVBlock();
            if (!status.ok()) {
                throw std::runtime_error(status.ToString());
            }
            stream->setIsContextStream(false);
            stream->setScoreLen(2);
            stream->setReturnAllProbs(true);
            stream->generate_status_->status = StreamState::RUNNING;
            auto sp                          = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp->tokens                       = torch::tensor({token, draft_token}, torch::kInt32).reshape({1, 2});
            sp->target_token_gpu             = torch::full({1}, token, cuda_i32);
            sp->propose_tokens_gpu           = torch::full({1}, draft_token, cuda_i32);
            sp->all_probs                    = torch::zeros({1, 4}, cuda_i32.dtype(torch::kFloat32));
            sp->all_probs[0][draft_token]    = 1.0;
            sp->hidden_states                = torch::zeros({1, 2}, cuda_i32.dtype(torch::kFloat32));
            stream->setSPOutputBuffer(sp);
            GenerateStream::MtpAsyncDeviceState state;
            state.accept_len_gpu         = torch::ones({1}, cuda_i32);
            state.accept_tokens_gpu      = torch::tensor({token, 0}, torch::kInt32).reshape({1, 2}).to(torch::kCUDA);
            state.propose_tokens_gpu     = sp->propose_tokens_gpu;
            state.next_seq_len_gpu       = torch::full({1}, stream->seqLength(), cuda_i32);
            state.draft_all_probs_gpu    = sp->all_probs;
            state.last_hidden_states_gpu = sp->hidden_states;
            state.last_real_seq_len      = stream->seqLength();
            state.next_real_seq_len      = stream->seqLength();
            stream->setMtpAsyncDeviceState(std::move(state));
            return stream;
        };

        // Eight initial blocks fit the fixture's nine usable two-token blocks.
        // B's incorrect initial draft must be rejected by the real sampler.
        auto a       = make_stream(1, 2, 6, 5, true);
        auto b       = make_stream(2, 0, 3, 5, false);
        auto c       = make_stream(3, 0, 6, 3, false);
        auto grammar = std::make_shared<CountingSpecProcessor>();
        c->logits_processor_list_.push_back(grammar);
        std::list<GenerateStreamPtr> active;
        int                          forwards = 0;
        auto                         target   = std::make_unique<TokenSuccessorModel>();
        target->before_forward                = [&](const GptModelInputs& input) {
            ++forwards;
            for (const auto& stream : active) {
                // A successful fast forward holds a claim; the barrier path
                // has drained the previous dispatch before invoking the model.
                EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), overlap);
            }
            for (const auto& tensor :
                 {input.combo_tokens, input.input_lengths, input.prefix_lengths, input.lm_output_indexes}) {
                result.target_inputs.push_back(tensor.clone());
            }
        };
        executor.setTargetModel(std::move(target));
        executor.setDraftModel(std::make_unique<TokenSuccessorModel>());
        MtpMetricsCollector metrics;
        active = {a, b};
        ASSERT_TRUE(executor.decodeStep(active, metrics).ok());
        // Do not drain the worker between these rounds. C is a fresh request
        // and A/B move to different rows while consuming round one's GPU state.
        active = {b, c, a};
        ASSERT_TRUE(executor.decodeStep(active, metrics).ok());
        executor.waitPreviousBookkeepingAndKvSwaps(active);
        EXPECT_TRUE(b->generate_status_->hasEvent(StreamEvents::GenerateDone));
        EXPECT_EQ(b->getCompleteTokenIds()->completeTokenIdsVec(0), std::vector<int>({2, 3, 0, 1}));
        b->releaseResource();
        EXPECT_TRUE(b->stream_cache_resource_->isResourceReleased());
        // Model the scheduler growing live requests after B releases its blocks.
        for (const auto& stream : {a, c}) {
            stream->setReserveStep(2);
            ASSERT_TRUE(stream->incrKVBlock().ok());
        }
        active = {c, a};
        ASSERT_TRUE(executor.decodeStep(active, metrics).ok());
        executor.waitPreviousBookkeepingAndKvSwaps(active);
        ASSERT_EQ(forwards, 3);
        EXPECT_TRUE(a->generate_status_->hasEvent(StreamEvents::GenerateDone));
        EXPECT_EQ(a->getCompleteTokenIds()->completeTokenIdsVec(0), std::vector<int>({1, 2, 3, 0, 1, 2, 3}));
        EXPECT_EQ(c->getCompleteTokenIds()->completeTokenIdsVec(0), std::vector<int>({3, 0, 1, 2, 3}));
        EXPECT_GT(grammar->calls, 0);
        EXPECT_EQ(grammar->acceptedTokenLen(), c->outputTokenLen());
        result.grammar_calls = grammar->calls;
        for (const auto& stream : {a, b, c}) {
            EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
            EXPECT_FALSE(stream->hasError());
            EXPECT_FALSE(stream->forceSpAccept());
            result.tokens.push_back(stream->getCompleteTokenIds()->completeTokenIdsVec(0));
            const auto& state = stream->getMtpAsyncDeviceState();
            for (const auto& tensor :
                 {state.accept_len_gpu, state.accept_tokens_gpu, state.next_seq_len_gpu, state.propose_tokens_gpu}) {
                ASSERT_TRUE(tensor.defined());
                result.states.push_back(tensor.cpu().clone());
            }
            EXPECT_EQ(state.draft_all_probs_gpu.defined(), !mtpFusionEnabled());
            result.states.push_back(state.draft_all_probs_gpu.defined() ? state.draft_all_probs_gpu.cpu().clone() :
                                                                          torch::Tensor());
            stream->releaseResource();
        }
        ASSERT_TRUE(a->token_logprobs_.defined());
        ASSERT_TRUE(a->top_logprob_token_ids_.defined());
        ASSERT_TRUE(a->top_logprobs_.defined());
        result.logprobs     = a->token_logprobs_.narrow(1, 0, a->logprobs_history_size_).clone();
        result.top_ids      = a->top_logprob_token_ids_.narrow(1, 0, a->logprobs_history_size_).clone();
        result.top_logprobs = a->top_logprobs_.narrow(1, 0, a->logprobs_history_size_).clone();
        EXPECT_EQ(a->logprobs_history_size_, 6);
    };

    Result barrier, overlap;
    run(false, barrier);
    ASSERT_FALSE(HasFatalFailure());
    run(true, overlap);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(overlap.tokens, barrier.tokens);
    EXPECT_EQ(overlap.grammar_calls, barrier.grammar_calls);
    ASSERT_EQ(overlap.target_inputs.size(), barrier.target_inputs.size());
    for (size_t i = 0; i < overlap.target_inputs.size(); ++i) {
        checkTensorEqual(overlap.target_inputs[i], barrier.target_inputs[i]);
    }
    ASSERT_EQ(overlap.states.size(), barrier.states.size());
    for (size_t i = 0; i < overlap.states.size(); ++i) {
        checkTensorEqual(overlap.states[i], barrier.states[i]);
    }
    checkTensorEqual(overlap.logprobs, barrier.logprobs);
    checkTensorEqual(overlap.top_ids, barrier.top_ids);
    checkTensorEqual(overlap.top_logprobs, barrier.top_logprobs);
}

TEST_F(MtpExecutorTest, testFakeDecodeHiddenWidthDistinguishesHy4AndCompressedResiduals) {
    MtpExecutorTestConfig test_config;
    auto                  components = createMtpExecutorComponents(test_config);
    components.model_config.hc_mult  = 4;
    auto make_fake_stream            = [&]() {
        return MtpExecutor::createMinFakeDecodeStream(test_config.gen_num_per_cycle,
                                                      components.model_config,
                                                      components.runtime_config,
                                                      components.resource_context,
                                                      test_config.vocab_size);
    };
    auto hy4 = make_fake_stream();
    ASSERT_TRUE(hy4->getSPOutputBuffer());
    ASSERT_TRUE(hy4->getSPOutputBuffer()->hidden_states.defined());
    EXPECT_EQ(hy4->getSPOutputBuffer()->hidden_states.size(-1), components.model_config.hidden_size);

    components.model_config.attn_config.layer_compress_ratios = {4};
    auto compressed                                           = make_fake_stream();
    ASSERT_TRUE(compressed->getSPOutputBuffer());
    EXPECT_EQ(compressed->getSPOutputBuffer()->hidden_states.size(-1), components.model_config.hidden_size * 4);
}

TEST_F(MtpExecutorTest, testFakeDecodeStreamGetsReusableIndexerSeedBeforeColdCheck) {
    MtpExecutorTestConfig test_config;
    auto                  components                = createMtpExecutorComponents(test_config);
    components.executor->mtp_indexer_share_enabled_ = true;
    components.executor->mtp_indexer_topk_          = 4;

    auto fake_stream = MtpExecutor::createMinFakeDecodeStream(test_config.gen_num_per_cycle,
                                                              components.model_config,
                                                              components.runtime_config,
                                                              components.resource_context,
                                                              test_config.vocab_size);
    ASSERT_FALSE(fake_stream->getMtpAsyncDeviceState().mtp_indexer_topk_gpu.defined());
    std::list<GenerateStreamPtr> streams{fake_stream};
    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;
    components.executor->prepareStreams(streams, prefill_streams, decode_streams);

    ASSERT_TRUE(prefill_streams.empty());
    ASSERT_EQ(decode_streams.size(), 1);
    const auto& seed = fake_stream->getMtpAsyncDeviceState().mtp_indexer_topk_gpu;
    ASSERT_TRUE(seed.defined());
    ASSERT_TRUE(seed.is_cuda());
    ASSERT_EQ(seed.scalar_type(), torch::kInt32);
    ASSERT_EQ(seed.sizes(), torch::IntArrayRef({1, 4}));
    const auto seed_cpu = seed.cpu();
    EXPECT_EQ(seed_cpu.index({0, 0}).item<int32_t>(), 0);
    EXPECT_EQ(seed_cpu.index({0, 1}).item<int32_t>(), 1);
    EXPECT_EQ(seed_cpu.index({0, 2}).item<int32_t>(), -1);
    EXPECT_EQ(seed_cpu.index({0, 3}).item<int32_t>(), -1);

    // NormalEngine creates a fresh fake stream on every idle DP step. Verify
    // the seed is synthesized for each new object rather than relying on a
    // previous fake stream's state.
    auto next_fake_stream = MtpExecutor::createMinFakeDecodeStream(test_config.gen_num_per_cycle,
                                                                   components.model_config,
                                                                   components.runtime_config,
                                                                   components.resource_context,
                                                                   test_config.vocab_size);
    std::list<GenerateStreamPtr> next_streams{next_fake_stream};
    prefill_streams.clear();
    decode_streams.clear();
    components.executor->prepareStreams(next_streams, prefill_streams, decode_streams);
    ASSERT_TRUE(next_fake_stream->getMtpAsyncDeviceState().mtp_indexer_topk_gpu.defined());

    components.executor->mtp_indexer_share_enabled_ = false;
    auto disabled_fake_stream = MtpExecutor::createMinFakeDecodeStream(test_config.gen_num_per_cycle,
                                                                       components.model_config,
                                                                       components.runtime_config,
                                                                       components.resource_context,
                                                                       test_config.vocab_size);
    std::list<GenerateStreamPtr> disabled_streams{disabled_fake_stream};
    prefill_streams.clear();
    decode_streams.clear();
    components.executor->prepareStreams(disabled_streams, prefill_streams, decode_streams);
    EXPECT_FALSE(disabled_fake_stream->getMtpAsyncDeviceState().mtp_indexer_topk_gpu.defined());
}

TEST_F(MtpExecutorTest, testComputeMtpTargetLogprobsKeepsOnlyCompactFp32Statistics) {
    auto logits = torch::tensor({3.0f, -1.0f, 0.5f, 2.0f, 0.0f, 4.0f, 1.0f, 3.0f}, torch::kFloat16).reshape({2, 4});
    auto expected_logprobs   = torch::log_softmax(logits.to(torch::kFloat32), -1);
    auto expected_topk       = expected_logprobs.topk(2, -1, true, true);
    auto emitted_ids         = torch::tensor({3, 1}, torch::kInt64);
    auto raw_logits_snapshot = logits.clone();

    auto result = computeMtpTargetLogprobs(logits, 2, /*real_vocab_size=*/4);

    ASSERT_TRUE(result.defined());
    EXPECT_FALSE(result.finalized());
    EXPECT_EQ(result.raw_logits.scalar_type(), torch::kFloat16);
    EXPECT_EQ(result.raw_logits.numel(), logits.numel());
    EXPECT_FALSE(result.source_row_indices_cpu_owner.defined());
    EXPECT_EQ(result.row_max.scalar_type(), torch::kFloat32);
    EXPECT_EQ(result.row_max.numel(), logits.size(0));
    EXPECT_EQ(result.row_shifted_logsumexp.scalar_type(), torch::kFloat32);
    EXPECT_EQ(result.row_shifted_logsumexp.numel(), logits.size(0));
    EXPECT_EQ(result.top_logits.numel(), logits.size(0) * 2);
    EXPECT_EQ(result.top_logprob_token_ids.scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::equal(result.top_logprob_token_ids, std::get<1>(expected_topk).to(torch::kInt32)));

    // MTP prefill's inherited sampler is allowed to mutate model_output.logits
    // in place. The selected-token gather must use its pre-sampling snapshot,
    // while row normalizers and Top-K remain those captured before sampling.
    logits.fill_(99.0f);
    finalizeMtpTargetLogprobs(result, emitted_ids, raw_logits_snapshot);
    raw_logits_snapshot.fill_(-99.0f);  // Final compact tensors must not alias the override either.
    ASSERT_TRUE(result.finalized());
    EXPECT_FALSE(result.raw_logits.defined());
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());
    auto expected_selected = expected_logprobs.gather(1, emitted_ids.unsqueeze(1)).squeeze(1);
    EXPECT_TRUE(torch::allclose(result.token_logprobs, expected_selected, 1e-3, 1e-3));
    EXPECT_TRUE(torch::allclose(result.top_logprobs, std::get<0>(expected_topk), 1e-3, 1e-3));

    auto no_top = computeMtpTargetLogprobs(expected_logprobs, -1, /*real_vocab_size=*/4);
    finalizeMtpTargetLogprobs(no_top, emitted_ids);
    EXPECT_EQ(no_top.top_logprobs.size(0), 2);
    EXPECT_EQ(no_top.top_logprobs.size(1), 0);
    EXPECT_EQ(no_top.top_logprob_token_ids.size(0), 2);
    EXPECT_EQ(no_top.top_logprob_token_ids.size(1), 0);

    auto clamped_top = computeMtpTargetLogprobs(expected_logprobs, 99, /*real_vocab_size=*/4);
    EXPECT_EQ(clamped_top.maxTopLogprobs(), 4);
}

TEST_F(MtpExecutorTest, testMtpLogprobsPreserveSmallNegativeNormalizationCorrection) {
    // Reconstructing row_max + log(sum(exp(x - row_max))) in FP32 rounds the
    // 1e-6 correction away at this scale and incorrectly returns exactly zero.
    auto logits      = torch::tensor({{80.0f, 66.0f, 65.0f, 64.0f}}, torch::kFloat32);
    auto emitted_ids = torch::tensor({0}, torch::kInt64);
    auto expected    = torch::log_softmax(logits, -1).index({0, 0}).item<float>();
    ASSERT_LT(expected, 0.0f);

    auto cpu_result = computeMtpTargetLogprobs(logits, /*max_top_logprobs=*/1, /*real_vocab_size=*/4);
    finalizeMtpTargetLogprobs(cpu_result, emitted_ids);
    EXPECT_LT(cpu_result.token_logprobs.item<float>(), 0.0f);
    EXPECT_NEAR(cpu_result.token_logprobs.item<float>(), expected, 1e-9f);
    EXPECT_LT(cpu_result.top_logprobs.index({0, 0}).item<float>(), 0.0f);

#if USING_CUDA
    auto cuda_logits = logits.to(torch::kCUDA);
    auto cuda_ids    = emitted_ids.to(torch::kCUDA).to(torch::kInt32).reshape({1, 1});
    auto cuda_result = captureMtpDecodeTargetLogprobs(cuda_logits, /*max_top_logprobs=*/1, /*real_vocab_size=*/4);
    finalizeSelectedMtpTargetLogprobs(cuda_result, cuda_ids, /*selected_dense_row_indices=*/{0});
    EXPECT_LT(cuda_result.token_logprobs.cpu().item<float>(), 0.0f);
    EXPECT_NEAR(cuda_result.token_logprobs.cpu().item<float>(), expected, 1e-9f);
    EXPECT_LT(cuda_result.top_logprobs.cpu().index({0, 0}).item<float>(), 0.0f);
#endif
}

TEST_F(MtpExecutorTest, testFinalizeSelectedMtpTargetLogprobsReducesOnlyAcceptedRows) {
    // P=5 produces six target rows, but an immediate rejection returns only
    // the replacement token from row zero. Capture must not launch any
    // full-vocabulary reduction before acceptance is known.
    constexpr int64_t positions_per_batch = 6;
    auto              logits              = torch::tensor({{3.0f, -1.0f, 0.5f, 2.0f, 1.0f},
                                                           {0.0f, 4.0f, 1.0f, 3.0f, 2.0f},
                                                           {2.0f, 1.0f, 5.0f, 0.0f, 3.0f},
                                                           {1.0f, 3.0f, 0.0f, 4.0f, 2.0f},
                                                           {4.0f, 0.0f, 3.0f, 2.0f, 1.0f},
                                                           {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}},
                                torch::kFloat32);
    auto              emitted_ids = torch::tensor({4, 3, 2, 1, 0, 4}, torch::kInt32).reshape({1, positions_per_batch});

    auto result = captureMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5);

    ASSERT_TRUE(result.defined());
    EXPECT_FALSE(result.finalized());
    EXPECT_EQ(result.raw_logits.size(0), positions_per_batch);
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());
    EXPECT_FALSE(result.top_logprob_token_ids.defined());
    EXPECT_FALSE(result.token_logprobs.defined());
    EXPECT_FALSE(result.top_logprobs.defined());

    finalizeSelectedMtpTargetLogprobs(result, emitted_ids, /*source_row_indices=*/{0});

    ASSERT_TRUE(result.finalized());
    EXPECT_EQ(result.token_logprobs.size(0), 1);
    ASSERT_EQ(result.top_logprob_token_ids.dim(), 2);
    EXPECT_EQ(result.top_logprob_token_ids.size(0), 1);
    EXPECT_EQ(result.top_logprob_token_ids.size(1), 2);
    ASSERT_EQ(result.top_logprobs.dim(), 2);
    EXPECT_EQ(result.top_logprobs.size(0), 1);
    EXPECT_EQ(result.top_logprobs.size(1), 2);
    auto expected          = torch::log_softmax(logits.narrow(0, 0, 1), -1);
    auto expected_selected = expected.index({0, emitted_ids.index({0, 0}).item<int64_t>()});
    auto expected_topk     = expected.topk(2, -1, true, true);
    EXPECT_TRUE(torch::allclose(result.token_logprobs, expected_selected.reshape({1})));
    EXPECT_TRUE(torch::equal(result.top_logprob_token_ids, std::get<1>(expected_topk).to(torch::kInt32)));
    EXPECT_TRUE(torch::allclose(result.top_logprobs, std::get<0>(expected_topk)));
}

TEST_F(MtpExecutorTest, testFinalizeSelectedMtpTargetLogprobsAllowsEmptyContentSelection) {
    auto logits      = torch::tensor({{3.0f, -1.0f, 0.5f, 2.0f}, {0.0f, 4.0f, 1.0f, 3.0f}}, torch::kFloat32);
    auto emitted_ids = torch::tensor({1, 2}, torch::kInt32).reshape({1, 2});
    auto result      = captureMtpDecodeTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/4);

    ASSERT_TRUE(result.retainsFullLmHeadStorage());
    ASSERT_FALSE(result.finalized());
    finalizeSelectedMtpTargetLogprobs(result, emitted_ids, /*selected_dense_row_indices=*/{});

    ASSERT_TRUE(result.finalized());
    EXPECT_FALSE(result.retainsFullLmHeadStorage());
    EXPECT_FALSE(result.raw_logits.defined());
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());
    EXPECT_EQ(result.token_logprobs.sizes(), (torch::IntArrayRef{0}));
    EXPECT_EQ(result.top_logprob_token_ids.sizes(), (torch::IntArrayRef{0, 2}));
    EXPECT_EQ(result.top_logprobs.sizes(), (torch::IntArrayRef{0, 2}));
}

TEST_F(MtpExecutorTest, testDenseMixedMtpDecodeLogprobCaptureRemainsZeroCopy) {
    // Model a P=5 batch where 127/128 streams request logprobs and each emits
    // one row. Decode capture must not materialize the almost-full [762,V]
    // requested-row copy before acceptance is known.
    constexpr int64_t batch_size          = 128;
    constexpr int64_t positions_per_batch = 6;
    constexpr int64_t padded_vocab_size   = 7;
    constexpr int64_t real_vocab_size     = 5;
    const int64_t     dense_rows          = batch_size * positions_per_batch;
    auto              logits = torch::arange(0, padded_vocab_size, torch::kFloat32).repeat({dense_rows, 1});

    auto result = captureMtpDecodeTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/real_vocab_size);

    ASSERT_EQ(result.raw_logits.sizes(), (torch::IntArrayRef{dense_rows, real_vocab_size}));
    EXPECT_EQ(result.raw_logits.data_ptr<float>(), logits.data_ptr<float>());
    EXPECT_TRUE(result.captured_dense_row_indices.empty());
    EXPECT_FALSE(result.source_row_indices.defined());
    EXPECT_TRUE(result.retainsFullLmHeadStorage());
    EXPECT_TRUE(shouldFinalizeMtpTargetLogprobsEarly(/*stream_async_enabled=*/true, result));
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());

    std::vector<int64_t> accepted_requested_rows;
    accepted_requested_rows.reserve(batch_size - 1);
    for (int64_t stream_idx = 0; stream_idx < batch_size - 1; ++stream_idx) {
        accepted_requested_rows.push_back(stream_idx * positions_per_batch);
    }
    auto emitted_ids = torch::zeros({batch_size, positions_per_batch}, torch::kInt32);
    finalizeSelectedMtpTargetLogprobs(result, emitted_ids, accepted_requested_rows);

    ASSERT_TRUE(result.finalized());
    EXPECT_FALSE(result.retainsFullLmHeadStorage());
    EXPECT_EQ(result.token_logprobs.size(0), batch_size - 1);
    EXPECT_EQ(result.top_logprobs.sizes(), (torch::IntArrayRef{batch_size - 1, 2}));
}

TEST_F(MtpExecutorTest, testSparseMtpLogprobCaptureMapsDifferentAcceptanceLengthsOnCpu) {
    // P=2: dense rows [0,3) and [6,9) request logprobs, while the middle
    // stream is plain. The two requesting streams later emit 2 and 1 tokens.
    auto                       logits              = torch::arange(0, 63, torch::kFloat32).reshape({9, 7});
    auto                       emitted_ids         = torch::tensor({4, 3, 2, 1, 0, 4, 3, 2, 1}, torch::kInt32);
    const std::vector<int64_t> captured_dense_rows = {0, 1, 2, 6, 7, 8};
    const std::vector<int64_t> selected_dense_rows = {0, 1, 6};

    // The generic helper keeps an explicit full identity zero-copy, while its
    // sparse mode remains available to prefill and focused utility tests.
    auto identity_capture = captureMtpTargetLogprobs(logits,
                                                     /*max_top_logprobs=*/2,
                                                     /*real_vocab_size=*/5,
                                                     /*captured_dense_row_indices=*/{0, 1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_EQ(identity_capture.raw_logits.data_ptr<float>(), logits.data_ptr<float>());
    EXPECT_TRUE(identity_capture.captured_dense_row_indices.empty());
    EXPECT_TRUE(identity_capture.retainsFullLmHeadStorage());

    auto result = captureMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5, captured_dense_rows);

    ASSERT_EQ(result.raw_logits.sizes(), (torch::IntArrayRef{6, 5}));
    EXPECT_EQ(result.dense_row_count, 9);
    EXPECT_EQ(result.captured_dense_row_indices, captured_dense_rows);
    EXPECT_NE(result.raw_logits.data_ptr<float>(), logits.data_ptr<float>());
    EXPECT_FALSE(result.retainsFullLmHeadStorage());
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());
    EXPECT_FALSE(result.top_logprob_token_ids.defined());

    finalizeSelectedMtpTargetLogprobs(result, emitted_ids, selected_dense_rows);

    ASSERT_TRUE(result.finalized());
    auto dense_indices   = torch::tensor(selected_dense_rows, torch::kInt64);
    auto expected_logits = logits.narrow(1, 0, 5).index_select(0, dense_indices);
    auto expected        = torch::log_softmax(expected_logits, -1);
    auto expected_ids    = emitted_ids.index_select(0, dense_indices).to(torch::kInt64);
    auto expected_topk   = expected.topk(2, -1, true, true);
    EXPECT_TRUE(torch::allclose(result.token_logprobs, expected.gather(1, expected_ids.unsqueeze(1)).squeeze(1)));
    EXPECT_TRUE(torch::equal(result.top_logprob_token_ids, std::get<1>(expected_topk).to(torch::kInt32)));
    EXPECT_TRUE(torch::allclose(result.top_logprobs, std::get<0>(expected_topk)));
}

TEST_F(MtpExecutorTest, testSparseMtpLogprobCaptureMapsDifferentAcceptanceLengthsOnCuda) {
    // Same mixed P=2 layout as the CPU case, with acceptance lengths 1 and 3.
    // Padded vocabulary columns must neither be retained nor reduced.
    auto logits =
        torch::arange(0, 63, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)).reshape({9, 7});
    auto emitted_ids =
        torch::tensor({4, 3, 2, 1, 0, 4, 3, 2, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const std::vector<int64_t> captured_dense_rows = {0, 1, 2, 6, 7, 8};
    const std::vector<int64_t> selected_dense_rows = {0, 6, 7, 8};

    auto identity_capture = captureMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5);
    EXPECT_EQ(identity_capture.raw_logits.data_ptr<at::Half>(), logits.data_ptr<at::Half>());
    EXPECT_TRUE(identity_capture.retainsFullLmHeadStorage());

    auto result = captureMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5, captured_dense_rows);

    ASSERT_EQ(result.raw_logits.sizes(), (torch::IntArrayRef{6, 5}));
    EXPECT_NE(result.raw_logits.data_ptr<at::Half>(), logits.data_ptr<at::Half>());
    EXPECT_FALSE(result.retainsFullLmHeadStorage());
    ASSERT_TRUE(result.source_row_indices_cpu_owner.defined());
    EXPECT_TRUE(result.source_row_indices_cpu_owner.is_pinned());
    EXPECT_EQ(toVec<int64_t>(result.source_row_indices_cpu_owner), captured_dense_rows);
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());

    finalizeSelectedMtpTargetLogprobs(result, emitted_ids, selected_dense_rows);

    ASSERT_TRUE(result.finalized());
    // Dense accepted rows [0,6,7,8] map to compact raw rows [0,3,4,5].
    EXPECT_EQ(toVec<int64_t>(result.source_row_indices_cpu_owner), (std::vector<int64_t>{0, 3, 4, 5}));
    auto dense_indices =
        torch::tensor(selected_dense_rows, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto expected_logits = logits.narrow(1, 0, 5).index_select(0, dense_indices).to(torch::kFloat32);
    auto expected        = torch::log_softmax(expected_logits, -1);
    auto expected_ids    = emitted_ids.index_select(0, dense_indices).to(torch::kInt64);
    auto expected_topk   = expected.topk(2, -1, true, true);
    EXPECT_TRUE(torch::allclose(
        result.token_logprobs.cpu(), expected.gather(1, expected_ids.unsqueeze(1)).squeeze(1).cpu(), 1e-3, 1e-3));
    EXPECT_TRUE(torch::equal(result.top_logprob_token_ids.cpu(), std::get<1>(expected_topk).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::allclose(result.top_logprobs.cpu(), std::get<0>(expected_topk).cpu(), 1e-3, 1e-3));
}

TEST_F(MtpExecutorTest, testComputeMtpTargetLogprobsExcludesPaddedVocabColumns) {
    auto logits      = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f, 100.0f, 90.0f}, {4.0f, 1.0f, 3.0f, 2.0f, 80.0f, 70.0f}},
                                torch::kFloat32);
    auto emitted_ids = torch::tensor({0, 2}, torch::kInt64);
    auto expected    = torch::log_softmax(logits.narrow(1, 0, 4), -1);

    auto result = computeMtpTargetLogprobs(logits, /*max_top_logprobs=*/4, /*real_vocab_size=*/4);
    EXPECT_EQ(result.raw_logits.size(1), 4);
    finalizeMtpTargetLogprobs(result, emitted_ids);

    EXPECT_TRUE(torch::all(result.top_logprob_token_ids < 4).item<bool>());
    EXPECT_TRUE(torch::equal(result.top_logprob_token_ids, std::get<1>(expected.topk(4, -1)).to(torch::kInt32)));
    EXPECT_TRUE(torch::allclose(result.top_logprobs, std::get<0>(expected.topk(4, -1))));
    EXPECT_TRUE(torch::allclose(result.token_logprobs, expected.gather(1, emitted_ids.unsqueeze(1)).squeeze(1)));
}

TEST_F(MtpExecutorTest, testFinalizeMtpTargetLogprobsSafelyClampsInvalidTailIds) {
    auto logits       = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}, {4.0f, 3.0f, 2.0f, 1.0f}, {0.0f, 1.0f, 4.0f, 2.0f}},
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto emitted_ids  = torch::tensor({2, -1, 99}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto original_ids = emitted_ids.clone();
    auto expected_ids = torch::tensor({2, 0, 3}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto expected     = torch::log_softmax(logits, -1).gather(1, expected_ids.unsqueeze(1)).squeeze(1);

    auto result = computeMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/4);
    finalizeMtpTargetLogprobs(result, emitted_ids);

    EXPECT_TRUE(torch::equal(emitted_ids, original_ids));
    EXPECT_TRUE(torch::allclose(result.token_logprobs, expected));
}

TEST_F(MtpExecutorTest, testReshapeMtpTargetAllProbsUsesActualPaddedWidth) {
    auto all_probs = torch::arange(0, 24, torch::kFloat32).reshape({3, 8});

    auto reshaped =
        reshapeMtpTargetAllProbs(all_probs, /*batch_size=*/1, /*positions_per_batch=*/3, /*logits_width=*/8);

    EXPECT_EQ(reshaped.size(0), 1);
    EXPECT_EQ(reshaped.size(1), 3);
    EXPECT_EQ(reshaped.size(2), 8);
    EXPECT_TRUE(torch::equal(reshaped.reshape({3, 8}), all_probs));
}

TEST_F(MtpExecutorTest, testComputeMtpTargetLogprobsSelectsOnlyRequestedRows) {
    auto logits      = torch::arange(0, 30, torch::kFloat32).reshape({6, 5});
    auto emitted_ids = torch::tensor({0, 1, 2, 3, 4, 0}, torch::kInt64);
    auto expected    = torch::log_softmax(logits, -1);

    auto result =
        computeMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5, /*source_row_indices=*/{2, 3});
    ASSERT_TRUE(result.source_row_indices.defined());
    EXPECT_FALSE(result.source_row_indices.is_cuda());
    EXPECT_FALSE(result.source_row_indices_cpu_owner.defined());
    EXPECT_EQ(result.raw_logits.size(0), 2);
    EXPECT_EQ(result.row_max.numel(), 2);
    EXPECT_EQ(result.row_shifted_logsumexp.numel(), 2);
    EXPECT_EQ(result.top_logits.numel(), 4);

    finalizeMtpTargetLogprobs(result, emitted_ids);
    auto requested_ids     = emitted_ids.index_select(0, torch::tensor({2, 3}, torch::kInt64));
    auto expected_selected = expected.narrow(0, 2, 2).gather(1, requested_ids.unsqueeze(1)).squeeze(1);
    EXPECT_TRUE(torch::allclose(result.token_logprobs, expected_selected));
    EXPECT_EQ(result.token_logprobs.numel(), 2);
}

TEST_F(MtpExecutorTest, testCudaSparseMtpTargetRowsUsePinnedAsyncIndexOwner) {
    auto logits =
        torch::arange(0, 24, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({4, 6});
    auto emitted_ids = torch::tensor({0, 1, 2, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto result =
        computeMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5, /*source_row_indices=*/{1, 3});

    ASSERT_TRUE(result.source_row_indices.defined());
    EXPECT_TRUE(result.source_row_indices.is_cuda());
    ASSERT_TRUE(result.source_row_indices_cpu_owner.defined());
    EXPECT_FALSE(result.source_row_indices_cpu_owner.is_cuda());
    EXPECT_TRUE(result.source_row_indices_cpu_owner.is_pinned());
    EXPECT_EQ(toVec<int64_t>(result.source_row_indices_cpu_owner), (std::vector<int64_t>{1, 3}));
    EXPECT_EQ(toVec<int64_t>(result.source_row_indices), (std::vector<int64_t>{1, 3}));
    auto expected_raw_logits = logits.narrow(1, 0, 5).index_select(0, result.source_row_indices);
    EXPECT_TRUE(torch::equal(result.raw_logits.cpu(), expected_raw_logits.cpu()));

    finalizeMtpTargetLogprobs(result, emitted_ids);
    EXPECT_TRUE(result.finalized());
    // Keep the pinned source alive through the later dispatch/event boundary;
    // it is tiny O(requested_rows) state, not a retained logits allocation.
    EXPECT_TRUE(result.source_row_indices_cpu_owner.defined());
}

TEST_F(MtpExecutorTest, testCudaMtpRowLogSoftmaxStatsMatchFp32Reference) {
    auto logits   = torch::tensor({{3.0f, -1.0f, 0.5f, 2.0f, 50.0f}, {0.0f, 4.0f, 1.0f, 3.0f, 60.0f}},
                                torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto expected = torch::log_softmax(logits.narrow(1, 0, 4).to(torch::kFloat32), -1);
    auto ids      = torch::tensor({3, 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto result = computeMtpTargetLogprobs(logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/4);
    finalizeMtpTargetLogprobs(result, ids);

    EXPECT_TRUE(torch::allclose(
        result.token_logprobs.cpu(), expected.gather(1, ids.unsqueeze(1)).squeeze(1).cpu(), 1e-3, 1e-3));
    EXPECT_TRUE(torch::allclose(result.top_logprobs.cpu(), std::get<0>(expected.topk(2, -1)).cpu(), 1e-3, 1e-3));
}

TEST_F(MtpExecutorTest, testCudaSelectedMtpLogprobsAvoidDenseAcceptedRowCopy) {
    // Select five of six dense rows: an accepted-row index_select would create
    // an almost full [5,V] logits copy. The CUDA path must instead produce only
    // O(rows*K) final tensors while reading the source rows by index.
    auto padded_logits = torch::tensor({{3.0f, -1.0f, 0.5f, 2.0f, 1.0f, 100.0f, 99.0f},
                                        {0.0f, 4.0f, 1.0f, 3.0f, 2.0f, 98.0f, 97.0f},
                                        {2.0f, 1.0f, 5.0f, 0.0f, 3.0f, 96.0f, 95.0f},
                                        {1.0f, 3.0f, 0.0f, 4.0f, 2.0f, 94.0f, 93.0f},
                                        {4.0f, 0.0f, 3.0f, 2.0f, 1.0f, 92.0f, 91.0f},
                                        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 90.0f, 89.0f}},
                                       torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto emitted_ids =
        torch::tensor({4, 3, 2, 1, 0, 4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const std::vector<int64_t> selected_rows = {0, 1, 2, 3, 4};

    auto result = captureMtpTargetLogprobs(padded_logits, /*max_top_logprobs=*/5, /*real_vocab_size=*/5);
    finalizeSelectedMtpTargetLogprobs(result, emitted_ids, selected_rows);

    ASSERT_TRUE(result.finalized());
    EXPECT_FALSE(result.raw_logits.defined());
    EXPECT_FALSE(result.row_max.defined());
    EXPECT_FALSE(result.row_shifted_logsumexp.defined());
    EXPECT_FALSE(result.top_logits.defined());
    EXPECT_FALSE(result.source_row_indices.defined());
    ASSERT_TRUE(result.source_row_indices_cpu_owner.defined());
    EXPECT_TRUE(result.source_row_indices_cpu_owner.is_pinned());
    EXPECT_EQ(result.token_logprobs.sizes(), (torch::IntArrayRef{5}));
    EXPECT_EQ(result.top_logprob_token_ids.sizes(), (torch::IntArrayRef{5, 5}));
    EXPECT_EQ(result.top_logprobs.sizes(), (torch::IntArrayRef{5, 5}));

    auto row_indices = torch::tensor(selected_rows, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto selected_logits = padded_logits.narrow(1, 0, 5).index_select(0, row_indices).to(torch::kFloat32);
    auto expected        = torch::log_softmax(selected_logits, -1);
    auto expected_ids    = emitted_ids.index_select(0, row_indices).to(torch::kInt64);
    auto expected_topk   = expected.topk(5, -1, true, true);
    EXPECT_TRUE(torch::allclose(
        result.token_logprobs.cpu(), expected.gather(1, expected_ids.unsqueeze(1)).squeeze(1).cpu(), 1e-3, 1e-3));
    EXPECT_TRUE(torch::equal(result.top_logprob_token_ids.cpu(), std::get<1>(expected_topk).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::allclose(result.top_logprobs.cpu(), std::get<0>(expected_topk).cpu(), 1e-3, 1e-3));

    auto zero_top = captureMtpTargetLogprobs(padded_logits, /*max_top_logprobs=*/0, /*real_vocab_size=*/5);
    finalizeSelectedMtpTargetLogprobs(zero_top, emitted_ids, /*source_row_indices=*/{5});
    ASSERT_TRUE(zero_top.finalized());
    EXPECT_EQ(zero_top.top_logprob_token_ids.sizes(), (torch::IntArrayRef{1, 0}));
    EXPECT_EQ(zero_top.top_logprobs.sizes(), (torch::IntArrayRef{1, 0}));
    auto last_expected = torch::log_softmax(padded_logits.index({5, torch::indexing::Slice(0, 5)}).to(torch::kFloat32),
                                            /*dim=*/0)
                             .index({emitted_ids.index({5}).item<int64_t>()});
    EXPECT_TRUE(torch::allclose(zero_top.token_logprobs.cpu(), last_expected.reshape({1}).cpu(), 1e-3, 1e-3));
}

TEST_F(MtpExecutorTest, testSingleBatchPrefill) {
    autil::EnvGuard                 snapshot_env("RTP_LLM_CACHE_STATUS_SNAPSHOT", "1");
    std::shared_ptr<KVCacheManager> cache_manager;
    MtpExecutorTestConfig           test_config;
    test_config.gen_num_per_cycle = 4;
    bool target_forward_finished  = false;
    bool target_sampler_started   = false;
    int  refresh_count            = 0;
    auto components               = createMtpExecutorComponents(
        test_config,
        [&](const std::list<GenerateStreamPtr>& refresh_streams) {
            EXPECT_TRUE(target_forward_finished);
            EXPECT_FALSE(target_sampler_started);
            EXPECT_EQ(refresh_streams.size(), 1);
            ++refresh_count;
            // SCR prefill runs the real MTP callback before connectors are started.
            ASSERT_NE(cache_manager, nullptr);
            EXPECT_EQ(cache_manager->connectorCoordinator(), nullptr);
            cache_manager->refreshKVCacheInfoSnapshot();
            EXPECT_EQ(cache_manager->cache_status_snapshot_, nullptr);
        },
        /*defer_connector_start=*/true);
    cache_manager = components.executor->cache_manager_;
    ASSERT_NE(cache_manager->allocator_, nullptr);
    ASSERT_EQ(cache_manager->connectorCoordinator(), nullptr);

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
    components.fake_target_model->setForwardCallback([&]() { target_forward_finished = true; });

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
    components.fake_sampler->setForwardCallback([&]() { target_sampler_started = true; });

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
    EXPECT_EQ(refresh_count, 1);
    EXPECT_TRUE(target_sampler_started);

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {0.17, 0.18});

    // After SCR release, snapshot refresh must resume normally.
    cache_manager->startDeferredServices();
    ASSERT_NE(cache_manager->connectorCoordinator(), nullptr);
    cache_manager->refreshKVCacheInfoSnapshot();
    ASSERT_NE(cache_manager->cache_status_snapshot_, nullptr);
    EXPECT_GT(cache_manager->getKVCacheInfo(-1, true).total_kv_cache, 0);
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
    next_draft_input.lm_output_indexes = torch::tensor({2}, torch::kInt32);

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

class IneligibleSpecProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    void process(const SamplerInputs&, size_t, size_t) override {}
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const torch::Tensor&, int32_t) override {}

    bool isStateful() const override {
        return true;
    }

    bool isSpecVerifyEligible() const override {
        return false;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        return request.propose_step;
    }
};

TEST_F(MtpExecutorTest, testErroredSpecLogitsStreamDoesNotAbortExecutor) {
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
    stream->logits_processor_list_.push_back(std::make_shared<IneligibleSpecProcessor>());
    stream->reportError(ErrorCode::INVALID_PARAMS, "grammar accept_token error: parser rejected token");

    auto target_input  = GptModelInputs{};
    auto target_output = GptModelOutputs{};
    // Decode gathering consumes the published GPU token mirrors even after
    // a request-level error; processing must preserve that error without aborting.
    target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.7f, 0.2f, 0.1f, 0.0f}).reshape({2, 4}).to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({3, 2}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({2}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({1}, torch::kInt32);
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

    auto target_sampler_output      = SamplerOutput{torch::tensor({1, 2}, torch::kInt32).reshape({2, 1})};
    target_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({2, 4});
    components.fake_sampler->setInputs({SamplerInputs{target_output.logits.clone()}});
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
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(stream->hasError());
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
