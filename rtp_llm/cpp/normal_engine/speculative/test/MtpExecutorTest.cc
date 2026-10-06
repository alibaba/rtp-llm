#include <algorithm>
#include <memory>
#include <chrono>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
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

class ScopedDisableCoreDumpOnException {
public:
    ScopedDisableCoreDumpOnException(): saved_(StaticConfig::user_ft_core_dump_on_exception) {
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    ~ScopedDisableCoreDumpOnException() {
        StaticConfig::user_ft_core_dump_on_exception = saved_;
    }

private:
    bool saved_;
};

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

    void drainPendingCacheStore() override {
        if (drain_hook) {
            drain_hook();
        }
    }
    std::function<void()> drain_hook;

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
        EXPECT_EQ(inputs.last_hidden_states_layout, expected_inputs.last_hidden_states_layout)
            << "unexpected MTP hidden layout";
    }

    void setOutputs(const vector<GptModelOutputs>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<GptModelInputs>& inputs) {
        input_holder.push(inputs);
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t num_tokens) override {
        last_requested_mtp_hidden_rows = num_tokens;
        if (!mtp_target_hidden_states.defined() || num_tokens < 0) {
            return mtp_target_hidden_states;
        }
        RTP_LLM_CHECK_WITH_INFO(num_tokens <= mtp_target_hidden_states.size(0),
                                "test MTP hidden request exceeds buffer rows");
        return mtp_target_hidden_states.narrow(0, 0, num_tokens);
    }

    torch::Tensor mtp_target_hidden_states;
    int64_t       last_requested_mtp_hidden_rows = 0;

private:
    TestDataHolder<GptModelInputs>  input_holder;
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
                                           SamplerOutput&                      target_sampler_output,
                                           const torch::Tensor&                active_verify_lengths = {}) override {
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
        if (capture_inputs) {
            captured_inputs = inputs;
        }
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

    bool          capture_inputs = false;
    SamplerInputs captured_inputs;

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

// Synthetic constraint, but the artifact builder, mask application, cap
// application and dispatch below are the real executor process() path.
class FixedOffsetSpecProcessor: public RejectDraftTokenSpecProcessor {
public:
    FixedOffsetSpecProcessor(int cap, int64_t accepted_token_len):
        RejectDraftTokenSpecProcessor(3, accepted_token_len), cap_(cap) {}

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        std::fill_n(request.bitmask_cpu_out,
                    static_cast<size_t>(request.propose_step + 1) * request.bitmask_size_int32,
                    SpecLogitsProcessor::kBitmaskAllowAll);
        if (cap_ < request.propose_step) {
            request.bitmask_cpu_out[cap_ * request.bitmask_size_int32] &= ~(1u << 3);
        }
        return cap_;
    }

private:
    int cap_;
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

TEST_F(MtpExecutorTest, CompactVerifySamplingEligibilityFailsClosed) {
    auto make_inputs = []() {
        SamplerInputs inputs;
        inputs.phase             = LogitsProcessorPhase::MTP_VERIFY;
        inputs.compact_token_ids = true;
        inputs.batch_size = inputs.batch_size_out = 8;
        inputs.top_k                              = torch::zeros({8}, torch::kInt32);
        inputs.all_probs                          = torch::zeros({8, 4}, torch::kFloat32);
        return inputs;
    };
    for (int32_t top_k : {0, -1, 1, 5}) {
        auto inputs = make_inputs();
        inputs.top_k.fill_(top_k);
        EXPECT_EQ(MtpExecutor::canSampleCompactVerifyRows(inputs), top_k <= 1);
    }
    for (int mode = 0; mode < 9; ++mode) {
        auto inputs = make_inputs();
        switch (mode) {
            case 0:
                inputs.compact_token_ids = false;
                break;
            case 1:
                inputs.cum_log_probs = torch::zeros({8});
                break;
            case 2:
                inputs.return_original_all_probs = true;
                break;
            case 3:
                inputs.all_probs = torch::Tensor();
                break;
            case 4:
                inputs.spec_cap_gpu = torch::ones({1});
                break;
            case 5:
                inputs.spec_vocab_mask_gpu = torch::zeros({8, 4}, torch::kBool);
                break;
            case 6:
                inputs.top_k[0] = 1;
                break;  // mixed no-limit and greedy
            case 7:
                inputs.phase = LogitsProcessorPhase::NORMAL_DECODE;
                break;
            case 8:
                inputs.batch_size_out = 9;
                break;
        }
        EXPECT_FALSE(MtpExecutor::canSampleCompactVerifyRows(inputs)) << mode;
    }
}

TEST_F(MtpExecutorTest, AdaptiveVerifyCapRefreshesBookkeepingMirror) {
    const auto                     opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    spec::SpeculativeSamplerOutput output;
    output.accept_len        = torch::tensor({8, 2, 8, 1}, opts);
    output.accept_len_cpu    = torch::tensor({8, 2, 8, 1}, torch::kInt32);
    output.accept_tokens     = torch::arange(32, opts).reshape({4, 8});
    const auto tokens_before = output.accept_tokens.clone();
    MtpExecutor::capDSparkVerifyLengths(output, torch::tensor({3, 5, 7, 1}, opts));
    output.transfer_done_event->synchronize();
    const auto expected = torch::tensor({3, 2, 7, 1}, torch::kInt32);
    EXPECT_TRUE(torch::equal(output.accept_len_cpu, expected));
    EXPECT_TRUE(torch::equal(output.accept_len.cpu(), expected));
    EXPECT_TRUE(torch::equal(output.accept_tokens, tokens_before));
    // A later round must not reuse the earlier CPU mirror or transfer event state.
    MtpExecutor::capDSparkVerifyLengths(output, torch::ones({4}, opts));
    output.transfer_done_event->synchronize();
    EXPECT_TRUE(torch::equal(output.accept_len_cpu, torch::ones({4}, torch::kInt32)));
}

TEST_F(MtpExecutorTest, PrepareDSparkSkipsUnusedMtpProposalMirror) {
    for (bool dspark : {false, true}) {
        auto  components    = createMtpExecutorComponents(MtpExecutorTestConfig{});
        auto& executor      = *components.executor;
        executor.is_dspark_ = dspark;
        auto stream         = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {2, 3});
        stream->setIsContextStream(false);
        auto buffer              = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->tokens           = torch::tensor({3, -1}, torch::kInt32).reshape({1, 2});
        buffer->target_token_gpu = torch::tensor({3}, torch::kInt32).to(torch::kCUDA);
        auto* anchor             = buffer->target_token_gpu.data_ptr<int>();
        stream->setSPOutputBuffer(buffer);
        std::list<GenerateStreamPtr> prefill, decode;
        for (int round = 0; round < 2; ++round) {
            prefill.clear();
            decode.clear();
            executor.prepareStreams({stream}, prefill, decode);
            EXPECT_TRUE(prefill.empty());
            ASSERT_EQ(decode.size(), 1);
            EXPECT_EQ(buffer->target_token_gpu.data_ptr<int>(), anchor);
            EXPECT_EQ(buffer->target_token_gpu.item<int>(), 3);
            if (dspark) {
                EXPECT_FALSE(buffer->propose_tokens_gpu.defined());
            } else {
                ASSERT_TRUE(buffer->propose_tokens_gpu.is_cuda());
                EXPECT_EQ(buffer->propose_tokens_gpu.item<int>(), -1);
            }
        }
    }
}

TEST_F(MtpExecutorTest, DSparkKvLeaseFiltersRetiredRowsAndPreservesEpPhase) {
    auto  components                     = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto& executor                       = *components.executor;
    executor.parallelism_config_.dp_size = 4;
    auto retired =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {2, 3});
    auto healthy =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {1, 2});
    retired->setIsContextStream(false);
    healthy->setIsContextStream(false);
    retired->releaseResource();
    std::vector<GenerateStreamPtr> leases;
    leases.reserve(2);
    auto active = executor.acquireKvExecutionStreams({retired, healthy}, leases);
    ASSERT_EQ(active.size(), 1);
    EXPECT_EQ(active.front(), healthy);
    EXPECT_FALSE(active.front()->isFakeStream());
    ASSERT_EQ(leases.size(), 1);
    healthy->finishKvExecution(1);
    leases.clear();
    active = executor.acquireKvExecutionStreams({retired}, leases);
    ASSERT_EQ(active.size(), 1);
    EXPECT_TRUE(active.front()->isFakeStream());
    EXPECT_FALSE(active.front()->isContextStream());
    EXPECT_TRUE(leases.empty());
    active = executor.acquireKvExecutionStreams({}, leases);
    EXPECT_TRUE(active.empty());  // non-root empty input keeps the old protocol
}

TEST_F(MtpExecutorTest, DSparkKvWorkerExceptionAndLaunchRollbackRetainExecutionOwner) {
    for (bool fail_launch : {false, true}) {
        auto  components    = createMtpExecutorComponents(MtpExecutorTestConfig{});
        auto& executor      = *components.executor;
        executor.is_dspark_ = true;
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = SP_TYPE_DSPARK;
        sp_config.gen_num_per_cycle = 4;
        CacheConfig cache;
        cache.group_types = {CacheGroupType::FULL};
        executor.setBatchProcessor(std::make_unique<MtpBatchStreamProcessor>(
            components.model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, sp_config, false));
        auto resource          = components.resource_context;
        resource.cache_manager = executor.cache_manager_;
        auto stream = createContextStream(components.model_config, components.runtime_config, resource, {2, 3});
        ASSERT_TRUE(stream->initKVBlock().ok());
        stream->setIsContextStream(false);
        auto buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->tokens = torch::tensor({3, -1}, torch::kInt32).reshape({1, 2});
        stream->setSPOutputBuffer(buffer);
        ASSERT_TRUE(stream->tryAcquireKvExecution());
        const auto                            free_before = executor.cache_manager_->freeBlocksNum();
        speculative::SpeculativeSamplerOutput output;
        output.accept_tokens_cpu = torch::tensor({{1, 0, 0, 0, 0}}, torch::kInt32);
        output.accept_len_cpu    = torch::ones({1}, torch::kInt32);
        output.accept_tokens     = output.accept_tokens_cpu.to(torch::kCUDA);
        output.accept_len        = output.accept_len_cpu.to(torch::kCUDA);
        output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
        if (fail_launch) {
            executor.spec_bookkeeping_runner_.launch([] { throw std::runtime_error("previous worker failure"); });
            EXPECT_THROW(executor.dispatchDecodeAsync(StreamGroups({stream}), output, MergedOutput{}, nullptr, nullptr),
                         std::runtime_error);
            EXPECT_FALSE(executor.spec_bookkeeping_runner_.joinAndDrain());
        } else {
            // GPU mirrors remain valid; malformed CPU staging fails inside the
            // real dispatch worker, after launch and dependency submission.
            output.accept_tokens_cpu = torch::empty({1, 0}, torch::kInt32);
            ASSERT_TRUE(
                executor.dispatchDecodeAsync(StreamGroups({stream}), output, MergedOutput{}, nullptr, nullptr).ok());
            EXPECT_TRUE(executor.spec_bookkeeping_runner_.joinAndDrain());
            EXPECT_TRUE(stream->getPendingSwapDoneEvent());
        }
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_EQ(executor.cache_manager_->freeBlocksNum(), free_before);
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "test cancellation after failed dispatch");
        stream->moveToNext();
        EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
        cuda_graph::graphGetCurrentStream().synchronize();
        const torch::Stream producer = cuda_graph::graphGetCurrentStream();
        stream->finishKvExecution(producer.hash());
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
        EXPECT_GT(executor.cache_manager_->freeBlocksNum(), free_before);
    }
}

TEST_F(MtpExecutorTest, DSparkKvProcessExceptionDrainsPrepareBeforeDroppingLease) {
    ScopedDisableCoreDumpOnException no_core;
    auto                             components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto&                            executor   = *components.executor;
    setupFakeModels(&executor,
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));
    executor.is_dspark_    = true;
    executor.role_type_    = RoleType::PREFILL;
    auto resource          = components.resource_context;
    resource.cache_manager = executor.cache_manager_;
    auto stream            = createContextStream(components.model_config, components.runtime_config, resource, {2, 3});
    ASSERT_TRUE(stream->initKVBlock().ok());
    const auto free_before = executor.cache_manager_->freeBlocksNum();
    auto       scratch     = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    cuda_graph::graphGetCurrentStream().synchronize();
    executor.target_verify_prepare_runner_.launch([scratch] {
        scratch.fill_(19);
        throw std::runtime_error("injected prepare failure after GPU submission");
    });
    // The test model deliberately has no queued forward input/output, causing
    // the real process body to fail. Its exception cleanup must still join the
    // independently submitted prepare task and drain its actual GPU stream.
    EXPECT_ANY_THROW(executor.processWithKvLease({stream}, 0));
    EXPECT_TRUE(torch::equal(scratch.cpu(), torch::full({64}, 19, torch::kFloat32)));
    EXPECT_FALSE(executor.target_verify_prepare_runner_.joinAndDrain());
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    EXPECT_EQ(stream->async_bookkeeping_->kv_execution.users.load(), 0);
    EXPECT_EQ(executor.cache_manager_->freeBlocksNum(), free_before);
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "test cancellation after process error");
    stream->moveToNext();
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    EXPECT_GT(executor.cache_manager_->freeBlocksNum(), free_before);
}

TEST_F(MtpExecutorTest, DSparkKvProcessExceptionDrainsExternalReadersBeforeLeaseRelease) {
    ScopedDisableCoreDumpOnException no_core;
    for (const bool fail_reader : {false, true}) {
        auto  components = createMtpExecutorComponents(MtpExecutorTestConfig{});
        auto& executor   = *components.executor;
        auto* target     = components.fake_target_model.get();
        auto* draft      = components.fake_draft_model.get();
        setupFakeModels(&executor,
                        std::move(components.fake_target_model),
                        std::move(components.fake_draft_model),
                        std::move(components.fake_fast_topk_sampler),
                        std::move(components.fake_speculative_sampler),
                        std::move(components.fake_sampler));
        executor.is_dspark_    = true;
        executor.role_type_    = RoleType::PREFILL;
        auto resource          = components.resource_context;
        resource.cache_manager = executor.cache_manager_;
        auto stream = createContextStream(components.model_config, components.runtime_config, resource, {2, 3});
        ASSERT_TRUE(stream->initKVBlock().ok());
        const auto            free_before = executor.cache_manager_->freeBlocksNum();
        CacheStoreAsyncWriter writer;
        writer.init();
        writer.trackExternalTask();
        std::atomic<bool> entered{false};
        bool              draft_drained = false;
        target->drain_hook              = [&] {
            entered.store(true, std::memory_order_release);
            writer.drainIfRunning();
        };
        draft->drain_hook = [&] { draft_drained = true; };
        std::thread callback([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            EXPECT_TRUE(entered.load(std::memory_order_acquire));
            EXPECT_EQ(stream->async_bookkeeping_->kv_execution.users.load(), 1);
            EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
            writer.finishExternalTask(
                fail_reader ? std::make_exception_ptr(std::runtime_error("reader completion failed")) : nullptr);
        });
        // Empty FakeModel input queue injects the forward failure. An outstanding
        // real writer callback must finish while the real cache page remains leased.
        EXPECT_ANY_THROW(executor.processWithKvLease({stream}, 0));
        callback.join();
        EXPECT_TRUE(draft_drained);
        EXPECT_EQ(stream->async_bookkeeping_->kv_execution.users.load(), 0);
        EXPECT_EQ(executor.cache_manager_->freeBlocksNum(), free_before);
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "cancel after reader drain");
        stream->moveToNext();
        EXPECT_EQ(stream->streamCacheResource().isResourceReleased(), !fail_reader);
        if (fail_reader) {
            EXPECT_TRUE(stream->async_bookkeeping_->kv_execution.completion_failed);
            EXPECT_EQ(executor.cache_manager_->freeBlocksNum(), free_before);
        } else {
            EXPECT_GT(executor.cache_manager_->freeBlocksNum(), free_before);
        }
    }
}

TEST_F(MtpExecutorTest, DSparkAsyncFailureRetainsGpuStateAndCommitsHealthyPeer) {
    auto  components    = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto& executor      = *components.executor;
    executor.is_dspark_ = true;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 4;
    CacheConfig cache;
    cache.group_types = {CacheGroupType::FULL};
    executor.setBatchProcessor(std::make_unique<MtpBatchStreamProcessor>(
        components.model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, sp_config, false));
    auto failed =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {2, 3});
    auto healthy =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {1, 2});
    const auto gpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    for (const auto& stream : {failed, healthy}) {
        stream->setIsContextStream(false);
        stream->setNeedReleaseResource(false);
        auto buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->tokens = torch::tensor({3, -1}, torch::kInt32).reshape({1, 2});
        stream->setSPOutputBuffer(buffer);
        GenerateStream::MtpAsyncDeviceState previous;
        previous.accept_len_gpu    = torch::ones({1}, gpu);
        previous.accept_tokens_gpu = torch::tensor({{3, 0, 0, 0, 0}}, gpu);
        previous.next_seq_len_gpu  = torch::tensor({2}, gpu);
        stream->setMtpAsyncDeviceState(std::move(previous));
    }
    const auto                            old_tokens = failed->getAcceptTokensGpu().clone();
    const auto                            old_length = failed->getAcceptLenGpu().clone();
    const auto                            old_seq    = failed->getNextSeqLenGpu().clone();
    speculative::SpeculativeSamplerOutput output;
    output.accept_tokens_cpu = torch::tensor({{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}}, torch::kInt32);
    output.accept_len_cpu    = torch::tensor({1, 1}, torch::kInt32);
    output.success_cpu       = torch::tensor({false, true}, torch::kBool);
    output.accept_tokens     = output.accept_tokens_cpu.to(torch::kCUDA);
    output.accept_len        = output.accept_len_cpu.to(torch::kCUDA);
    output.success           = output.success_cpu.to(torch::kCUDA);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
    EXPECT_TRUE(
        executor.dispatchDecodeAsync(StreamGroups({failed, healthy}), output, MergedOutput{}, nullptr, nullptr).ok());
    executor.spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
    EXPECT_TRUE(failed->hasError());
    EXPECT_EQ(failed->seqLength(), 2);
    EXPECT_TRUE(torch::equal(failed->getAcceptTokensGpu(), old_tokens));
    EXPECT_TRUE(torch::equal(failed->getAcceptLenGpu(), old_length));
    EXPECT_TRUE(torch::equal(failed->getNextSeqLenGpu(), old_seq));
    EXPECT_EQ(failed->getSPOutputBuffer()->target_token_gpu.cpu().item<int>(), 3);
    EXPECT_FALSE(healthy->hasError());
    EXPECT_EQ(healthy->seqLength(), 3);
}

TEST_F(MtpExecutorTest, DSparkAsyncMixedPriorWidthAndMissingStateRetainAnchors) {
    auto  components    = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto& executor      = *components.executor;
    executor.is_dspark_ = true;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 4;
    CacheConfig cache;
    cache.group_types = {CacheGroupType::FULL};
    executor.setBatchProcessor(std::make_unique<MtpBatchStreamProcessor>(
        components.model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, sp_config, false));
    auto changed =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {2, 3});
    auto missing =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {2, 3});
    auto healthy =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {1, 2});
    const auto gpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    for (const auto& stream : {changed, missing, healthy}) {
        stream->setIsContextStream(false);
        stream->setNeedReleaseResource(false);
        auto buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->tokens = torch::tensor({3, -1}, torch::kInt32).reshape({1, 2});
        stream->setSPOutputBuffer(buffer);
    }
    GenerateStream::MtpAsyncDeviceState previous;
    previous.accept_len_gpu    = torch::tensor({2}, gpu);
    previous.accept_tokens_gpu = torch::tensor({{3, 7, 9}}, gpu);
    previous.next_seq_len_gpu  = torch::tensor({2}, gpu);
    changed->setMtpAsyncDeviceState(std::move(previous));
    speculative::SpeculativeSamplerOutput output;
    output.accept_tokens_cpu = torch::tensor({{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 2, 3, 0, 0}}, torch::kInt32);
    // Failed rows can have no accepted token. Their next anchor must come
    // from retained state, never from a negative index into sampler output.
    output.accept_len_cpu = torch::tensor({0, 0, 3}, torch::kInt32);
    output.success_cpu    = torch::tensor({false, false, true}, torch::kBool);
    output.accept_tokens  = output.accept_tokens_cpu.to(torch::kCUDA);
    output.accept_len     = output.accept_len_cpu.to(torch::kCUDA);
    output.success        = output.success_cpu.to(torch::kCUDA);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
    auto rejection_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    auto draft_event     = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    rejection_event->record(cuda_graph::graphGetCurrentStream());
    draft_event->record(cuda_graph::graphGetCurrentStream());
    EXPECT_TRUE(executor
                    .dispatchDecodeAsync(
                        StreamGroups({changed, missing, healthy}), output, MergedOutput{}, rejection_event, draft_event)
                    .ok());
    // The worker and published views must own their inputs independently of
    // caller scope. Exercise a non-first failed row beside a multi-token peer.
    output = speculative::SpeculativeSamplerOutput{};
    rejection_event.reset();
    draft_event.reset();
    executor.spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
    for (const auto& pair : {std::make_pair(changed, 7), std::make_pair(missing, 3)}) {
        const auto& stream = pair.first;
        EXPECT_TRUE(stream->hasError());
        EXPECT_EQ(stream->seqLength(), 2);
        EXPECT_TRUE(torch::equal(stream->getAcceptTokensGpu().cpu(), torch::full({1, 5}, pair.second, torch::kInt32)));
        EXPECT_TRUE(torch::equal(stream->getAcceptLenGpu().cpu(), torch::ones({1}, torch::kInt32)));
        EXPECT_EQ(stream->getNextSeqLenGpu().cpu().item<int>(), 2);
        EXPECT_EQ(stream->getSPOutputBuffer()->target_token_gpu.cpu().item<int>(), pair.second);
    }
    EXPECT_FALSE(healthy->hasError());
    EXPECT_EQ(healthy->seqLength(), 5);
    EXPECT_EQ(healthy->getSPOutputBuffer()->target_token_gpu.cpu().item<int>(), 3);
}

TEST_F(MtpExecutorTest, DSparkVerifyBudgetDoesNotChangeProposalWidthOrLegacyMtp) {
    MtpExecutorTestConfig config;
    config.gen_num_per_cycle = 7;
    auto  components         = createMtpExecutorComponents(config);
    auto& executor           = *components.executor;
    EXPECT_EQ(executor.verifySteps(), 7);
    for (const size_t budget : {1, 3, 4, 5, 7}) {
        executor.dspark_verify_step_ = budget;
        executor.is_dspark_          = false;
        EXPECT_EQ(executor.verifySteps(), 7);
        executor.is_dspark_ = true;
        EXPECT_EQ(executor.verifySteps(), budget);
        EXPECT_EQ(executor.propose_step_, 7);
    }
    executor.dspark_verify_step_ = 0;
    EXPECT_EQ(executor.verifySteps(), 7);
}

TEST_F(MtpExecutorTest, AcceptMetricsReportCurrentRoundWithoutPendingTail) {
    MtpExecutorTestConfig config;
    config.gen_num_per_cycle = 7;
    auto  components         = createMtpExecutorComponents(config);
    auto& executor           = *components.executor;
    // Synthetic acceptance inputs exercise real CPU/CUDA metric staging, not
    // model accuracy. Every round must finish before the next dataset starts.
    for (const bool cuda : {false, true}) {
        for (const std::vector<int32_t>& lengths :
             {std::vector<int32_t>{3}, std::vector<int32_t>{1, 8}, std::vector<int32_t>{2}}) {
            std::list<GenerateStreamPtr> streams;
            int64_t                      expected_sum = 0;
            for (const auto length : lengths) {
                expected_sum += length;
                streams.push_back(createContextStream(
                    components.model_config, components.runtime_config, components.resource_context, {0, 1}));
            }
            auto accept_len = torch::tensor(lengths, torch::kInt32);
            auto ready      = cuda_graph::makeGraphEvent();
            if (cuda) {
                accept_len = accept_len.cuda();
                ready.record(cuda_graph::graphGetCurrentStream());
            }
            executor.stageAcceptLenMetrics(accept_len, ready, streams.size());
            MtpMetricsCollector collector;
            executor.collectDecodeMetrics(StreamGroups(streams), collector);
            EXPECT_EQ(collector.sp_engine_collector.total_accepted_token_num, expected_sum);
            EXPECT_EQ(collector.sp_engine_collector.total_stream_num, lengths.size());
            EXPECT_EQ(collector.sp_engine_collector.total_propose_token_num, lengths.size() * 7);
            EXPECT_EQ(collector.executor_collector.execute_token_size, expected_sum);
            EXPECT_FALSE(executor.consumePendingAcceptLenMetrics().valid);
            EXPECT_FALSE(executor.metrics_accept_len_sum_cpu_.defined());
            EXPECT_FALSE(executor.metrics_accept_len_sum_gpu_.defined());
        }
    }
}

TEST_F(MtpExecutorTest, testMtpHiddenOverrideUsesExplicitCpLocalRowsAndLayout) {
    auto  components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto* source     = components.fake_target_model.get();
    source->mtp_target_hidden_states =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({6, 2});

    GptModelInputs input;
    input.combo_tokens = torch::arange(6, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    ASSERT_TRUE(components.executor->maybeOverrideLastHiddenWithMtpBuffer(
        input, *source, MtpHiddenStatesLayout::CP_LOCAL, /*requested_rows=*/4));
    EXPECT_EQ(source->last_requested_mtp_hidden_rows, 4);
    EXPECT_EQ(input.last_hidden_states.size(0), 4);
    EXPECT_EQ(input.last_hidden_states_layout, MtpHiddenStatesLayout::CP_LOCAL);
    checkTensorEqual(input.last_hidden_states, source->mtp_target_hidden_states.narrow(0, 0, 4));

    ASSERT_TRUE(components.executor->maybeOverrideLastHiddenWithMtpBuffer(input, *source));
    EXPECT_EQ(source->last_requested_mtp_hidden_rows, 6);
    EXPECT_EQ(input.last_hidden_states.size(0), 6);
    EXPECT_EQ(input.last_hidden_states_layout, MtpHiddenStatesLayout::GLOBAL);
}

TEST_F(MtpExecutorTest, testMtpHiddenCarrierRejectsMissingOrInvalidLayout) {
    GptModelInputs input;
    auto           hidden = torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    ScopedDisableCoreDumpOnException disable_core_dump;

    EXPECT_THROW(input.setLastHiddenStates(hidden, MtpHiddenStatesLayout::NONE), std::exception);
    EXPECT_THROW(input.setLastHiddenStates(hidden, static_cast<MtpHiddenStatesLayout>(99)), std::exception);

    input.setLastHiddenStates(torch::empty({0, 4}, hidden.options()), MtpHiddenStatesLayout::GLOBAL);
    EXPECT_EQ(input.last_hidden_states_layout, MtpHiddenStatesLayout::NONE);
    input.clearLastHiddenStates();
    EXPECT_FALSE(input.last_hidden_states.defined());
    EXPECT_EQ(input.last_hidden_states_layout, MtpHiddenStatesLayout::NONE);
}

TEST_F(MtpExecutorTest, testMtpHiddenOverrideRejectsInvalidRowPolicyAndMissingCpLocalBuffer) {
    auto  components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto* source     = components.fake_target_model.get();

    GptModelInputs input;
    input.combo_tokens = torch::arange(6, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    ScopedDisableCoreDumpOnException disable_core_dump;

    EXPECT_THROW(components.executor->maybeOverrideLastHiddenWithMtpBuffer(
                     input, *source, MtpHiddenStatesLayout::GLOBAL, /*requested_rows=*/4),
                 std::exception);
    EXPECT_THROW(components.executor->maybeOverrideLastHiddenWithMtpBuffer(
                     input, *source, MtpHiddenStatesLayout::CP_LOCAL, /*requested_rows=*/0),
                 std::exception);
    EXPECT_THROW(components.executor->maybeOverrideLastHiddenWithMtpBuffer(
                     input, *source, static_cast<MtpHiddenStatesLayout>(99), /*requested_rows=*/-1),
                 std::exception);
    EXPECT_THROW(components.executor->maybeOverrideLastHiddenWithMtpBuffer(
                     input, *source, MtpHiddenStatesLayout::CP_LOCAL, /*requested_rows=*/4),
                 std::exception);
}

TEST_F(MtpExecutorTest, testDeterministicDraftSamplerReportsDraftPointMassAndMappedToken) {
    auto d2t_map = torch::tensor({0, 1, 3, 2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    spec::FastTopKSampler sampler(d2t_map, spec::DraftProposalMode::DETERMINISTIC);
    auto                  logits =
        torch::tensor({{0.0f, 1.0f, 4.0f, 2.0f}}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto output = sampler.forward(logits);

    EXPECT_EQ(output.token_ids.item<int64_t>(), 3);
    checkTensorEqual(output.all_probs, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}}).to(torch::kCUDA));
}

TEST_F(MtpExecutorTest, testLegacyDraftSamplerPreservesSoftmaxProposal) {
    auto d2t_map = torch::tensor({0, 1, 3, 2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    spec::FastTopKSampler sampler(d2t_map);
    auto                  logits =
        torch::tensor({{0.0f, 1.0f, 4.0f, 2.0f}}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto output = sampler.forward(logits);

    EXPECT_EQ(output.token_ids.item<int64_t>(), 3);
    checkTensorEqual(output.all_probs, torch::softmax(logits, -1));
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
    auto draft_input              = GptModelInputs{};
    auto draft_output             = GptModelOutputs{};
    draft_input.combo_tokens      = torch::tensor({1, 2, 3, 1}, torch::kInt32);
    draft_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    draft_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    draft_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);
    draft_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
    draft_output.logits = torch::tensor({0.5f, 0.6f, 0.7f, 0.8f}).reshape({(int64_t)batch_size, 4});
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

    draft_input.combo_tokens      = torch::tensor({1, 2, 3, 1, 3, 0}, torch::kInt32);
    draft_input.input_lengths     = torch::tensor({4, 2}, torch::kInt32);
    draft_input.prefix_lengths    = torch::tensor({0, 0}, torch::kInt32);
    draft_input.lm_output_indexes = torch::tensor({3, 5}, torch::kInt32);
    draft_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
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

    draft_input_1.combo_tokens      = torch::tensor({3}, torch::kInt32);
    draft_input_1.input_lengths     = torch::tensor({2}, torch::kInt32);
    draft_input_1.sequence_lengths  = torch::tensor({3}, torch::kInt32);
    draft_input_1.lm_output_indexes = torch::tensor({0}, torch::kInt32);
    draft_input_1.setLastHiddenStates(stream1_hidden_states, MtpHiddenStatesLayout::GLOBAL);

    draft_input_2.combo_tokens      = torch::tensor({2}, torch::kInt32);
    draft_input_2.input_lengths     = torch::tensor({2}, torch::kInt32);
    draft_input_2.sequence_lengths  = torch::tensor({4}, torch::kInt32);
    draft_input_2.lm_output_indexes = torch::tensor({0}, torch::kInt32);
    draft_input_2.setLastHiddenStates(draft_output_1.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

    draft_input_3.combo_tokens      = torch::tensor({1}, torch::kInt32);
    draft_input_3.input_lengths     = torch::tensor({2}, torch::kInt32);
    draft_input_3.sequence_lengths  = torch::tensor({5}, torch::kInt32);
    draft_input_3.lm_output_indexes = torch::tensor({0}, torch::kInt32);
    draft_input_3.setLastHiddenStates(draft_output_2.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

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

    next_draft_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

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

    // Exercise production process() metric gating as well as the helper test.
    // This single decode must not strand acceptance until a second request.
    // The model/sampler fixtures isolate bookkeeping, not model precision.
    components.executor->metrics_reporter_ =
        std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags());
    auto status = components.executor->process({stream1});
    ASSERT_TRUE(status.ok());
    EXPECT_FALSE(components.executor->metrics_accept_len_sum_cpu_.defined());
    EXPECT_FALSE(components.executor->consumePendingAcceptLenMetrics().valid);

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 2, 0}, {0, 1}, {0.0, 1.0, 0.0, 0.0}, {0.3, 0.33});
}

TEST_F(MtpExecutorTest, SpecLogitsCapCompactAndHistoryStrideMixedCapsWidths4567ThroughProcess) {
    // Exercise the anonymous production cap helper through process(), not a
    // copied helper. Legacy proposal fixtures isolate the shared cap consumer;
    // this does not claim coverage of DSpark proposal generation or sampling.
    constexpr int64_t                                    batch = 3;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> retained;
    std::vector<SamplerInputs>                           retained_artifacts;
    for (const int64_t width : {4, 7, 5, 6, 4}) {
        for (const int64_t stride : {1, 9}) {
            SCOPED_TRACE(::testing::Message() << "width=" << width << " stride=" << stride);
            MtpExecutorTestConfig config;
            config.gen_num_per_cycle                = width;
            config.vocab_size_override              = 4;
            auto                         components = createMtpExecutorComponents(config);
            const std::vector<int32_t>   caps       = {0, static_cast<int32_t>(width / 2), static_cast<int32_t>(width)};
            std::list<GenerateStreamPtr> streams;
            auto                         hidden = torch::zeros({batch, 2});
            for (int64_t row = 0; row < batch; ++row) {
                StreamSpecUpdateInfo update{torch::tensor({{2}}, torch::kInt32),
                                            1,
                                            3,
                                            hidden.narrow(0, row, 1),
                                            torch::tensor({{0.f, 0.f, 0.f, 1.f}})};
                auto                 stream = createDecodeStream(
                    components.model_config, components.runtime_config, components.resource_context, {0, 1}, update);
                stream->logits_processor_list_.push_back(
                    std::make_shared<FixedOffsetSpecProcessor>(caps[row], stream->outputTokenLen()));
                streams.push_back(stream);
            }

            std::vector<GptModelInputs>              draft_inputs;
            std::vector<GptModelOutputs>             draft_outputs;
            std::vector<torch::Tensor>               draft_logits;
            std::vector<spec::FastTopKSamplerOutput> draft_samples;
            for (int64_t step = 0; step < width - 1; ++step) {
                GptModelInputs input;
                input.combo_tokens      = torch::full({batch}, 3, torch::kInt32);
                input.input_lengths     = torch::full({batch}, 2, torch::kInt32);
                input.sequence_lengths  = torch::full({batch}, 3 + step, torch::kInt32);
                input.lm_output_indexes = torch::arange(batch, torch::kInt32);
                input.setLastHiddenStates(hidden, MtpHiddenStatesLayout::GLOBAL);
                GptModelOutputs output;
                output.logits            = torch::ones({batch, 4});
                output.all_hidden_states = hidden;
                draft_inputs.push_back(input);
                draft_outputs.push_back(output);
                draft_logits.push_back(output.logits);
                draft_samples.push_back({torch::zeros({batch, 4}), torch::full({batch, 1}, 3, torch::kInt32)});
            }

            GptModelInputs target_input;
            auto           target_combo = torch::full({batch, width + 1}, 3, torch::kInt32);
            target_combo.select(1, 0).fill_(2);
            target_input.combo_tokens      = target_combo.flatten();
            target_input.input_lengths     = torch::full({batch}, width + 1, torch::kInt32);
            target_input.prefix_lengths    = torch::full({batch}, 2, torch::kInt32);
            target_input.lm_output_indexes = torch::arange(batch * (width + 1), torch::kInt32);
            GptModelOutputs target_output;
            target_output.logits            = torch::ones({batch * (width + 1), 4}).cuda();
            target_output.all_hidden_states = torch::zeros({batch * (width + 1), 2});
            components.fake_target_model->setInputs({target_input});
            components.fake_target_model->setOutputs({target_output});

            auto original_tokens = torch::full({batch, width + 1}, 3, torch::kInt32);
            original_tokens.select(1, width).fill_(2);  // Original rejection-sampler bonus.
            auto                 expected_tokens = original_tokens.clone();
            std::vector<int32_t> indexes;
            for (int64_t row = 0; row < batch; ++row) {
                if (caps[row] < width) {
                    expected_tokens[row][caps[row]].fill_(1);  // Target correction, not history poison.
                }
                indexes.push_back(static_cast<int32_t>(row * (width + 1) + caps[row]));
            }
            GptModelInputs next_input;
            next_input.combo_tokens      = expected_tokens.flatten();
            next_input.input_lengths     = target_input.input_lengths;
            next_input.prefix_lengths    = target_input.prefix_lengths;
            next_input.lm_output_indexes = torch::tensor(indexes, torch::kInt32);
            next_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
            GptModelOutputs next_output;
            next_output.logits            = torch::ones({batch, 4});
            next_output.all_hidden_states = target_output.all_hidden_states;
            draft_inputs.push_back(next_input);
            draft_outputs.push_back(next_output);
            draft_logits.push_back(next_output.logits);
            draft_samples.push_back({torch::zeros({batch, 4}), torch::zeros({batch, 1}, torch::kInt32)});
            components.fake_draft_model->setInputs(draft_inputs);
            components.fake_draft_model->setOutputs(draft_outputs);
            components.fake_fast_topk_sampler->setInputs(draft_logits);
            components.fake_fast_topk_sampler->setOutputs(draft_samples);

            SamplerInputs expected_sampler_input{target_output.logits.clone()};
            for (int64_t row = 0; row < batch; ++row) {
                if (caps[row] < width) {
                    expected_sampler_input.logits[row * (width + 1) + caps[row]][3] = BaseLogitsProcessor::neg_inf;
                }
            }
            SamplerOutput target_sample;
            target_sample.token_ids = torch::full({batch * (width + 1), stride}, -99, torch::kInt32).cuda();
            target_sample.token_ids.select(1, stride - 1).fill_(1);
            target_sample.all_probs = torch::zeros({batch * (width + 1), 4});
            auto* sampler           = components.fake_sampler.get();
            sampler->capture_inputs = true;
            sampler->setInputs({expected_sampler_input});
            sampler->setOutputs({target_sample});
            spec::SpeculativeSamplerOutput rejection;
            rejection.accept_tokens_cpu = original_tokens;
            rejection.accept_tokens     = original_tokens.cuda();
            rejection.accept_len_cpu    = torch::full({batch}, width + 1, torch::kInt32);
            rejection.accept_len        = rejection.accept_len_cpu.cuda();
            components.fake_speculative_sampler->setOutputs({rejection});
            setupFakeModels(components.executor.get(),
                            std::move(components.fake_target_model),
                            std::move(components.fake_draft_model),
                            std::move(components.fake_fast_topk_sampler),
                            std::move(components.fake_speculative_sampler),
                            std::move(components.fake_sampler));
            ASSERT_TRUE(components.executor->process(streams).ok());

            const auto& artifacts = sampler->captured_inputs;
            ASSERT_TRUE(artifacts.spec_mask_ready_event);
            ASSERT_TRUE(artifacts.spec_mask_consumed_event);
            // Real cap application re-records this shared event after its D2H
            // mirrors, and records consumed only after the final artifact read.
            rejection.transfer_done_event->synchronize();
            artifacts.spec_mask_consumed_event->synchronize();
            EXPECT_TRUE(artifacts.spec_mask_ready_event->query());
            EXPECT_TRUE(artifacts.spec_mask_consumed_event->query());
            checkTensorEqual(artifacts.spec_cap_gpu, torch::tensor(caps, torch::kInt32));
            retained.emplace_back(artifacts.spec_cap_gpu, torch::tensor(caps, torch::kInt32));
            retained_artifacts.push_back(artifacts);
            int64_t row = 0;
            for (const auto& stream : streams) {
                stream->waitPendingAsyncBookkeeping();
                std::vector<int> expected_complete{0, 1, 2};
                for (int p = 0; p <= caps[row]; ++p) {
                    expected_complete.push_back(expected_tokens[row][p].item<int32_t>());
                }
                // CPU dispatch consumed the newly capped accept_len/tokens;
                // device-state publication must retain the same correction/bonus.
                EXPECT_EQ(stream->getCompleteTokenIds()->completeTokenIdsVec(0), expected_complete);
                ASSERT_TRUE(stream->getAcceptLenGpu().defined());
                ASSERT_TRUE(stream->getAcceptTokensGpu().defined());
                EXPECT_EQ(stream->getAcceptLenGpu().item<int32_t>(), caps[row] + 1);
                checkTensorEqual(stream->getAcceptTokensGpu().reshape({width + 1}), expected_tokens[row]);
                ASSERT_TRUE(stream->getSPOutputBuffer()->target_token_gpu.defined());
                EXPECT_EQ(stream->getSPOutputBuffer()->target_token_gpu.item<int32_t>(),
                          expected_tokens[row][caps[row]].item<int32_t>());
                retained.emplace_back(stream->getAcceptLenGpu(), stream->getAcceptLenGpu().cpu().clone());
                retained.emplace_back(stream->getAcceptTokensGpu(), expected_tokens[row].clone());
                ++row;
            }
            // Keep actual output tensors, not just copies, through subsequent
            // growing/shrinking widths and both target-token strides.
            for (const auto& value : retained) {
                checkTensorEqual(value.first.flatten(), value.second.flatten());
            }
        }
    }
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

    auto draft_input_1              = GptModelInputs{};
    auto draft_output_1             = GptModelOutputs{};
    draft_input_1.combo_tokens      = torch::tensor({3}, torch::kInt32);
    draft_input_1.input_lengths     = torch::tensor({2}, torch::kInt32);
    draft_input_1.sequence_lengths  = torch::tensor({3}, torch::kInt32);
    draft_input_1.lm_output_indexes = torch::tensor({0}, torch::kInt32);
    draft_input_1.setLastHiddenStates(stream_hidden_states, MtpHiddenStatesLayout::GLOBAL);
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

    auto next_draft_input              = GptModelInputs{};
    auto next_draft_output             = GptModelOutputs{};
    next_draft_input.combo_tokens      = torch::tensor({1, 0, 0}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({3}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({0}, torch::kInt32);
    next_draft_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
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
    target_input.sequence_lengths  = torch::empty({0}, torch::kInt32).to(torch::kCUDA);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.7f, 0.2f, 0.1f, 0.0f}).reshape({2, 4}).to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});

    auto next_draft_input              = GptModelInputs{};
    auto next_draft_output             = GptModelOutputs{};
    next_draft_input.combo_tokens      = torch::tensor({1, 2}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    next_draft_input.sequence_lengths  = torch::empty({0}, torch::kInt32).to(torch::kCUDA);
    next_draft_input.lm_output_indexes = torch::tensor({0}, torch::kInt32);
    next_draft_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
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

    draft_input_1.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    draft_input_1.input_lengths     = torch::tensor({3, 2}, torch::kInt32);
    draft_input_1.sequence_lengths  = torch::tensor({4, 3}, torch::kInt32);
    draft_input_1.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    draft_input_1.setLastHiddenStates(torch::tensor({0.03f, 0.04f, 2.1f, 2.12f}).reshape({2, 2}),
                                      MtpHiddenStatesLayout::GLOBAL);

    draft_input_2.combo_tokens      = torch::tensor({1, 0}, torch::kInt32);
    draft_input_2.input_lengths     = torch::tensor({3, 2}, torch::kInt32);
    draft_input_2.sequence_lengths  = torch::tensor({5, 4}, torch::kInt32);
    draft_input_2.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    draft_input_2.setLastHiddenStates(draft_output_1.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

    draft_input_3.combo_tokens      = torch::tensor({2, 2}, torch::kInt32);
    draft_input_3.input_lengths     = torch::tensor({3, 2}, torch::kInt32);
    draft_input_3.sequence_lengths  = torch::tensor({6, 5}, torch::kInt32);
    draft_input_3.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    draft_input_3.setLastHiddenStates(draft_output_2.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

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

    next_draft_input.setLastHiddenStates(target_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

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
