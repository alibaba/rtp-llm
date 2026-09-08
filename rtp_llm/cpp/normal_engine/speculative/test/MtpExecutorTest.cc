#include <algorithm>
#include <array>
#include <memory>
#include <chrono>
#include <limits>
#include <mutex>
#include <thread>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
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

TEST(MtpExecutorPolicyTest, DSparkPrefillCPRequiresPrefillRole) {
    PrefillCPConfig prefill_cp_config;
    prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_TRUE(MtpExecutor::dsparkPrefillCPRoleIsValid(prefill_cp_config, RoleType::PREFILL));

    PrefillCPConfig decode_cp_config;
    decode_cp_config.method = CPRotateMethod::PREFILL_CP;
    EXPECT_FALSE(decode_cp_config.is_enabled());
    EXPECT_TRUE(MtpExecutor::dsparkPrefillCPRoleIsValid(decode_cp_config, RoleType::DECODE));

    // A split-enabled decode/colocated engine would conflate the two roles:
    // its proposal block would be fed through the prefill CP splitter.
    decode_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_FALSE(MtpExecutor::dsparkPrefillCPRoleIsValid(decode_cp_config, RoleType::DECODE));
    EXPECT_FALSE(MtpExecutor::dsparkPrefillCPRoleIsValid(decode_cp_config, RoleType::PDFUSION));
}

TEST(MtpExecutorPolicyTest, DSparkPrefillRoleDisablesDraftGraphCapture) {
    // A DSpARK PREFILL worker only seeds the draft feature KV and never runs
    // the fixed-width decode proposal/commit; capturing those graphs there
    // wastes graph-pool memory and can OOM CP-RR startup.
    EXPECT_FALSE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/true, RoleType::PREFILL));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/true, RoleType::DECODE));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/true, RoleType::PDFUSION));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/false, RoleType::PREFILL));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/false, RoleType::DECODE));
}

TEST(MtpExecutorPolicyTest, CpRestoreSnapshotOwnsMutableHostInput) {
    auto input_lengths = torch::tensor({3683}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    TensorHolder holder;
    auto         snapshot = MtpExecutor::snapshotMutableHostInputToCuda(input_lengths, holder);

    ASSERT_TRUE(snapshot.is_cuda());
    ASSERT_EQ(holder.tensors.size(), 1);
    ASSERT_TRUE(holder.tensors.front().is_pinned());
    ASSERT_NE(holder.tensors.front().data_ptr(), input_lengths.data_ptr());
    input_lengths.fill_(922);
    EXPECT_EQ(holder.tensors.front().item<int32_t>(), 3683);
    EXPECT_EQ(snapshot.cpu().item<int32_t>(), 3683);

    auto device_input    = input_lengths.to(torch::kCUDA);
    auto device_snapshot = MtpExecutor::snapshotMutableHostInputToCuda(device_input, holder);
    EXPECT_EQ(device_snapshot.data_ptr(), device_input.data_ptr());
    EXPECT_EQ(holder.tensors.size(), 1);
}

struct MtpExecutorTestConfig {
    size_t  max_seq_len            = 2048;
    size_t  vocab_size             = 4;
    size_t  num_layers             = 1;
    size_t  gen_num_per_cycle      = 4;
    size_t  vocab_size_override    = 0;  // 0 means use vocab_size
    int64_t mm_position_ids_style  = 0;
    int     position_id_len_factor = 1;

    SpeculativeType sp_type              = SP_TYPE_MTP;
    int64_t         dspark_mask_token_id = -1;
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
        ++forward_count_;
        return output_holder.get();
    }

    size_t forwardCount() const {
        return forward_count_;
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

    void checkTensorField(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
        RTP_LLM_LOG_INFO("check %s", name);
        checkTensorEqual(actual, expected);
    }

    void checkInputs(const GptModelInputs& inputs) {
        GptModelInputs expected_inputs = input_holder.get();
        if (expected_is_target_verify_.has_value()) {
            EXPECT_EQ(inputs.is_target_verify, expected_is_target_verify_.value());
        }
        checkTensorField("combo_tokens", inputs.combo_tokens, expected_inputs.combo_tokens);
        checkTensorField("input_lengths", inputs.input_lengths, expected_inputs.input_lengths);
        checkTensorField("sequence_lengths", inputs.sequence_lengths, expected_inputs.sequence_lengths);
        checkTensorField("prefix_lengths", inputs.prefix_lengths, expected_inputs.prefix_lengths);
        checkTensorField("lm_output_indexes", inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
        checkTensorField("last_hidden_states", inputs.last_hidden_states, expected_inputs.last_hidden_states);
        checkTensorField("combo_position_ids", inputs.combo_position_ids, expected_inputs.combo_position_ids);
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

    void expectTargetVerify(bool expected) {
        expected_is_target_verify_ = expected;
    }

    bool hasPendingPrepareInputs() const {
        return !prepare_input_holder.test_data.empty();
    }

    // Test stand-in for the shared MTP hidden buffer view (DSpARK aux rows).
    void setMtpTargetHiddenStates(torch::Tensor rows) {
        mtp_target_hidden_rows_ = std::move(rows);
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t num_tokens) override {
        if (!mtp_target_hidden_rows_.defined()) {
            return torch::Tensor();
        }
        if (num_tokens < 0) {
            return mtp_target_hidden_rows_;
        }
        return mtp_target_hidden_rows_.slice(0, 0, num_tokens);
    }

private:
    TestDataHolder<GptModelInputs>  input_holder;
    TestDataHolder<GptModelInputs>  prepare_input_holder;
    TestDataHolder<GptModelOutputs> output_holder;
    torch::Tensor                   mtp_target_hidden_rows_;
    size_t                          forward_count_ = 0;
    std::optional<bool>             expected_is_target_verify_;
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

class RejectDraftTokenSpecProcessor: public BaseLogitsProcessor {
public:
    explicit RejectDraftTokenSpecProcessor(int32_t rejected_token, int64_t accepted_token_len):
        rejected_token_(rejected_token), accepted_token_len_(accepted_token_len) {}

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        inputs.logits.narrow(0, start_idx, finish_idx - start_idx).fill_(BaseLogitsProcessor::neg_inf);
        return std::nullopt;
    }
    void                     updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t num_new_tokens) override {
        accepted_token_len_ += num_new_tokens;
        return std::nullopt;
    }

    bool isStateful() const override {
        return true;
    }

    int64_t acceptedTokenLen() const override {
        return accepted_token_len_;
    }

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::SPEC_VERIFY, {}};
    }

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override {
        {
            std::lock_guard<std::mutex> lock(observation_mutex_);
            invocation_thread_id_ = std::this_thread::get_id();
            observed_draft_tokens_.clear();
            if (request.draft_tokens != nullptr && request.propose_step > 0) {
                observed_draft_tokens_.assign(request.draft_tokens, request.draft_tokens + request.propose_step);
            }
        }
        if (request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
            return ErrorResult<int>(int(request.propose_step));
        }
        std::fill_n(request.bitmask_cpu_out,
                    static_cast<size_t>(request.propose_step + 1) * request.bitmask_size_int32,
                    SpecLogitsProcessorRequest::kBitmaskAllowAll);
        if (request.bitmask_size_int32 > 0 && rejected_token_ >= 0
            && static_cast<size_t>(rejected_token_) < request.vocab_size) {
            request.bitmask_cpu_out[rejected_token_ / 32] &= ~(1u << (rejected_token_ % 32));
        }
        if (request.draft_tokens != nullptr && request.draft_tokens[0] == rejected_token_) {
            return ErrorResult<int>(0);
        }
        return ErrorResult<int>(int(request.propose_step));
    }

    std::thread::id invocationThreadId() const {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        return invocation_thread_id_;
    }

    std::vector<int32_t> observedDraftTokens() const {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        return observed_draft_tokens_;
    }

private:
    int32_t rejected_token_;
    int64_t accepted_token_len_;

    mutable std::mutex   observation_mutex_;
    std::thread::id      invocation_thread_id_;
    std::vector<int32_t> observed_draft_tokens_;
};

struct MtpExecutorComponents {
    std::unique_ptr<MtpExecutor>            executor;
    std::unique_ptr<FakeModel>              fake_target_model;
    std::unique_ptr<FakeModel>              fake_draft_model;
    std::unique_ptr<FakeModel>              fake_draft_prefill_model;
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

        model_config.max_seq_len                           = test_config.max_seq_len;
        model_config.vocab_size                            = test_config.vocab_size;
        model_config.num_layers                            = test_config.num_layers;
        model_config.mm_model_config.mm_position_ids_style = test_config.mm_position_ids_style;
        model_config.attn_config.rope_config.index_factor  = test_config.position_id_len_factor;
        sp_config.type                                     = test_config.sp_type;
        sp_config.gen_num_per_cycle                        = test_config.gen_num_per_cycle;
        sp_config.sp_dspark_mask_token_id                  = test_config.dspark_mask_token_id;

        resource_context.cache_manager =
            std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                                            /*block_num=*/10,
                                                                            /*tokens_per_block=*/2,
                                                                            rtp_llm::TYPE_INT8,
                                                                            /*local_head_num_kv=*/128,
                                                                            /*size_per_head=*/256));

        EngineInitParams params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
        params.sp_config        = sp_config;
        if (test_config.vocab_size_override > 0) {
            params.model_config_.vocab_size = test_config.vocab_size_override;
        }

        ModelConfig score_cache_model_config   = params.model_config_;
        ModelConfig propose_cache_model_config = params.model_config_;
        score_cache_model_config.num_layers    = 1;
        propose_cache_model_config.num_layers  = 1;
        setDefaultMhaKVCacheSpecDescs(score_cache_model_config);
        setDefaultMhaKVCacheSpecDescs(propose_cache_model_config);

        KVCacheConfig test_kv_cache_config;
        test_kv_cache_config.test_block_num            = 10;
        test_kv_cache_config.seq_size_per_block        = 2;
        test_kv_cache_config.kernel_seq_size_per_block = 2;

        auto configure_cache_model = [](ModelConfig& cache_model_config) {
            cache_model_config.data_type                    = rtp_llm::TYPE_INT8;
            cache_model_config.attn_config.kv_head_num      = 128;
            cache_model_config.attn_config.size_per_head    = 256;
            cache_model_config.attn_config.tokens_per_block = 2;
            cache_model_config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
        };
        configure_cache_model(score_cache_model_config);
        configure_cache_model(propose_cache_model_config);

        auto cache_sp_config              = sp_config;
        cache_sp_config.gen_num_per_cycle = 1;
        auto cache_config                 = CacheConfigCreator::createSpConfig(score_cache_model_config,
                                                               propose_cache_model_config,
                                                               params.parallelism_config,
                                                               params.runtime_config,
                                                               test_kv_cache_config,
                                                               cache_sp_config,
                                                               /*warm_up_result=*/std::nullopt,
                                                               /*is_mtp=*/true,
                                                               /*is_eagle=*/false);

        // Create propose model engine init params
        auto mtp_model_params   = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        auto mtp_params         = std::make_unique<EngineInitParams>(params);
        mtp_params->py_sp_model = py::none();
        if (test_config.sp_type == SP_TYPE_DSPARK) {
            auto markov_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            mtp_params->gpt_weights.dspark_markov_w1 =
                torch::zeros({static_cast<int64_t>(test_config.vocab_size), 1}, markov_options);
            mtp_params->gpt_weights.dspark_markov_w2 =
                torch::zeros({static_cast<int64_t>(test_config.vocab_size), 1}, markov_options);
        }

        mtp_model_params->push_back(std::move(mtp_params));

        auto propose_params = std::make_unique<ProposeModelEngineInitParams>(
            test_config.sp_type, sp_config.gen_num_per_cycle, std::move(mtp_model_params));

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
        auto fake_draft_prefill_model = std::make_unique<FakeModel>(draft_model_params);
        auto fake_fast_topk_sampler   = std::make_unique<FakeFastTopKSampler>();
        auto fake_speculative_sampler = std::make_unique<FakeSpeculativeSampler>(sp_config.gen_num_per_cycle);
        auto fake_sampler             = std::make_unique<FakeSampler>(SamplerInitParams{});

        MtpExecutorComponents components;
        components.executor                 = std::move(executor);
        components.fake_target_model        = std::move(fake_target_model);
        components.fake_draft_model         = std::move(fake_draft_model);
        components.fake_draft_prefill_model = std::move(fake_draft_prefill_model);
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
                         std::unique_ptr<FakeSampler>            fake_sampler,
                         std::unique_ptr<FakeModel>              fake_draft_prefill_model = nullptr) {
        executor->setTargetModel(std::move(fake_target_model));
        executor->setDraftModel(std::move(fake_draft_model));
        if (fake_draft_prefill_model) {
            executor->setDraftPrefillModel(std::move(fake_draft_prefill_model));
        }
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

TEST_F(MtpExecutorTest, testDSparkPrefillCommitDoesNotUseTargetVerifyContract) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle    = 3;
    test_config.vocab_size_override  = test_config.vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    GenerateStreamPtr stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    GptModelInputs commit_input     = target_input;
    commit_input.last_hidden_states = target_output.all_hidden_states;
    components.fake_draft_prefill_model->setInputs({commit_input});
    components.fake_draft_prefill_model->setOutputs({GptModelOutputs{}});
    components.fake_draft_prefill_model->expectTargetVerify(false);

    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1}, torch::kInt32).reshape({1, 1})};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    auto status = components.executor->process({stream});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ((std::vector<int>{0, 1, 2, 3, 1}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    EXPECT_TRUE(stream->getProposeToken().empty());
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

    next_draft_input.combo_tokens      = torch::tensor({3, 2, 0}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({3}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({2}, torch::kInt32);

    // set fake model outputs
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3, 2, 1, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({5}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
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

    next_draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 3);

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3});
    components.fake_draft_model->expectTargetVerify(false);
    components.fake_draft_prefill_model->setInputs({next_draft_input});
    components.fake_draft_prefill_model->setOutputs({next_draft_output});
    components.fake_draft_prefill_model->expectTargetVerify(false);

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

    // The active draft model runs the decode proposal steps (the first draft
    // token is carried over from the previous cycle); the next-round draft
    // prefill executes on the dedicated prefill slot.
    auto* active_draft_model       = components.fake_draft_model.get();
    auto* draft_prefill_fake_model = components.fake_draft_prefill_model.get();
    auto* fake_target_model        = components.fake_target_model.get();

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    // Verify executor was created successfully
    auto status = components.executor->process({stream1});
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(active_draft_model->forwardCount(), propose_step - 1);
    EXPECT_EQ(draft_prefill_fake_model->forwardCount(), 1u);
    if (components.executor->useAsyncPrepare()) {
        EXPECT_FALSE(fake_target_model->hasPendingPrepareInputs());
    }

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
    target_output.logits =
        torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.2f, 0.1f, 0.8f, 0.4f, 0.7f, 0.2f, 0.1f, 0.0f})
            .reshape({3, 4})
            .to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f}).reshape({3, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({1}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({1}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 1);
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f}).reshape({1, 2});

    components.fake_draft_model->setInputs({draft_input_1, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, next_draft_output});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    auto draft_sampler_output_1 = spec::FastTopKSamplerOutput{
        torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({1, 4}),
        torch::tensor({0}, torch::kInt32).reshape({1, 1})};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}),
        torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({draft_output_1.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output_1, next_draft_sampler_output});

    auto sampler_input         = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3] = BaseLogitsProcessor::neg_inf;
    auto target_sampler_output  = SamplerOutput{torch::tensor({1, 2, 2}, torch::kInt32).reshape({3, 1})};
    target_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 1.0f, 0.0f,
                                                     0.0f, 0.0f, 1.0f, 0.0f})
                                          .reshape({3, 4});
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

TEST_F(MtpExecutorTest, testDSparkGammaThreeSpecLogitsVerifyRunsOnAsyncWorker) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;

    auto components = createMtpExecutorComponents(test_config);

    GenerateStreamPtr stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = false;
    stream->generateConfig()->temperature = 0.0f;
    auto sp_buffer                        = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_buffer->propose_step               = gamma;
    sp_buffer->tokens                     = torch::empty({1, 1}, torch::kInt32);
    stream->setSPOutputBuffer(sp_buffer);

    // Simulate the commit-only prefill handoff: append the first target token
    // but leave proposal/probability/hidden state empty. The first decode
    // round must produce its proposal at the round head.
    StreamSpecUpdateInfo spec_update_info{torch::tensor({{2}}, torch::kInt32), 1, -1, {}, {}};
    stream->specUpdate(spec_update_info);
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_FALSE(stream->getProposeTokensGpu().defined());

    auto processor = std::make_shared<RejectDraftTokenSpecProcessor>(3, stream->outputTokenLen());
    stream->logits_processor_list_.push_back(processor);
    const auto main_thread_id = std::this_thread::get_id();

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 2, 1, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({gamma + 1}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::arange(0, gamma + 1, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.2f, 0.6f, 0.1f, 0.1f, 0.7f, 0.1f, 0.1f, 0.1f})
            .reshape({gamma + 1, vocab_size})
            .to(torch::kCUDA);
    auto target_aux_features =
        torch::arange(0, 2 * (gamma + 1), torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({gamma + 1, 2});
    target_output.all_hidden_states = target_aux_features;
    components.fake_target_model->setMtpTargetHiddenStates(target_aux_features);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    auto sampler_input         = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3] = BaseLogitsProcessor::neg_inf;
    SamplerOutput target_sampler_output{torch::tensor({1, 2, 1, 0}, torch::kInt32).reshape({gamma + 1, 1})};
    target_sampler_output.all_probs = torch::eye(vocab_size, torch::kFloat32).to(torch::kCUDA);

    speculative::SpeculativeSamplerOutput speculative_sampler_output;
    speculative_sampler_output.accept_tokens_cpu = torch::tensor({{1, 0, 0, 0}}, torch::kInt32);
    speculative_sampler_output.accept_tokens     = speculative_sampler_output.accept_tokens_cpu.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({1}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    // Round-head propose call: fixed-width block anchored on the stream's
    // current last token at its own position (committed_end = seq_len - 1),
    // no feature input.
    GptModelInputs draft_input;
    draft_input.combo_tokens      = torch::tensor({2, 0, 0}, torch::kInt32);
    draft_input.input_lengths     = torch::tensor({gamma}, torch::kInt32);
    draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    draft_input.lm_output_indexes = torch::arange(0, gamma, torch::kInt32);

    // Commit call: dense verify rows at the old prefix (accept-independent).
    GptModelInputs commit_input;
    commit_input.combo_tokens       = target_input.combo_tokens;
    commit_input.input_lengths      = torch::tensor({gamma + 1}, torch::kInt32);
    commit_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    commit_input.lm_output_indexes  = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    commit_input.last_hidden_states = target_aux_features;

    GptModelOutputs commit_output;
    commit_output.hidden_states = torch::zeros({gamma + 1, 2}, torch::kFloat32).to(torch::kCUDA);

    GptModelOutputs draft_output;
    draft_output.logits = torch::tensor({0.1f, 0.2f, 0.9f, 0.3f, 0.1f, 0.8f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.9f})
                              .reshape({gamma, vocab_size})
                              .to(torch::kCUDA);
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});
    components.fake_draft_model->expectTargetVerify(true);
    components.fake_draft_prefill_model->setInputs({commit_input});
    components.fake_draft_prefill_model->setOutputs({commit_output});
    components.fake_draft_prefill_model->expectTargetVerify(true);

    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({target_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    auto status = components.executor->process({stream});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NE(std::thread::id(), processor->invocationThreadId());
    EXPECT_NE(main_thread_id, processor->invocationThreadId());
    EXPECT_EQ((std::vector<int32_t>{2, 1, 3}), processor->observedDraftTokens());
}

TEST_F(MtpExecutorTest, testDSparkDraftUsesFlashInferSamplingAndReturnsExactQ) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = true;
    stream->generateConfig()->top_k       = 0;
    stream->generateConfig()->top_p       = 1.0f;
    stream->generateConfig()->temperature = 0.5f;
    StreamGroups stream_groups({stream});

    auto base_logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.7f, 0.1f, 0.0f, -0.1f})
                           .reshape({1, gamma, vocab_size})
                           .to(torch::kCUDA);
    auto anchors       = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto padded_logits = torch::cat(
        {base_logits.reshape({gamma, vocab_size}),
         torch::full({gamma, 3}, 1000.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))},
        1);

    auto output = components.executor->sampleDSparkDraft(stream_groups, padded_logits, anchors);

    ASSERT_TRUE(output.token_ids.is_cuda());
    ASSERT_EQ((std::vector<int64_t>{1, gamma}), output.token_ids.sizes().vec());
    ASSERT_TRUE(output.all_probs.is_cuda());
    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    EXPECT_FALSE(output.token_ids_are_point_mass);
    EXPECT_TRUE(torch::allclose(output.all_probs, torch::softmax(base_logits / 0.5f, -1), 1e-5, 1e-6));
    auto sampled_q = output.all_probs.gather(2, output.token_ids.to(torch::kLong).unsqueeze(-1));
    EXPECT_TRUE(sampled_q.gt(0).all().item<bool>());
}

TEST_F(MtpExecutorTest, testDSparkDraftZeroTemperatureUsesClampedFullSoftmaxQ) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = true;
    stream->generateConfig()->top_k       = 0;
    stream->generateConfig()->top_p       = 0.8f;
    stream->generateConfig()->temperature = 0.0f;
    StreamGroups stream_groups({stream});

    auto base_logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.4f, 0.3f})
                           .reshape({1, gamma, vocab_size})
                           .to(torch::kCUDA);
    auto anchors = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto output =
        components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);

    ASSERT_FALSE(output.token_ids_are_point_mass);
    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    auto expected_q = torch::softmax(base_logits / 1.0e-6f, -1);
    EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));
    EXPECT_TRUE(torch::isfinite(output.all_probs).all().item<bool>());
    EXPECT_TRUE(torch::allclose(output.all_probs.sum(-1).cpu(), torch::ones({1, gamma})));
    EXPECT_TRUE(torch::equal(output.token_ids.cpu(), expected_q.argmax(-1).to(torch::kInt32).cpu()));
}

TEST_F(MtpExecutorTest, testDSparkDraftGreedyRequestsCollapseQToArgmaxOneHot) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto base_logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.4f, 0.3f})
                           .reshape({1, gamma, vocab_size})
                           .to(torch::kCUDA);
    auto anchors    = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto expected_q = torch::softmax(base_logits / 1.0e-6f, -1);

    // do_sample=false is greedy even with a hot request temperature: the target
    // takes its argmax and acceptance requires an exact token match.
    {
        auto stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1});
        stream->generateConfig()->do_sample   = false;
        stream->generateConfig()->temperature = 1.0f;
        StreamGroups stream_groups({stream});

        auto output =
            components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);
        EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));
        EXPECT_TRUE(torch::equal(output.token_ids.cpu(), expected_q.argmax(-1).to(torch::kInt32).cpu()));
    }

    // top_k=1 is greedy as well regardless of do_sample/temperature.
    {
        auto stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1});
        stream->generateConfig()->do_sample   = true;
        stream->generateConfig()->top_k       = 1;
        stream->generateConfig()->temperature = 0.7f;
        StreamGroups stream_groups({stream});

        auto output =
            components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);
        EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));
        EXPECT_TRUE(torch::equal(output.token_ids.cpu(), expected_q.argmax(-1).to(torch::kInt32).cpu()));
    }
}

TEST_F(MtpExecutorTest, testDSparkDraftUsesDenseSequentialMarkovDistribution) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;
    constexpr float   top_p      = 0.7f;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = true;
    stream->generateConfig()->top_k       = 0;
    stream->generateConfig()->top_p       = top_p;
    stream->generateConfig()->temperature = 1.0f;
    StreamGroups stream_groups({stream});

    auto cuda_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto markov_w1 = torch::tensor({0.0f, 0.4f, 0.8f, 1.2f}, torch::kFloat32).reshape({vocab_size, 1}).to(torch::kCUDA);
    auto markov_w2 =
        torch::tensor({-0.5f, -0.1f, 0.3f, 0.7f}, torch::kFloat32).reshape({vocab_size, 1}).to(torch::kCUDA);
    components.executor->dspark_markov_w1_ = markov_w1;
    components.executor->dspark_markov_w2_ = markov_w2;

    auto base_logits = torch::tensor({-10.0f, -10.0f, -10.0f, 10.0f, 0.1f, 0.4f, 0.3f, 0.2f, 0.2f, 0.1f, 0.4f, 0.3f})
                           .reshape({1, gamma, vocab_size})
                           .to(cuda_options);
    auto anchors = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto output =
        components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);

    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    ASSERT_EQ(output.token_ids[0][0].item<int32_t>(), 3);
    auto previous_tokens = anchors.to(torch::kLong);
    for (int64_t step = 0; step < gamma; ++step) {
        auto markov_bias = torch::mm(markov_w1.index_select(0, previous_tokens), markov_w2.transpose(0, 1));
        auto raw_probs   = torch::softmax(base_logits.select(1, step) + markov_bias, -1);
        if (step == 1) {
            auto anchor_bias =
                torch::mm(markov_w1.index_select(0, anchors.to(torch::kLong)), markov_w2.transpose(0, 1));
            EXPECT_FALSE(torch::allclose(raw_probs, torch::softmax(base_logits.select(1, step) + anchor_bias, -1)));
        }
        auto actual_q = output.all_probs.select(1, step);
        EXPECT_TRUE(torch::allclose(actual_q, raw_probs, 1e-5, 1e-6));
        EXPECT_EQ(actual_q.gt(0).sum().item<int64_t>(), vocab_size);
        auto sampled_token = output.token_ids.select(1, step).to(torch::kLong);
        EXPECT_TRUE(actual_q.gather(1, sampled_token.unsqueeze(1)).gt(0).all().item<bool>());
        previous_tokens = std::move(sampled_token);
    }
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
    target_output.logits           = torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.7f, 0.2f, 0.1f, 0.0f})
                               .reshape({2, 4})
                               .to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({1}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({1}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 1);
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f}).reshape({1, 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setInputs({next_draft_input});
    components.fake_draft_model->setOutputs({next_draft_output});

    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}),
        torch::tensor({2}, torch::kInt32).reshape({1, 1})};
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
    next_draft_output.all_hidden_states =
        torch::tensor({0.1f, 0.11f, 1.1f, 1.11f, 1.2f, 1.22f, 1.3f, 1.33f, 1.4f, 1.44f, 1.5f, 1.55f}).reshape({6, 2});

    next_draft_input.combo_tokens      = torch::tensor({3, 3, 0, 2, 2, 1}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({1, 5}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({3, 2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({0, 5}, torch::kInt32);

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

    next_draft_input.last_hidden_states =
        torch::cat({target_output.all_hidden_states.narrow(0, 0, 1), target_output.all_hidden_states.narrow(0, 5, 5)});

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

TEST_F(MtpExecutorTest, testDraftModelDecodeExpandsTargetVerifyPositionIds) {
    size_t propose_step = 4;
    size_t batch_size   = 2;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle      = propose_step;
    test_config.vocab_size_override    = vocab_size;
    test_config.mm_position_ids_style  = MROPE;
    test_config.position_id_len_factor = 3;
    auto components                    = createMtpExecutorComponents(test_config);

    auto stream_model_config                                  = components.model_config;
    stream_model_config.mm_model_config.mm_position_ids_style = MROPE;
    stream_model_config.attn_config.rope_config.index_factor  = 3;
    auto stream1                                              = createContextStream(
        stream_model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3, 0, 1});
    auto stream2 = createContextStream(
        stream_model_config, components.runtime_config, components.resource_context, {1, 2, 3, 0, 1, 2, 3, 0});
    stream1->setContextPositionIds(torch::tensor({0, 0, 0}, torch::kInt32));
    stream2->setContextPositionIds(torch::tensor({0, 0, 0}, torch::kInt32));

    auto sp_output_buffer1    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer1->tokens = torch::tensor({10, 11}, torch::kInt32).reshape({1, 2});
    stream1->setSPOutputBuffer(sp_output_buffer1);

    auto sp_output_buffer2    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer2->tokens = torch::tensor({20, 21}, torch::kInt32).reshape({1, 2});
    stream2->setSPOutputBuffer(sp_output_buffer2);

    StreamGroups stream_groups({stream1, stream2});

    GptModelInputs model_input;
    model_input.combo_tokens       = torch::tensor({11, 21}, torch::kInt32);
    model_input.input_lengths      = torch::tensor({5, 6}, torch::kInt32);
    model_input.sequence_lengths   = torch::tensor({5, 7}, torch::kInt32);
    model_input.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    model_input.last_hidden_states = torch::tensor({0.1f, 0.2f, 1.1f, 1.2f}, torch::kFloat32).reshape({2, 2});
    model_input.combo_position_ids = torch::tensor({5, 5, 5, 7, 7, 7}, torch::kInt32);

    auto makeDraftInput = [](std::vector<int> combo_tokens,
                             std::vector<int> sequence_lengths,
                             torch::Tensor    last_hidden_states,
                             std::vector<int> combo_position_ids = {}) {
        GptModelInputs input;
        input.combo_tokens       = torch::tensor(combo_tokens, torch::kInt32);
        input.input_lengths      = torch::tensor({5, 6}, torch::kInt32);
        input.sequence_lengths   = torch::tensor(sequence_lengths, torch::kInt32);
        input.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
        input.last_hidden_states = std::move(last_hidden_states);
        if (!combo_position_ids.empty()) {
            input.combo_position_ids = torch::tensor(combo_position_ids, torch::kInt32);
        }
        return input;
    };

    auto draft_output_1 = createRandomGptModelOutputs(batch_size, vocab_size, 2);
    auto draft_output_2 = createRandomGptModelOutputs(batch_size, vocab_size, 2);
    auto draft_output_3 = createRandomGptModelOutputs(batch_size, vocab_size, 2);

    components.fake_draft_model->setInputs({
        makeDraftInput({11, 21}, {5, 7}, model_input.last_hidden_states, {5, 5, 5, 7, 7, 7}),
        makeDraftInput({12, 22}, {6, 8}, draft_output_1.all_hidden_states, {6, 6, 6, 8, 8, 8}),
        makeDraftInput({13, 23}, {7, 9}, draft_output_2.all_hidden_states, {7, 7, 7, 9, 9, 9}),
    });
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3});

    spec::FastTopKSamplerOutput draft_sampler_output_1;
    draft_sampler_output_1.token_ids = torch::tensor({12, 22}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_1.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, torch::kFloat32)
                                           .reshape({(int64_t)batch_size, (int64_t)vocab_size});

    spec::FastTopKSamplerOutput draft_sampler_output_2;
    draft_sampler_output_2.token_ids = torch::tensor({13, 23}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_2.all_probs = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, torch::kFloat32)
                                           .reshape({(int64_t)batch_size, (int64_t)vocab_size});

    spec::FastTopKSamplerOutput draft_sampler_output_3;
    draft_sampler_output_3.token_ids = torch::tensor({14, 24}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_3.all_probs = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32)
                                           .reshape({(int64_t)batch_size, (int64_t)vocab_size});

    components.fake_fast_topk_sampler->setInputs({draft_output_1.logits, draft_output_2.logits, draft_output_3.logits});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3});

    components.executor->setDraftModel(std::move(components.fake_draft_model));
    components.executor->setFastTopKSampler(std::move(components.fake_fast_topk_sampler));

    std::vector<torch::Tensor> draft_probs_list;
    torch::Tensor              draft_token_ids_t;
    int64_t                    model_forward_us = 0;
    components.executor->draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t, model_forward_us);

    EXPECT_EQ((std::vector<int>{10, 11, 12, 13, 14, 20, 21, 22, 23, 24}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int>{5, 5}), toVec<int>(model_input.input_lengths));
    // Post draft-position fix: target verification starts one position below
    // the draft-decode base so it first re-writes the carried target token.
    EXPECT_EQ((std::vector<int>{4, 6}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(0, model_input.sequence_lengths.numel());
    EXPECT_EQ((std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), toVec<int>(model_input.lm_output_indexes));

    EXPECT_EQ((std::vector<int>{5, 5, 5, 6, 6, 6, 7, 7, 7, 8,  8,  8,  9,  9,  9,
                                7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11}),
              toVec<int>(model_input.combo_position_ids));
}

TEST_F(MtpExecutorTest, testDSparkDraftMasksInvalidTemperatureAndUsesProbabilitySampling) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 6;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stochastic_stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    auto config         = stochastic_stream->generateConfig();
    config->do_sample   = true;
    config->top_k       = 0;
    config->temperature = std::numeric_limits<float>::quiet_NaN();
    StreamGroups stream_groups({stochastic_stream});

    auto base_logits = torch::tensor({6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}, torch::kFloat32).repeat({gamma, 1});
    base_logits      = base_logits.to(torch::kCUDA);
    auto anchors     = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto output = components.executor->sampleDSparkDraft(stream_groups, base_logits, anchors);

    EXPECT_FLOAT_EQ(config->temperature, 1.0f);
    ASSERT_FALSE(output.token_ids_are_point_mass);
    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    EXPECT_TRUE(output.token_ids.ge(0).all().item<bool>());
    EXPECT_TRUE(output.token_ids.lt(vocab_size).all().item<bool>());
    auto expected_q = torch::softmax(base_logits.reshape({1, gamma, vocab_size}), -1);
    EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));

    config->temperature  = -1.0f;
    auto negative_output = components.executor->sampleDSparkDraft(stream_groups, base_logits, anchors);
    EXPECT_FLOAT_EQ(config->temperature, 1.0f);
    EXPECT_TRUE(torch::allclose(negative_output.all_probs, expected_q));
}

TEST_F(MtpExecutorTest, testDSparkFakeDecodeStartsWithoutProposalState) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 16;

    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    model_config.max_seq_len = 64;
    model_config.vocab_size  = vocab_size;
    model_config.hidden_size = 8;
    model_config.data_type   = TYPE_FP16;

    auto stream =
        MtpExecutor::createMinFakeDecodeStream(gamma, model_config, runtime_config, resource_context, vocab_size, true);
    auto sp_buffer = stream->getSPOutputBuffer();
    ASSERT_NE(sp_buffer, nullptr);
    EXPECT_EQ((std::vector<int64_t>{1, 1}), sp_buffer->tokens.sizes().vec());
    EXPECT_FALSE(sp_buffer->all_probs.defined());
    EXPECT_FALSE(sp_buffer->hidden_states.defined());
    EXPECT_FALSE(stream->getProposeTokensGpu().defined());

    StreamSpecUpdateInfo update_info{
        torch::tensor({7}, torch::kInt32).reshape({1, 1}), 1, -1, torch::Tensor(), torch::Tensor()};
    update_info.speculative_propose_step = 3;
    update_info.accepted_draft_tokens    = 2;
    stream->specUpdate(update_info);

    EXPECT_EQ((std::vector<int32_t>{7}), toVec<int32_t>(sp_buffer->tokens));
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_EQ(stream->spIterCount(), 1);
    EXPECT_EQ((std::vector<int32_t>{1, 1, 0}), stream->speculativeAcceptedTokensPerPos());
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
    EXPECT_TRUE(torch::equal(actual_next, expected_next))
        << "next_seq_len mismatch:\n"
        << actual_next << "\nvs expected:\n"
        << expected_next;

    // Verify hidden_idx = accept_len - 1
    auto expected_idx = (accept_len.to(torch::kInt64) - 1).cpu();
    auto actual_idx   = hidden_idx.cpu();
    EXPECT_TRUE(torch::equal(actual_idx, expected_idx))
        << "hidden_idx mismatch:\n"
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
    RTP_LLM_LOG_INFO("[dispatch-bench] batched: %ld us total, %.2f us/iter",
                     us_batched, (double)us_batched / iterations);
    RTP_LLM_LOG_INFO("[dispatch-bench] scalar:  %ld us total, %.2f us/iter",
                     us_scalar, (double)us_scalar / iterations);
    RTP_LLM_LOG_INFO("[dispatch-bench] speedup: %.1fx", speedup);
}

// MTP-incompatible stateful processor: main's capability contract makes the
// stream (not the whole engine step) fail, so decodeStep must keep running.
class IncompatibleMtpProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }
    void                     updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        return std::nullopt;
    }

    bool isStateful() const override {
        return true;
    }

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::UNSUPPORTED, "test processor has no spec-verify support"};
    }
};

TEST_F(MtpExecutorTest, testErroredSpecLogitsStreamDoesNotAbortExecutor) {
    size_t propose_step = 1;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    auto stream_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto stream_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto stream_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});
    StreamSpecUpdateInfo spec_update_info{stream_new_tokens, 1, 3, stream_hidden_states, stream_draft_token_probs};

    GenerateStreamPtr stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    stream->logits_processor_list_.push_back(std::make_shared<IncompatibleMtpProcessor>());
    stream->reportError(ErrorCode::INVALID_PARAMS, "grammar accept_token error: parser rejected token");

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

}  // namespace rtp_llm
