#include <algorithm>
#include <cstddef>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {
namespace {

constexpr size_t kProposalStep = 3;
constexpr size_t kBatchSize    = 2;

struct ForwardRecord {
    size_t calls = 0;
};

struct SamplingRecord {
    size_t target_sampler_calls      = 0;
    size_t fast_topk_sampler_calls   = 0;
    size_t speculative_sampler_calls = 0;
};

class RecordingSampler final: public Sampler {
public:
    explicit RecordingSampler(std::shared_ptr<SamplingRecord> record): Sampler(SamplerInitParams{}), record_(record) {}

    SamplerOutput forward(const SamplerInputs& inputs) override {
        ++record_->target_sampler_calls;
        return Sampler::forward(inputs);
    }

private:
    std::shared_ptr<SamplingRecord> record_;
};

class RecordingFastTopKSampler final: public speculative::FastTopKSampler {
public:
    explicit RecordingFastTopKSampler(std::shared_ptr<SamplingRecord> record): record_(std::move(record)) {}

    speculative::FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1) override {
        ++record_->fast_topk_sampler_calls;
        return speculative::FastTopKSampler::forward(logits, top_k);
    }

private:
    std::shared_ptr<SamplingRecord> record_;
};

class RecordingSpeculativeSampler final: public speculative::SpeculativeSampler {
public:
    RecordingSpeculativeSampler(size_t propose_step, std::shared_ptr<SamplingRecord> record):
        SpeculativeSampler(propose_step), record_(std::move(record)) {}

    speculative::SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                                   SamplerOutput&                      draft_sampler_output,
                                                   SamplerOutput&                      target_sampler_output) override {
        ++record_->speculative_sampler_calls;
        return speculative::SpeculativeSampler::forward(streams, draft_sampler_output, target_sampler_output);
    }

private:
    std::shared_ptr<SamplingRecord> record_;
};

class WarmUpShapeModel: public ModelBase {
public:
    WarmUpShapeModel(size_t vocab_size, size_t hidden_size, std::shared_ptr<ForwardRecord> record):
        vocab_size_(vocab_size), hidden_size_(hidden_size), record_(std::move(record)) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        ++record_->calls;
        const int64_t logits_rows = inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.numel() : 1;
        GptModelOutputs outputs;
        outputs.logits = torch::zeros(
            {logits_rows, static_cast<int64_t>(vocab_size_)},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        outputs.all_hidden_states = torch::zeros(
            {inputs.combo_tokens.numel(), static_cast<int64_t>(hidden_size_)},
            torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
        return outputs;
    }

private:
    size_t                         vocab_size_;
    size_t                         hidden_size_;
    std::shared_ptr<ForwardRecord> record_;
};

class ProcessingWarmUpEngine final: public NormalEngine {
public:
    using NormalEngine::NormalEngine;

    absl::StatusOr<std::vector<GenerateStreamPtr>> preRunWithResourceContext(
        const std::vector<std::shared_ptr<GenerateInput>>& inputs,
        preRunMode                                          mode,
        const ResourceContext&                              resource_context) override {
        auto* mtp_executor = dynamic_cast<MtpExecutor*>(executor_.get());
        RTP_LLM_CHECK_WITH_INFO(mtp_executor != nullptr, "warmup requires an MTP executor");
        modes.push_back(mode);
        context_roles.push_back(resource_context.role_type);
        executor_roles.push_back(mtp_executor->role_type_);
        mtp_executor->setTargetModel(
            std::make_unique<WarmUpShapeModel>(model_config_.vocab_size, model_config_.hidden_size, target_record));
        mtp_executor->setDraftModel(
            std::make_unique<WarmUpShapeModel>(model_config_.vocab_size, model_config_.hidden_size, draft_record));
        mtp_executor->setSampler(std::make_unique<RecordingSampler>(sampling_record));
        mtp_executor->setFastTopKSampler(std::make_unique<RecordingFastTopKSampler>(sampling_record));
        mtp_executor->setSpeculativeSampler(
            std::make_unique<RecordingSpeculativeSampler>(mtp_executor->propose_step_, sampling_record));
        return NormalEngine::preRunWithResourceContext(inputs, mode, resource_context);
    }

    std::shared_ptr<ForwardRecord>  target_record   = std::make_shared<ForwardRecord>();
    std::shared_ptr<ForwardRecord>  draft_record    = std::make_shared<ForwardRecord>();
    std::shared_ptr<SamplingRecord> sampling_record = std::make_shared<SamplingRecord>();
    std::vector<preRunMode>         modes;
    std::vector<RoleType>           context_roles;
    std::vector<RoleType>           executor_roles;
};

class NormalEngineWarmUpTest: public DeviceTestBase {
protected:
    void TearDown() override {
        NormalExecutor::test_model_factory = nullptr;
        DeviceTestBase::TearDown();
    }

    EngineInitParams makeParams(RoleType role) {
        CustomConfig  config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);

        params.runtime_config.warm_up                                      = false;
        params.runtime_config.max_generate_batch_size                      = kBatchSize;
        params.runtime_config.fifo_scheduler_config.max_context_batch_size = kBatchSize;
        params.kv_cache_config.test_block_num                              = 2;
        params.pd_sep_config.role_type                                     = role;
        params.sp_config.type                                              = SP_TYPE_MTP;
        params.sp_config.gen_num_per_cycle                                 = kProposalStep;
        params.model_config_.gen_num_per_cycle                             = kProposalStep;
        return params;
    }

    std::unique_ptr<ProposeModelEngineInitParams> makeProposal(const EngineInitParams& params) {
        auto model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        model_params->push_back(std::make_unique<EngineInitParams>(params));
        return std::make_unique<ProposeModelEngineInitParams>(
            SP_TYPE_MTP, static_cast<size_t>(params.sp_config.gen_num_per_cycle), std::move(model_params));
    }

    std::unique_ptr<ProcessingWarmUpEngine> makeStoppedEngine(const EngineInitParams& params) {
        const auto vocab_size = params.model_config_.vocab_size;
        NormalExecutor::test_model_factory = [vocab_size](const GptModelInitParams&) {
            return std::make_unique<MockModel>(vocab_size);
        };
        auto engine = std::make_unique<ProcessingWarmUpEngine>(params, nullptr);
        EXPECT_TRUE(engine->stop().ok());
        engine->executor_.reset();
        engine->propose_params_ = makeProposal(params);
        return engine;
    }
};

TEST_F(NormalEngineWarmUpTest, mtpFusionWarmUpUsesTargetProposalInputVocabIntersection) {
    auto params                           = makeParams(RoleType::PDFUSION);
    params.model_config_.vocab_size       = 4;
    params.model_config_.input_vocab_size = 4;
    params.model_config_.embedding_size   = 4;
    auto engine                           = makeStoppedEngine(params);
    engine->propose_params_->mtp_model_params_->front()->model_config_.input_vocab_size = 1;

    engine->warmUp(params);

    EXPECT_EQ(engine->target_record->calls, 2);
    EXPECT_EQ(engine->draft_record->calls, kProposalStep + 1);
}

TEST_F(NormalEngineWarmUpTest, mtpFusionWarmUpCoversPrefillAndDecode) {
    auto params = makeParams(RoleType::PDFUSION);
    auto engine = makeStoppedEngine(params);

    engine->warmUp(params);

    EXPECT_EQ(engine->modes,
              (std::vector<preRunMode>{preRunMode::prefill_warm_up, preRunMode::decode_warm_up}));
    EXPECT_EQ(engine->context_roles, (std::vector<RoleType>{RoleType::PREFILL, RoleType::DECODE}));
    EXPECT_EQ(engine->executor_roles, (std::vector<RoleType>{RoleType::PREFILL, RoleType::DECODE}));
    EXPECT_EQ(engine->target_record->calls, 2);
    EXPECT_EQ(engine->draft_record->calls, kProposalStep + 1);
    EXPECT_EQ(engine->sampling_record->target_sampler_calls, 2);
    EXPECT_EQ(engine->sampling_record->fast_topk_sampler_calls, kProposalStep + 1);
    EXPECT_EQ(engine->sampling_record->speculative_sampler_calls, 1);
}

}  // namespace
}  // namespace rtp_llm
