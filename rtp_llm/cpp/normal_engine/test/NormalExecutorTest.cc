#include <queue>
#include <memory>
#include <stdexcept>
#include <vector>
#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#define private public
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#undef private

namespace rtp_llm {

class FakeModel: public ModelBase {
public:
    FakeModel(size_t vocab_size): vocab_size_(vocab_size) {}

    GptModelOutputs forward(const GptModelInputs&) override {
        if (outputs_.empty()) {
            throw std::runtime_error("[test] FakeModel outputs queue is empty");
        }
        auto out = std::move(outputs_.front());
        outputs_.pop();
        return out;
    }

    void pushOutputs(const std::vector<GptModelOutputs>& outs) {
        for (const auto& o : outs) {
            outputs_.push(o);
        }
    }

private:
    size_t                      vocab_size_;
    std::queue<GptModelOutputs> outputs_;
};

class FakeSampler: public Sampler {
public:
    explicit FakeSampler(const SamplerInitParams& params): Sampler(params) {}

    SamplerOutput forward(const SamplerInputs&) override {
        if (outputs_.empty()) {
            throw std::runtime_error("[test] FakeSampler outputs queue is empty");
        }
        auto out = std::move(outputs_.front());
        outputs_.pop();
        return out;
    }

    void pushOutputs(const std::vector<SamplerOutput>& outs) {
        for (const auto& o : outs) {
            outputs_.push(o);
        }
    }

private:
    std::queue<SamplerOutput> outputs_;
};

class FakeBatchProcessor: public NormalBatchStreamProcessor {
public:
    FakeBatchProcessor(const ModelConfig&                 model_config,
                       const PDSepConfig&                 pd_sep_config,
                       const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                       const CacheConfig&                 cache_config):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false) {}

    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const override {
        const size_t   batch_size = stream_groups.size();
        GptModelInputs inputs;
        inputs.combo_tokens      = torch::zeros({(int64_t)batch_size}, torch::kInt32);
        inputs.input_lengths     = torch::ones({(int64_t)batch_size}, torch::kInt32);
        inputs.sequence_lengths  = torch::ones({(int64_t)batch_size}, torch::kInt32);
        inputs.prefix_lengths    = torch::zeros({(int64_t)batch_size}, torch::kInt32);
        inputs.lm_output_indexes = torch::zeros({(int64_t)batch_size}, torch::kInt32);
        return inputs;
    }

    absl::StatusOr<SamplerInputs>
    gatherSamplerInput(const StreamGroups&, const GptModelInputs&, const GptModelOutputs& model_output) const override {
        SamplerInputs inputs;
        inputs.logits = model_output.logits;
        return inputs;
    }

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const override {
        if (merge_outputs.model_output.nan_flag.defined()) {
            checkNanFlagAndSetFailed(stream_groups, merge_outputs.model_output.nan_flag);
        }

        for (auto& stream : stream_groups.allStreams()) {
            stream->setStop(ErrorCode::UNKNOWN_ERROR, "second stop should not overwrite first");
        }

        return absl::OkStatus();
    }
};

class NormalExecutorTest: public DeviceTestBase {
public:
    std::unique_ptr<NormalExecutor> createExecutorAndInjectFakes(ModelConfig                         model_config,
                                                                 RuntimeConfig                       runtime_config,
                                                                 std::unique_ptr<FakeModel>          fake_model,
                                                                 std::unique_ptr<FakeSampler>        fake_sampler,
                                                                 std::unique_ptr<FakeBatchProcessor> fake_processor) {
        CustomConfig  custom_config;
        KVCacheConfig kv_cache_config;

        auto params = createEngineInitParams(custom_config, model_config, runtime_config, kv_cache_config);

        auto cache_config  = test::makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                           /*block_num=*/10,
                                                           /*tokens_per_block=*/2,
                                                           rtp_llm::TYPE_INT8,
                                                           /*local_head_num_kv=*/128,
                                                           /*size_per_head=*/256);
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        EXPECT_TRUE(cache_manager->init());

        auto executor = std::make_unique<NormalExecutor>(params, cache_manager, false, false);
        executor->setModel(std::move(fake_model));
        executor->sampler_ = std::move(fake_sampler);
        executor->setBatchProcessor(std::move(fake_processor));
        return executor;
    }

    GenerateStreamPtr createSimpleContextStream(const ModelConfig&     model_config,
                                                const RuntimeConfig&   runtime_config,
                                                const ResourceContext& resource_context) {
        auto query                             = std::make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({0, 1}, torch::kInt32);
        query->generate_config                 = std::make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }
};

TEST_F(NormalExecutorTest, testNanFlagStopsAndStopReasonNotOverwritten) {
    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    KVCacheConfig   kv_cache_config;
    CustomConfig    custom_config;
    ResourceContext resource_context;

    auto params = createEngineInitParams(custom_config, model_config, runtime_config, kv_cache_config);

    auto fake_model     = std::make_unique<FakeModel>(model_config.vocab_size);
    auto fake_sampler   = std::make_unique<FakeSampler>(SamplerInitParams{});
    auto fake_processor = std::make_unique<FakeBatchProcessor>(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, CacheConfig());

    GptModelOutputs model_out;
    model_out.logits   = torch::tensor({{0.1f, 0.2f}});
    model_out.nan_flag = torch::tensor({1.0f});
    fake_model->pushOutputs({model_out});

    SamplerOutput sampler_out;
    sampler_out.token_ids = torch::zeros({1, 1}, torch::kInt32);
    fake_sampler->pushOutputs({sampler_out});

    auto executor = createExecutorAndInjectFakes(params.model_config_,
                                                 params.runtime_config,
                                                 std::move(fake_model),
                                                 std::move(fake_sampler),
                                                 std::move(fake_processor));

    auto stream = createSimpleContextStream(params.model_config_, params.runtime_config, resource_context);

    auto status = executor->process({stream});
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(stream->stopped());
    const auto reason = stream->stopReason();
    EXPECT_NE(reason.find("NaN detected"), std::string::npos);
    EXPECT_EQ(reason.find("second stop should not overwrite first"), std::string::npos);
}

}  // namespace rtp_llm
