#include <queue>
#include <memory>
#include <stdexcept>
#include <vector>
#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

#define private public
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#undef private

namespace rtp_llm {

class FakeModel: public GptModel {
public:
    FakeModel(const GptModelInitParams& params): GptModel(params) {}

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
        // Must populate these buffers: GptModelInputs::debugString() deref them unconditionally.
        const size_t   batch_size = stream_groups.size();
        GptModelInputs inputs;
        inputs.combo_tokens      = vector2Buffer<int32_t>(std::vector<int32_t>(batch_size, 0));
        inputs.input_lengths     = vector2Buffer<int32_t>(std::vector<int32_t>(batch_size, 1));
        inputs.sequence_lengths  = vector2Buffer<int32_t>(std::vector<int32_t>(batch_size, 1));
        inputs.prefix_lengths    = vector2Buffer<int32_t>(std::vector<int32_t>(batch_size, 0));
        inputs.lm_output_indexes = vector2Buffer<int32_t>(std::vector<int32_t>(batch_size, 0));
        return inputs;
    }

    absl::StatusOr<SamplerInputs>
    gatherSamplerInput(const StreamGroups&, const GptModelInputs&, const GptModelOutputs& model_output) const override {
        SamplerInputs inputs;
        inputs.logits = model_output.logits;
        return inputs;
    }

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const override {
        if (merge_outputs.model_output.nan_flag) {
            checkNanFlagAndSetFailed(stream_groups, merge_outputs.model_output.nan_flag);
        }

        // Also trigger a second stop attempt, to verify stop reason is not overwritten.
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

        auto params = createEngineInitParams(device_, custom_config, model_config, runtime_config, kv_cache_config);

        auto cache_config  = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                           /*block_num=*/10,
                                                           /*tokens_per_block=*/2,
                                                           rtp_llm::TYPE_INT8,
                                                           /*local_head_num_kv=*/128,
                                                           /*size_per_head=*/256);
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
        EXPECT_TRUE(cache_manager->init());

        auto executor = std::make_unique<NormalExecutor>(params, cache_manager, device_, nullptr, false);
        executor->setGptModel(std::move(fake_model));
        executor->sampler_ = std::move(fake_sampler);
        executor->setBatchProcessor(std::move(fake_processor));
        return executor;
    }

    GenerateStreamPtr createSimpleContextStream(const ModelConfig&     model_config,
                                                const RuntimeConfig&   runtime_config,
                                                const ResourceContext& resource_context) {
        auto query                             = std::make_shared<GenerateInput>();
        query->input_ids                       = createBuffer<int32_t>({2}, {0, 1}, AllocationType::HOST);
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

    auto params = createEngineInitParams(device_, custom_config, model_config, runtime_config, kv_cache_config);

    const auto desc = Executor::genModelDescription(
        params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config);
    GptModelInitParams fake_model_params(
        {device_, params.gpt_weights, desc, std::nullopt, std::nullopt, params.model_id});

    auto fake_model     = std::make_unique<FakeModel>(fake_model_params);
    auto fake_sampler   = std::make_unique<FakeSampler>(SamplerInitParams{device_});
    auto fake_processor = std::make_unique<FakeBatchProcessor>(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, CacheConfig());

    // Model output with nan_flag set for one stream (batch_size = 1).
    GptModelOutputs model_out;
    model_out.logits   = createBuffer<float>({1, 2}, {0.1f, 0.2f}, AllocationType::HOST);
    model_out.nan_flag = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    fake_model->pushOutputs({model_out});

    SamplerOutput sampler_out;
    sampler_out.token_ids = createBuffer<int32_t>({1, 1}, {0}, AllocationType::HOST);
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
