#include <cstdint>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

namespace {

template<typename T>
std::vector<T> tensorToVector(const torch::Tensor& tensor) {
    const auto cpu = tensor.cpu().contiguous();
    return std::vector<T>(cpu.data_ptr<T>(), cpu.data_ptr<T>() + cpu.numel());
}

torch::Tensor intTensor(std::vector<int32_t> values) {
    return torch::tensor(std::move(values), torch::kInt32);
}

}  // namespace

class PPBatchStreamProcessorTest: public DeviceTestBase {
protected:
    static ModelConfig makeModelConfig() {
        ModelConfig model_config;
        model_config.max_seq_len      = 32;
        model_config.vocab_size       = 64;
        model_config.input_vocab_size = 64;
        model_config.num_layers       = 1;
        return model_config;
    }

    static GenerateStreamPtr makeStream(const ResourceContext& resource_context,
                                        const ModelConfig&     model_config,
                                        int64_t                request_id,
                                        std::vector<int32_t>   input_ids,
                                        int32_t                num_return_sequences) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->request_id                            = request_id;
        input->input_ids                             = intTensor(std::move(input_ids));
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->num_return_sequences = num_return_sequences;

        RuntimeConfig runtime_config;
        auto          stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }
};

TEST_F(PPBatchStreamProcessorTest, MixedReturnSequencesPlanRoundTripAndDispatch) {
    ResourceContext resource_context;
    const auto      model_config = makeModelConfig();

    auto multi_stream                           = makeStream(resource_context, model_config, 101, {1, 2}, 2);
    auto single_stream                          = makeStream(resource_context, model_config, 202, {3, 4, 5}, 1);
    auto multi_config                           = multi_stream->generateConfig();
    multi_config->do_sample                     = true;
    multi_config->top_k                         = 7;
    multi_config->combo_token_size              = 2;
    multi_config->banned_combo_token_ids        = {{20, 21}};
    multi_config->end_think_token_ids           = {30, 31};
    multi_config->enable_cross_sequence_ban     = true;
    multi_config->cross_seq_diverge_start_combo = 3;
    single_stream->generateConfig()->do_sample  = true;
    single_stream->generateConfig()->top_k      = 3;

    std::list<GenerateStreamPtr> streams{multi_stream, single_stream};
    StreamGroups                 stream_groups(streams);
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    PPBatchStreamProcessor processor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    PPExecutionPlan plan;
    TensorHolder    holder;
    auto            model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    plan.model_input   = std::move(model_input.value());
    plan.sampling_plan = processor.gatherSamplingPlan(stream_groups);
    plan.output_config = processor.gatherOutputConfig(stream_groups);

    const auto& sampling_plan = plan.sampling_plan;
    EXPECT_EQ(sampling_plan.num_return_sequences, (std::vector<int32_t>{2, 1}));
    EXPECT_EQ(tensorToVector<int64_t>(sampling_plan.request_ids), (std::vector<int64_t>{101, 202}));
    EXPECT_EQ(sampling_plan.token_ids.sizes().vec(), (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.token_ids[0].narrow(0, 0, 2)), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.token_ids[1].narrow(0, 0, 2)), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.token_ids[2].narrow(0, 0, 3)), (std::vector<int32_t>{3, 4, 5}));
    EXPECT_EQ(tensorToVector<int32_t>(sampling_plan.top_k), (std::vector<int32_t>{7, 7, 3}));
    ASSERT_EQ(sampling_plan.logits_processor_configs.size(), 2);
    const auto& processor_config = sampling_plan.logits_processor_configs[0];
    EXPECT_EQ(processor_config.combo_token_size, 2);
    EXPECT_EQ(processor_config.banned_combo_token_ids, (std::vector<std::vector<int>>{{20, 21}}));
    EXPECT_EQ(processor_config.end_think_token_ids, (std::vector<int>{30, 31}));
    EXPECT_TRUE(processor_config.enable_cross_sequence_ban);
    EXPECT_EQ(processor_config.cross_seq_diverge_start_combo, 3);

    auto round_trip_plan = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    EXPECT_EQ(round_trip_plan.sampling_plan.num_return_sequences, (std::vector<int32_t>{2, 1}));
    EXPECT_EQ(tensorToVector<int64_t>(round_trip_plan.sampling_plan.request_ids), (std::vector<int64_t>{101, 202}));
    ASSERT_EQ(round_trip_plan.sampling_plan.logits_processor_configs.size(), 2);
    EXPECT_TRUE(round_trip_plan.sampling_plan.logits_processor_configs[0].enable_cross_sequence_ban);
    EXPECT_EQ(round_trip_plan.sampling_plan.logits_processor_configs[0].cross_seq_diverge_start_combo, 3);

    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    sampler_output.token_ids = torch::tensor({10, 11, 12}, torch::kInt32).reshape({3, 1});
    sampler_output.success   = torch::tensor({true, true, true}, torch::kBool);
    auto result              = processor.makeExecutionResult(plan, model_output, sampler_output);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->processor_errors.size(), 3);

    auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result.value()));
    ASSERT_TRUE(processor.dispatchExecutionResult(stream_groups, round_trip_result).ok());
    EXPECT_EQ(multi_stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 10}));
    EXPECT_EQ(multi_stream->completeTokenIdsVec(1), (std::vector<int>{1, 2, 11}));
    EXPECT_EQ(single_stream->completeTokenIdsVec(0), (std::vector<int>{3, 4, 5, 12}));
}

}  // namespace rtp_llm
