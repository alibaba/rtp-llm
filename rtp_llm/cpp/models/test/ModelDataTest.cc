
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/Sampler.h"

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder() = default;

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    SamplerInputs allocate(Config config) {
        SamplerInputs sampler_inputs;
        sampler_inputs.step           = config.max_length;
        sampler_inputs.batch_size     = config.batch_size;
        sampler_inputs.batch_size_out = config.batch_size;
        auto bs                       = (int64_t)config.batch_size;
        sampler_inputs.logits         = torch::empty(
            {bs, (int64_t)config.vocab_size},
            torch::TensorOptions().dtype(rtp_llm::dataTypeToTorchType(config.logits_type)).device(torch::kCUDA));
        sampler_inputs.sequence_lengths   = torch::empty({bs}, torch::kInt32);
        sampler_inputs.input_lengths      = torch::empty({bs}, torch::kInt32);
        sampler_inputs.num_beams_in       = torch::empty({bs}, torch::kLong);
        sampler_inputs.num_beams_out      = torch::empty({bs}, torch::kLong);
        sampler_inputs.top_k              = torch::empty({bs}, torch::kInt32);
        sampler_inputs.top_p              = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.temperature        = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.repetition_penalty = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.cum_log_probs      = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.token_ids          = torch::empty({bs, (int64_t)(sampler_inputs.step + 1)}, torch::kInt32);
        return sampler_inputs;
    };

    void setSequenceLengths(SamplerInputs& sampler_inputs, std::vector<int>& sequence_lengths) {
        RTP_LLM_CHECK(sequence_lengths.size() == sampler_inputs.batch_size);
        sampler_inputs.sequence_lengths = torch::tensor(sequence_lengths, torch::kInt32);
    };
};

class ModelDataTest: public DeviceTestBase {};

TEST_F(ModelDataTest, testConstruct) {
    SamplerDataBuilder builder;
    SamplerInputs      sampler_inputs   = builder.allocate({4, 1024, 1024});
    std::vector<int>   sequence_lengths = {1, 2, 3, 4};
    builder.setSequenceLengths(sampler_inputs, sequence_lengths);
    auto sl = sampler_inputs.sequence_lengths;
    EXPECT_EQ(std::vector<int>(sl.data_ptr<int>(), sl.data_ptr<int>() + sl.numel()), std::vector<int>({1, 2, 3, 4}));
}

TEST_F(ModelDataTest, testTensorHolderReleasesOnThirdRound) {
    TensorHolder holder;
    auto         t0 = torch::empty({1}, torch::kFloat32);
    auto         t1 = torch::empty({1}, torch::kFloat32);
    auto         t2 = torch::empty({1}, torch::kFloat32);

    holder.hold(t0);
    holder.release();
    ASSERT_EQ(holder.clear_tensors.size(), 1);
    EXPECT_EQ(holder.clear_tensors.front().front().data_ptr(), t0.data_ptr());

    holder.hold(t1);
    holder.release();
    ASSERT_EQ(holder.clear_tensors.size(), 2);
    EXPECT_EQ(holder.clear_tensors.front().front().data_ptr(), t0.data_ptr());

    holder.hold(t2);
    holder.release();
    ASSERT_EQ(holder.clear_tensors.size(), 2);
    EXPECT_EQ(holder.clear_tensors.front().front().data_ptr(), t1.data_ptr());
}

TEST_F(ModelDataTest, PrefillHostGeometryTracksTensorIdentityAndMutation) {
    GptModelInputs inputs;
    inputs.input_lengths = torch::tensor({80, 81, 419}, torch::kInt32);
    inputs.setPrefillHostCuSeqlens(torch::tensor({0, 80, 161, 580}, torch::kInt32));
    ASSERT_TRUE(inputs.hasPrefillHostCuSeqlens());
    auto copy = inputs;
    EXPECT_TRUE(copy.hasPrefillHostCuSeqlens());

    copy.input_lengths = inputs.input_lengths.clone();
    EXPECT_FALSE(copy.hasPrefillHostCuSeqlens());
    EXPECT_TRUE(inputs.hasPrefillHostCuSeqlens());
    copy               = inputs;
    copy.input_lengths = inputs.input_lengths.view({3});
    EXPECT_FALSE(copy.hasPrefillHostCuSeqlens());
    copy = inputs;
    inputs.input_lengths.add_(1);
    EXPECT_FALSE(inputs.hasPrefillHostCuSeqlens());
    EXPECT_FALSE(copy.hasPrefillHostCuSeqlens());

    inputs.setPrefillHostCuSeqlens(torch::tensor({0, 81, 163, 583}, torch::kInt32));
    EXPECT_TRUE(inputs.hasPrefillHostCuSeqlens());
    inputs.setPrefillHostCuSeqlens(torch::Tensor{});
    EXPECT_FALSE(inputs.hasPrefillHostCuSeqlens());
}

TEST_F(ModelDataTest, PrefillHostGeometryRejectsInvalidOrUnversionedMirrors) {
    GptModelInputs inputs;
    inputs.input_lengths = torch::tensor({80, 81}, torch::kInt32);
    inputs.setPrefillHostCuSeqlens(torch::tensor({0, 80}, torch::kInt32));
    EXPECT_FALSE(inputs.hasPrefillHostCuSeqlens());
    inputs.setPrefillHostCuSeqlens(torch::tensor({0, 80, 161}, torch::kInt64));
    EXPECT_FALSE(inputs.hasPrefillHostCuSeqlens());
    inputs.setPrefillHostCuSeqlens(torch::tensor({0, 80, 161}, torch::kInt32).cuda());
    EXPECT_FALSE(inputs.hasPrefillHostCuSeqlens());
    {
        c10::InferenceMode guard(true);
        inputs.input_lengths = torch::tensor({80, 81}, torch::kInt32);
        inputs.setPrefillHostCuSeqlens(torch::tensor({0, 80, 161}, torch::kInt32));
        EXPECT_FALSE(inputs.hasPrefillHostCuSeqlens());
    }
}

TEST_F(ModelDataTest, PrefillHostPrefixTracksTensorIdentityAndMutation) {
    GptModelInputs inputs;
    inputs.prefix_lengths = torch::tensor({80, 81, 419}, torch::kInt32);
    inputs.setPrefillHostPrefixLengths(torch::tensor({0, 128, 8192}, torch::kInt32));
    ASSERT_TRUE(inputs.hasPrefillHostPrefixLengths());
    auto copy = inputs;
    EXPECT_TRUE(copy.hasPrefillHostPrefixLengths());

    copy.prefix_lengths = inputs.prefix_lengths.clone();
    EXPECT_FALSE(copy.hasPrefillHostPrefixLengths());
    EXPECT_TRUE(inputs.hasPrefillHostPrefixLengths());
    copy                = inputs;
    copy.prefix_lengths = inputs.prefix_lengths.view({3});
    EXPECT_FALSE(copy.hasPrefillHostPrefixLengths());
    copy = inputs;
    inputs.prefix_lengths.add_(1);
    EXPECT_FALSE(inputs.hasPrefillHostPrefixLengths());
    EXPECT_FALSE(copy.hasPrefillHostPrefixLengths());

    inputs.setPrefillHostPrefixLengths(torch::tensor({1, 129, 8193}, torch::kInt32));
    EXPECT_TRUE(inputs.hasPrefillHostPrefixLengths());
    inputs.setPrefillHostPrefixLengths(torch::Tensor{});
    EXPECT_FALSE(inputs.hasPrefillHostPrefixLengths());
}

TEST_F(ModelDataTest, PrefillHostPrefixRejectsInvalidOrUnversionedMirrors) {
    GptModelInputs inputs;
    inputs.prefix_lengths = torch::tensor({80, 81}, torch::kInt32);
    inputs.setPrefillHostPrefixLengths(torch::tensor({0}, torch::kInt32));
    EXPECT_FALSE(inputs.hasPrefillHostPrefixLengths());
    inputs.setPrefillHostPrefixLengths(torch::tensor({0, 128}, torch::kInt64));
    EXPECT_FALSE(inputs.hasPrefillHostPrefixLengths());
    inputs.setPrefillHostPrefixLengths(torch::tensor({0, 128}, torch::kInt32).cuda());
    EXPECT_FALSE(inputs.hasPrefillHostPrefixLengths());
    {
        c10::InferenceMode guard(true);
        inputs.prefix_lengths = torch::tensor({80, 81}, torch::kInt32);
        inputs.setPrefillHostPrefixLengths(torch::tensor({0, 128}, torch::kInt32));
        EXPECT_FALSE(inputs.hasPrefillHostPrefixLengths());
    }
}

}  // namespace rtp_llm
