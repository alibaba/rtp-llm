
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"

#include <type_traits>

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

namespace {

const std::array<torch::Tensor GptModelInputs::*, 3> kV41ExecutionFields{
    &GptModelInputs::v41_request_id, &GptModelInputs::v41_state_ready, &GptModelInputs::v41_is_fake};

GptModelInputs makeV41DecodeInputs() {
    GptModelInputs inputs;
    inputs.combo_tokens         = torch::tensor({101, 202}, torch::kInt32);
    inputs.input_lengths        = torch::tensor({4, 7}, torch::kInt32);
    inputs.sequence_lengths     = torch::tensor({11, 22}, torch::kInt32);
    inputs.prefix_lengths       = torch::empty({0}, torch::kInt32);
    inputs.request_id           = torch::empty({0}, torch::kInt64);
    inputs.v41_token_types      = torch::full({2}, -1, torch::kInt32);
    inputs.v41_token_valid      = torch::ones({2}, torch::kBool);
    inputs.engram_history_ids   = torch::arange(6, torch::kInt32).reshape({2, 3});
    inputs.engram_history_valid = torch::ones({2, 3}, torch::kBool);
    inputs.v41_request_id       = torch::tensor({(int64_t{1} << 40) + 17, (int64_t{1} << 40) + 23}, torch::kInt64);
    inputs.v41_state_ready      = torch::ones({2}, torch::kBool);
    inputs.v41_is_fake          = torch::zeros({2}, torch::kBool);
    return inputs;
}

}  // namespace

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

TEST_F(ModelDataTest, testPrefillCPExecutionFollowsRoleConfig) {
    ParallelismConfig prefill_config;
    prefill_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_TRUE(buildExecProperties(prefill_config, DeviceResourceConfig{}).enable_prefill_cp);

    ParallelismConfig decode_config;
    decode_config.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    EXPECT_FALSE(buildExecProperties(decode_config, DeviceResourceConfig{}).enable_prefill_cp);
}

TEST_F(ModelDataTest, testDSparkLongPrefillShapeHintsStayInt64) {
    GptModelInputs inputs;
    // expand() preserves the logical DSpARK aux shape without allocating the
    // 6 GiB/24 GiB backing storage used by real 256K/1M prefills.
    auto backing              = torch::empty({1, 1}, torch::kBFloat16);
    inputs.last_hidden_states = backing.expand({262144, 12288});

    auto shape_hints = getModelInputShapeHints(inputs);
    static_assert(std::is_same_v<GptModelInputShapeHints::value_type, int64_t>);
    EXPECT_EQ(shape_hints[GptModelInputIndex::mtpHiddenStates], 3221225472LL);
    EXPECT_EQ(shape_hints[GptModelInputIndex::mtpHiddenStatesRows], 262144LL);
    const auto wire_hints = makeModelInputShapeHintsTensor(inputs);
    EXPECT_EQ(wire_hints.scalar_type(), torch::kInt64);
    EXPECT_EQ(wire_hints.data_ptr<int64_t>()[GptModelInputIndex::mtpHiddenStates], 3221225472LL);
    EXPECT_EQ(decodeMtpHiddenStatesShape(shape_hints[GptModelInputIndex::mtpHiddenStates],
                                        shape_hints[GptModelInputIndex::mtpHiddenStatesRows]),
              (std::array<int64_t, 2>{262144, 12288}));

    inputs.last_hidden_states = backing.expand({1048576, 12288});
    shape_hints               = getModelInputShapeHints(inputs);
    EXPECT_EQ(shape_hints[GptModelInputIndex::mtpHiddenStates], 12884901888LL);
    EXPECT_EQ(decodeMtpHiddenStatesShape(shape_hints[GptModelInputIndex::mtpHiddenStates],
                                        shape_hints[GptModelInputIndex::mtpHiddenStatesRows]),
              (std::array<int64_t, 2>{1048576, 12288}));
}

TEST_F(ModelDataTest, testMtpHiddenShapeRejectsInvalidMetadataBeforeAllocation) {
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(-1, 1), RTPException);
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(1, 0), RTPException);
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(5, 2), RTPException);
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(0, 1), RTPException);
}

TEST_F(ModelDataTest, testV41ExecutionShapeHintsIncludeDecodeRequests) {
    auto inputs = makeV41DecodeInputs();
    auto hints  = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::inputLengths], 2);
    EXPECT_EQ(hints[GptModelInputIndex::gptModelRequestLength], 0);
    EXPECT_EQ(hints[GptModelInputIndex::v41InputsPresent], 1);
    EXPECT_EQ(hints[GptModelInputIndex::v41ExecutionPresent], 1);
    const auto wire = makeModelInputShapeHintsTensor(inputs);
    EXPECT_EQ(wire.data_ptr<int64_t>()[GptModelInputIndex::v41ExecutionPresent], 1);

    for (const auto member : kV41ExecutionFields) {
        inputs.*member = torch::Tensor();
    }
    hints = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::v41InputsPresent], 1);
    EXPECT_EQ(hints[GptModelInputIndex::v41ExecutionPresent], 0);
    EXPECT_EQ(getModelInputShapeHints(GptModelInputs{})[GptModelInputIndex::v41ExecutionPresent], 0);
}

TEST_F(ModelDataTest, testV41ExecutionMetadataMustBeCompleteAndCanonical) {
    for (int present_mask = 1; present_mask < 7; ++present_mask) {
        SCOPED_TRACE(present_mask);
        auto inputs = makeV41DecodeInputs();
        for (size_t index = 0; index < kV41ExecutionFields.size(); ++index) {
            if (!(present_mask & (1 << index))) {
                inputs.*kV41ExecutionFields[index] = torch::Tensor();
            }
        }
        EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);
    }
    auto inputs                 = makeV41DecodeInputs();
    inputs.v41_token_types      = torch::Tensor();
    inputs.v41_token_valid      = torch::Tensor();
    inputs.engram_history_ids   = torch::Tensor();
    inputs.engram_history_valid = torch::Tensor();
    EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);
}

TEST_F(ModelDataTest, testV41ExecutionMetadataRejectsWrongShapeTypeAndStorage) {
    for (size_t index = 0; index < kV41ExecutionFields.size(); ++index) {
        SCOPED_TRACE(index);
        const auto member = kV41ExecutionFields[index];
        auto       inputs = makeV41DecodeInputs();
        inputs.*member    = (inputs.*member).to(torch::kInt32);
        EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);

        inputs         = makeV41DecodeInputs();
        inputs.*member = (inputs.*member).narrow(0, 0, 1);
        EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);

        inputs         = makeV41DecodeInputs();
        inputs.*member = (inputs.*member).reshape({1, 2});
        EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);

        inputs         = makeV41DecodeInputs();
        inputs.*member = torch::zeros({4}, (inputs.*member).options()).slice(0, 0, 4, 2);
        EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);

        inputs         = makeV41DecodeInputs();
        inputs.*member = (inputs.*member).to(torch::kCUDA);
        EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);
    }
    auto inputs          = makeV41DecodeInputs();
    inputs.input_lengths = torch::Tensor();
    EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);
    inputs.input_lengths = torch::zeros({1, 2}, torch::kInt32);
    EXPECT_THROW((void)getModelInputShapeHints(inputs), RTPException);
}

TEST_F(ModelDataTest, testEmptyV41ExecutionMetadataRemainsExplicitlyPresent) {
    auto inputs = makeV41DecodeInputs();
    for (const auto member : {&GptModelInputs::combo_tokens,
                              &GptModelInputs::input_lengths,
                              &GptModelInputs::sequence_lengths,
                              &GptModelInputs::v41_token_types,
                              &GptModelInputs::v41_token_valid,
                              &GptModelInputs::engram_history_ids,
                              &GptModelInputs::engram_history_valid,
                              &GptModelInputs::v41_request_id,
                              &GptModelInputs::v41_state_ready,
                              &GptModelInputs::v41_is_fake}) {
        inputs.*member = (inputs.*member).narrow(0, 0, 0);
    }
    const auto hints = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::comboTokens], 0);
    EXPECT_EQ(hints[GptModelInputIndex::inputLengths], 0);
    EXPECT_EQ(hints[GptModelInputIndex::v41InputsPresent], 1);
    EXPECT_EQ(hints[GptModelInputIndex::v41ExecutionPresent], 1);
}

}  // namespace rtp_llm
