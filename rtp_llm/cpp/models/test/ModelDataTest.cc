
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

TEST_F(ModelDataTest, testScoringRowHintsEncodeMetadataAndChunkRows) {
    GptModelInputs inputs;
    auto           hints = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::allLogitsRowMeta], 0);
    // No producer override: the hint carries the env/default value every rank agrees on.
    EXPECT_EQ(hints[GptModelInputIndex::scoringChunkRows], kDefaultScoringPartialChunkRows);

    inputs.all_logits_row_indexes     = torch::empty({0}, torch::kLong);
    inputs.all_logits_row_labels      = torch::empty({0}, torch::kInt32);
    inputs.scoring_partial_chunk_rows = 3;
    hints                             = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::allLogitsRowMeta], 7);  // defined + defined + equal
    EXPECT_EQ(hints[GptModelInputIndex::scoringChunkRows], 3);

    // Indexes without labels must encode meta=2 so the consistency check can reject it
    // identically on every rank.
    inputs.all_logits_row_labels = torch::Tensor();
    hints                        = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::allLogitsRowMeta], 2);
}

TEST_F(ModelDataTest, testScoringRowHintsEncodeLabelLengthMismatch) {
    // A defined row set whose label count differs from its index count encodes meta=3
    // (both defined, lengths unequal); the broadcast check then rejects it on every rank
    // before packing — a rank-local tensor comparison could not give that guarantee.
    GptModelInputs inputs;
    inputs.all_logits_row_indexes = torch::empty({3}, torch::kLong);
    inputs.all_logits_row_labels  = torch::empty({2}, torch::kInt32);
    auto hints                    = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::allLogitsRowMeta], 3);
    EXPECT_THROW(checkScoringRowMetaConsistent(/*need_all_logits=*/true, hints[GptModelInputIndex::allLogitsRowMeta]),
                 RTPException);
}

TEST_F(ModelDataTest, testScoringRowMetaConsistencyCheck) {
    // The predicate every tpSyncModelInputs rank evaluates on the broadcast hints — a
    // rejection is collective-uniform because the checked value is rank-identical.
    EXPECT_NO_THROW(checkScoringRowMetaConsistent(/*need_all_logits=*/false, 0));
    EXPECT_NO_THROW(checkScoringRowMetaConsistent(/*need_all_logits=*/true, 7));
    EXPECT_THROW(checkScoringRowMetaConsistent(true, 0), RTPException);    // flag without rows
    EXPECT_THROW(checkScoringRowMetaConsistent(false, 7), RTPException);   // rows without flag
    EXPECT_THROW(checkScoringRowMetaConsistent(false, 2), RTPException);   // rows without flag
    EXPECT_THROW(checkScoringRowMetaConsistent(true, 2), RTPException);    // indexes w/o labels
    EXPECT_THROW(checkScoringRowMetaConsistent(true, 1), RTPException);    // labels w/o indexes
    EXPECT_THROW(checkScoringRowMetaConsistent(true, 3), RTPException);    // unequal lengths
    EXPECT_THROW(checkScoringRowMetaConsistent(true, 6), RTPException);    // indexes + lengths w/o labels
    EXPECT_THROW(checkScoringRowMetaConsistent(false, -1), RTPException);  // corrupt hint
    EXPECT_THROW(checkScoringRowMetaConsistent(false, 8), RTPException);
}

}  // namespace rtp_llm
