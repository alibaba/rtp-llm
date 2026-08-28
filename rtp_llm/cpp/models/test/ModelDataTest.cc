
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

TEST_F(ModelDataTest, testShapeHintsCarryExactPackedBroadcastPlacement) {
    GptModelInputs inputs;
    inputs.combo_tokens             = torch::zeros({8}, torch::kInt32).cuda();
    inputs.input_lengths            = torch::zeros({1}, torch::kInt32).cuda();
    inputs.sequence_lengths         = torch::zeros({1}, torch::kInt32).cuda();
    inputs.prefix_lengths           = torch::zeros({1}, torch::kInt32).cuda();
    inputs.kv_cache_kernel_block_id = torch::zeros({4, 1, 93}, torch::kInt32).cuda();
    inputs.kv_cache_block_id        = torch::zeros({4, 1, 93}, torch::kInt32);
    inputs.kv_cache_group_types     = torch::zeros({4}, torch::kInt32);
    inputs.kv_cache_update_mapping  = torch::zeros({2, 3}, torch::kInt32).cuda();
    inputs.request_id               = torch::zeros({1}, torch::kInt64);
    inputs.request_pd_separation    = torch::zeros({1}, torch::kBool);
    inputs.lm_output_indexes        = torch::zeros({7}, torch::kInt32).cuda();
    // This is the exact mixed-device DSpARK TP2 case that previously made
    // rank 0 send 1504 CPU bytes while rank 1 waited for 1600.
    inputs.combo_position_ids = torch::zeros({24}, torch::kInt32).cuda();

    const auto hints     = getModelInputShapeHints(inputs);
    const auto map       = static_cast<uint32_t>(hints[GptModelInputIndex::tensorDeviceMap]);
    auto       is_device = [&](GptModelInputDeviceBit bit) { return (map & bit) != 0; };

    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitComboTokens));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitInputLengths));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitSequenceLengths));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitPrefixLengths));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitKernelBlockId));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitBlockId));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitCacheGroupTypes));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitCacheUpdateMapping));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitRequestId));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitRequestPdSeparation));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitLmOutputIndexes));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitComboPositionIds));
}

TEST_F(ModelDataTest, testShapeHintsCarryTpControlPlaneAndCacheGeometry) {
    GptModelInputs inputs;
    inputs.need_all_logits           = true;
    inputs.need_all_hidden_states    = true;
    inputs.need_moe_gating           = true;
    inputs.warmup                    = true;
    inputs.skip_run                  = true;
    inputs.is_fake_stream            = true;
    inputs.is_target_verify          = true;
    inputs.pd_separation             = true;
    inputs.decode_entrance           = true;
    inputs.use_opaque_kv_cache_store = true;
    inputs.kv_block_stride_bytes     = 4096;
    inputs.kv_scale_stride_bytes     = 256;
    inputs.seq_size_per_block        = 64;
    inputs.kernel_seq_size_per_block = 16;

    const auto hints = getModelInputShapeHints(inputs);
    const auto flags = static_cast<uint32_t>(hints[GptModelInputIndex::modelControlFlags]);
    auto       has   = [&](GptModelInputControlFlag flag) { return (flags & flag) != 0; };

    EXPECT_TRUE(has(GptModelInputControlFlag::kControlNeedAllLogits));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlNeedAllHiddenStates));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlNeedMoeGating));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlWarmup));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlSkipRun));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlFakeStream));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlTargetVerify));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlPdSeparation));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlDecodeEntrance));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlOpaqueKvCacheStore));
    EXPECT_EQ(hints[GptModelInputIndex::kvBlockStrideBytes], 4096);
    EXPECT_EQ(hints[GptModelInputIndex::kvScaleStrideBytes], 256);
    EXPECT_EQ(hints[GptModelInputIndex::seqSizePerBlock], 64);
    EXPECT_EQ(hints[GptModelInputIndex::kernelSeqSizePerBlock], 16);
}

}  // namespace rtp_llm
