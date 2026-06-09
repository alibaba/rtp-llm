
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PrefillCudaGraphEligibility.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"

#include <functional>
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

namespace {

GptModelDescription makePrefillCudaGraphMoeDescription() {
    GptModelDescription description;
    description.data_type   = DataType::TYPE_BF16;
    description.act_qscheme = QScheme::Qfp8PerTokenBlock;
    MoeConfigs model_moe_config;
    model_moe_config.expert_num     = 96;
    model_moe_config.top_k          = 8;
    model_moe_config.use_all_gather = true;
    description.ffn_conf.moe_configs.emplace(model_moe_config);
    return description;
}

MoeConfig makePrefillCudaGraphMoeRuntimeConfig() {
    MoeConfig config;
    config.moe_strategy           = "fp8_per_block_no_dp_masked";
    config.use_all_gather         = true;
    config.use_deepep_moe         = false;
    config.use_deepep_internode   = false;
    config.use_deepep_low_latency = false;
    return config;
}

}  // namespace

TEST_F(ModelDataTest, testPrefillCudaGraphSupportsDenseModel) {
    GptModelDescription description;
    EXPECT_TRUE(supportsPrefillCudaGraphMoe(description, ParallelismConfig{}, MoeConfig{}));
}

TEST_F(ModelDataTest, testPrefillCudaGraphRequiresSingleFullCacheGroup) {
    EXPECT_TRUE(supportsPrefillCudaGraphCacheTopology({CacheGroupType::FULL}));
    EXPECT_FALSE(supportsPrefillCudaGraphCacheTopology({}));
    EXPECT_FALSE(supportsPrefillCudaGraphCacheTopology({CacheGroupType::LINEAR}));
    EXPECT_FALSE(supportsPrefillCudaGraphCacheTopology({CacheGroupType::SWA}));
    EXPECT_FALSE(supportsPrefillCudaGraphCacheTopology({CacheGroupType::FULL, CacheGroupType::LINEAR}));
    EXPECT_FALSE(supportsPrefillCudaGraphCacheTopology({CacheGroupType::FULL, CacheGroupType::SWA}));
}

TEST_F(ModelDataTest, testDefaultPrefillCudaGraphBucketsAreClippedToModelLimit) {
    EXPECT_TRUE(defaultPrefillCudaGraphCaptureSeqLens(0).empty());
    EXPECT_EQ(defaultPrefillCudaGraphCaptureSeqLens(4), (std::vector<int>{4}));
    EXPECT_EQ(defaultPrefillCudaGraphCaptureSeqLens(64), (std::vector<int>{64}));
    EXPECT_EQ(defaultPrefillCudaGraphCaptureSeqLens(160), (std::vector<int>{64, 128, 160}));
    EXPECT_EQ(defaultPrefillCudaGraphCaptureSeqLens(4096), (std::vector<int>{64, 128, 256, 384, 512, 768, 1024}));

    for (int64_t max_seq_len : {7, 64, 159}) {
        const auto buckets = defaultPrefillCudaGraphCaptureSeqLens(max_seq_len);
        ASSERT_FALSE(buckets.empty());
        EXPECT_TRUE(std::is_sorted(buckets.begin(), buckets.end()));
        EXPECT_EQ(buckets.back(), max_seq_len);
    }
}

TEST_F(ModelDataTest, testPrefillCudaGraphSupportsSingleGpuFp8MaskedMoe) {
    EXPECT_TRUE(supportsPrefillCudaGraphMoe(
        makePrefillCudaGraphMoeDescription(), ParallelismConfig{}, makePrefillCudaGraphMoeRuntimeConfig()));
}

TEST_F(ModelDataTest, testPrefillCudaGraphRequiresSingleDeviceParallelism) {
    const auto expect_rejected = [](const std::function<void(ParallelismConfig&)>& mutate) {
        ParallelismConfig config;
        mutate(config);
        EXPECT_FALSE(isSingleDevicePrefillCudaGraphConfig(config));
    };

    EXPECT_TRUE(isSingleDevicePrefillCudaGraphConfig(ParallelismConfig{}));
    expect_rejected([](auto& c) { c.world_size = 2; });
    expect_rejected([](auto& c) { c.tp_size = 2; });
    expect_rejected([](auto& c) { c.dp_size = 2; });
    expect_rejected([](auto& c) { c.ep_size = 2; });
    expect_rejected([](auto& c) { c.pp_size = 2; });
    expect_rejected([](auto& c) { c.ffn_sp_size = 2; });
    expect_rejected([](auto& c) { c.ffn_tp_size = 2; });
    expect_rejected([](auto& c) { c.enable_sp = true; });
    expect_rejected([](auto& c) { c.prefill_cp_config.method = CPRotateMethod::ALL_GATHER; });
    expect_rejected([](auto& c) { c.prefill_cp_config.method = CPRotateMethod::PREFILL_CP; });
    expect_rejected([](auto& c) { c.ffn_disaggregate_config.enable_ffn_disaggregate = true; });
}

TEST_F(ModelDataTest, testPrefillCudaGraphRejectsAutoMoeStrategy) {
    auto config         = makePrefillCudaGraphMoeRuntimeConfig();
    config.moe_strategy = "auto";
    EXPECT_FALSE(supportsPrefillCudaGraphMoe(makePrefillCudaGraphMoeDescription(), ParallelismConfig{}, config));
}

TEST_F(ModelDataTest, testPrefillCudaGraphRejectsNonFp8PerBlockMoe) {
    auto description        = makePrefillCudaGraphMoeDescription();
    description.act_qscheme = QScheme::NoQuantize;
    EXPECT_FALSE(supportsPrefillCudaGraphMoe(description, ParallelismConfig{}, makePrefillCudaGraphMoeRuntimeConfig()));
}

TEST_F(ModelDataTest, testPrefillCudaGraphRejectsGraphUnsafeMoeTransport) {
    auto config                   = makePrefillCudaGraphMoeRuntimeConfig();
    config.use_deepep_low_latency = true;
    EXPECT_FALSE(supportsPrefillCudaGraphMoe(makePrefillCudaGraphMoeDescription(), ParallelismConfig{}, config));

    config                = makePrefillCudaGraphMoeRuntimeConfig();
    config.use_all_gather = false;
    EXPECT_FALSE(supportsPrefillCudaGraphMoe(makePrefillCudaGraphMoeDescription(), ParallelismConfig{}, config));
}

TEST_F(ModelDataTest, testPrefillCudaGraphMoeGateCoversEveryRuntimeConstraint) {
    const auto expect_rejected = [](const std::function<void(MoeConfig&)>& mutate) {
        auto config = makePrefillCudaGraphMoeRuntimeConfig();
        mutate(config);
        EXPECT_FALSE(supportsPrefillCudaGraphMoe(makePrefillCudaGraphMoeDescription(), ParallelismConfig{}, config));
    };

    expect_rejected([](auto& c) { c.use_deepep_moe = true; });
    expect_rejected([](auto& c) { c.use_deepep_internode = true; });
    expect_rejected([](auto& c) { c.use_deepep_low_latency = true; });
    expect_rejected([](auto& c) { c.use_deepep_p2p_low_latency = true; });
    expect_rejected([](auto& c) { c.use_mori_ep = true; });
    expect_rejected([](auto& c) { c.fake_balance_expert = true; });
    expect_rejected([](auto& c) { c.hack_moe_expert = true; });
    expect_rejected([](auto& c) { c.use_all_gather = false; });
}

TEST_F(ModelDataTest, testPrefillCudaGraphMoeGateCoversEveryModelConstraint) {
    const auto expect_rejected = [](const std::function<void(MoeConfigs&)>& mutate) {
        auto description = makePrefillCudaGraphMoeDescription();
        mutate(description.ffn_conf.moe_configs.value());
        EXPECT_FALSE(
            supportsPrefillCudaGraphMoe(description, ParallelismConfig{}, makePrefillCudaGraphMoeRuntimeConfig()));
    };

    expect_rejected([](auto& c) { c.tp_size = 2; });
    expect_rejected([](auto& c) { c.dp_size = 2; });
    expect_rejected([](auto& c) { c.ep_size = 2; });
    expect_rejected([](auto& c) { c.use_all_gather = false; });
    expect_rejected([](auto& c) { c.expert_num = 0; });
    expect_rejected([](auto& c) { c.top_k = 0; });
    expect_rejected([](auto& c) { c.top_k = c.expert_num + 1; });
    expect_rejected([](auto& c) { c.extra_expert_num = 1; });
    expect_rejected([](auto& c) { c.enable_eplb = true; });
}

TEST_F(ModelDataTest, testPrefillCudaGraphRejectsDistributedOrEplbMoe) {
    auto parallelism       = ParallelismConfig{};
    parallelism.ep_size    = 2;
    parallelism.dp_size    = 2;
    parallelism.world_size = 2;
    EXPECT_FALSE(supportsPrefillCudaGraphMoe(
        makePrefillCudaGraphMoeDescription(), parallelism, makePrefillCudaGraphMoeRuntimeConfig()));

    auto description                              = makePrefillCudaGraphMoeDescription();
    description.ffn_conf.moe_configs->enable_eplb = true;
    EXPECT_FALSE(supportsPrefillCudaGraphMoe(description, ParallelismConfig{}, makePrefillCudaGraphMoeRuntimeConfig()));
}

TEST_F(ModelDataTest, testContextParallelRejectsInputEmbeddingsBeforeMicroBatch) {
    ExecProperties device_props;
    device_props.enable_prefill_cp        = true;
    device_props.enable_layer_micro_batch = MicroBatchType::DS_PREFILL;

    GptModelInputs inputs;
    inputs.input_embeddings = std::vector<torch::Tensor>{torch::rand({1, 8}, torch::kFloat32)};

    EXPECT_THROW(PyWrappedModel::rejectContextParallelInputEmbeddings(device_props, inputs), std::exception);
}

TEST_F(ModelDataTest, testContextParallelAllowsEmptyInputEmbeddings) {
    ExecProperties device_props;
    device_props.enable_prefill_cp        = true;
    device_props.enable_layer_micro_batch = MicroBatchType::DS_PREFILL;

    GptModelInputs inputs;
    inputs.input_embeddings = std::vector<torch::Tensor>();

    EXPECT_NO_THROW(PyWrappedModel::rejectContextParallelInputEmbeddings(device_props, inputs));
}

}  // namespace rtp_llm
