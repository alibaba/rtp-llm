
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/FullPrefillCudaGraphEligibility.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
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

namespace {

GptModelDescription makeFullPrefillMoeDescription() {
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

MoeConfig makeFullPrefillMoeRuntimeConfig() {
    MoeConfig config;
    config.moe_strategy           = "fp8_per_block_no_dp_masked";
    config.use_all_gather         = true;
    config.use_deepep_moe         = false;
    config.use_deepep_internode   = false;
    config.use_deepep_low_latency = false;
    return config;
}

}  // namespace

TEST_F(ModelDataTest, testFullPrefillCudaGraphSupportsDenseModel) {
    GptModelDescription description;
    EXPECT_TRUE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, MoeConfig{}));
}

TEST_F(ModelDataTest, testFullPrefillCudaGraphSupportsSingleGpuFp8MaskedMoe) {
    EXPECT_TRUE(supportsFullPrefillCudaGraphMoe(
        makeFullPrefillMoeDescription(), ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig()));
}

TEST_F(ModelDataTest, testFullPrefillCudaGraphRejectsAutoMoeStrategy) {
    auto config         = makeFullPrefillMoeRuntimeConfig();
    config.moe_strategy = "auto";
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(makeFullPrefillMoeDescription(), ParallelismConfig{}, config));
}

TEST_F(ModelDataTest, testFullPrefillCudaGraphRejectsNonFp8PerBlockMoe) {
    auto description        = makeFullPrefillMoeDescription();
    description.act_qscheme = QScheme::NoQuantize;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig()));
}

TEST_F(ModelDataTest, testFullPrefillCudaGraphRejectsGraphUnsafeMoeTransport) {
    auto config                   = makeFullPrefillMoeRuntimeConfig();
    config.use_deepep_low_latency = true;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(makeFullPrefillMoeDescription(), ParallelismConfig{}, config));

    config                = makeFullPrefillMoeRuntimeConfig();
    config.use_all_gather = false;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(makeFullPrefillMoeDescription(), ParallelismConfig{}, config));
}

TEST_F(ModelDataTest, testFullPrefillCudaGraphRejectsDistributedOrEplbMoe) {
    auto parallelism       = ParallelismConfig{};
    parallelism.ep_size    = 2;
    parallelism.dp_size    = 2;
    parallelism.world_size = 2;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(
        makeFullPrefillMoeDescription(), parallelism, makeFullPrefillMoeRuntimeConfig()));

    auto description                              = makeFullPrefillMoeDescription();
    description.ffn_conf.moe_configs->enable_eplb = true;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig()));
}

}  // namespace rtp_llm
