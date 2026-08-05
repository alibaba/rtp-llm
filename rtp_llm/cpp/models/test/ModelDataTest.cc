
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/Sampler.h"

#include <thread>

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

TEST_F(ModelDataTest, CacheGroupHintsUseCompactCachedSchema) {
    const std::vector<CacheGroupHint> hints = {
        {"full", CacheGroupType::FULL, 8, 8},
        {"linear-attention", CacheGroupType::LINEAR, 3, 6},
    };
    const auto wire = encodeCacheGroupHintSchema(hints);
    EXPECT_EQ(wire.size(),
              2 * CacheGroupHintWireFormat::kSchemaHeaderWords + CacheGroupHintWireFormat::tagWords(hints[0].tag.size())
                  + CacheGroupHintWireFormat::tagWords(hints[1].tag.size()));
    EXPECT_LT(CacheGroupHintWireFormat::kShapeHintWords,
              GptModelInputIndex::gptModelInputLength + CacheGroupHintWireFormat::kMaxSchemaWords);
}

TEST_F(ModelDataTest, CacheGroupHintCodecReconstructsNonRootMetadata) {
    const std::vector<CacheGroupHint> root_hints = {
        {"full", CacheGroupType::FULL, 8, 9},
        {"linear-attention", CacheGroupType::LINEAR, 3, 6},
        {"swa", CacheGroupType::SWA, 2, 4},
    };
    const auto wire    = encodeCacheGroupHintSchema(root_hints);
    const auto decoded = decodeCacheGroupHints(wire, root_hints.size(), {8, 9, 3, 6, 2, 4});
    ASSERT_EQ(decoded.size(), root_hints.size());
    for (size_t i = 0; i < root_hints.size(); ++i) {
        EXPECT_EQ(decoded[i].tag, root_hints[i].tag);
        EXPECT_EQ(decoded[i].type, root_hints[i].type);
        EXPECT_EQ(decoded[i].block_width, root_hints[i].block_width);
        EXPECT_EQ(decoded[i].kernel_block_width, root_hints[i].kernel_block_width);
    }
}

TEST_F(ModelDataTest, CacheGroupHintsReconstructNonRootBlockTableViews) {
    const std::vector<CacheGroupHint> hints = {
        {"full", CacheGroupType::FULL, 2, 3},
        {"linear", CacheGroupType::LINEAR, 1, 2},
    };
    constexpr int64_t batch_size = 2;
    auto              physical   = torch::arange(0, 6, torch::kInt32);
    auto              kernel     = torch::arange(10, 20, torch::kInt32);
    auto              tables     = reconstructCacheGroupBlockTables(hints, batch_size, physical, kernel);

    ASSERT_EQ(tables.size(), 2);
    EXPECT_EQ(tables.at("full").type, CacheGroupType::FULL);
    EXPECT_EQ(tables.at("full").block_ids.sizes(), torch::IntArrayRef({2, 2}));
    EXPECT_EQ(tables.at("full").kernel_block_ids.sizes(), torch::IntArrayRef({2, 3}));
    EXPECT_EQ(tables.at("linear").block_ids.sizes(), torch::IntArrayRef({2, 1}));
    EXPECT_EQ(tables.at("linear").kernel_block_ids.sizes(), torch::IntArrayRef({2, 2}));
    const auto linear_blocks = tables.at("linear").block_ids.flatten().contiguous();
    EXPECT_EQ(std::vector<int>(linear_blocks.data_ptr<int>(), linear_blocks.data_ptr<int>() + linear_blocks.numel()),
              std::vector<int>({4, 5}));
    const auto linear_kernel_blocks = tables.at("linear").kernel_block_ids.flatten().contiguous();
    EXPECT_EQ(std::vector<int>(linear_kernel_blocks.data_ptr<int>(),
                               linear_kernel_blocks.data_ptr<int>() + linear_kernel_blocks.numel()),
              std::vector<int>({16, 17, 18, 19}));
    EXPECT_ANY_THROW(reconstructCacheGroupBlockTables(hints, batch_size, torch::zeros({5}), kernel));
}

TEST_F(ModelDataTest, CacheGroupHintCodecRejectsMalformedFrames) {
    const std::vector<CacheGroupHint> hints = {{"full", CacheGroupType::FULL, 8, 8}};
    auto                              wire  = encodeCacheGroupHintSchema(hints);
    EXPECT_ANY_THROW(decodeCacheGroupHints(wire, 2, {8, 8, 1, 1}));
    EXPECT_ANY_THROW(decodeCacheGroupHints(wire, 1, {8}));

    auto truncated = wire;
    truncated.pop_back();
    EXPECT_ANY_THROW(decodeCacheGroupHints(truncated, 1, {8, 8}));

    auto invalid_type = wire;
    invalid_type[1]   = 99;
    EXPECT_ANY_THROW(decodeCacheGroupHints(invalid_type, 1, {8, 8}));
    EXPECT_ANY_THROW(decodeCacheGroupHints(wire, 1, {-1, 8}));

    const auto duplicate_wire =
        encodeCacheGroupHintSchema({{"same", CacheGroupType::FULL}, {"same", CacheGroupType::LINEAR}});
    EXPECT_ANY_THROW(decodeCacheGroupHints(duplicate_wire, 2, {1, 1, 2, 2}));
}

TEST_F(ModelDataTest, CacheGroupSchemaCacheUsesRootAuthoritativePayloadPolicy) {
    CacheGroupSchemaCache             cache;
    const CacheGroupSchemaKey         key{1234, 1};
    const std::vector<CacheGroupHint> first   = {{"full", CacheGroupType::FULL}};
    const std::vector<CacheGroupHint> changed = {{"linear", CacheGroupType::LINEAR}};

    EXPECT_TRUE(cache.rootPayloadFollows(key, first));
    EXPECT_ANY_THROW(cache.lookup(key));
    cache.refresh(key, first);
    EXPECT_FALSE(cache.rootPayloadFollows(key, first));
    EXPECT_EQ(cache.lookup(key)[0].tag, "full");

    EXPECT_TRUE(cache.rootPayloadFollows(key, changed));
    cache.refresh(key, changed);
    EXPECT_FALSE(cache.rootPayloadFollows(key, changed));
    EXPECT_EQ(cache.lookup(key)[0].type, CacheGroupType::LINEAR);

    const CacheGroupSchemaKey next_key{5678, 1};
    EXPECT_TRUE(cache.rootPayloadFollows(next_key, first));
}

TEST_F(ModelDataTest, CacheGroupSchemaCacheIsSharedAcrossCallingThreads) {
    CacheGroupSchemaCache             cache;
    const CacheGroupSchemaKey         key{9876, 1};
    const std::vector<CacheGroupHint> schema = {{"swa", CacheGroupType::SWA}};
    std::thread                       writer([&]() { cache.refresh(key, schema); });
    writer.join();

    EXPECT_FALSE(cache.rootPayloadFollows(key, schema));
    EXPECT_EQ(cache.lookup(key)[0].tag, "swa");
}

TEST_F(ModelDataTest, CacheGroupSchemaCacheIsScopedToCommunicatorGeneration) {
    CacheGroupSchemaCache             cache;
    const std::vector<CacheGroupHint> schema = {{"full", CacheGroupType::FULL}};
    const CacheGroupSchemaKey         first_generation{1234, 1, 7};
    const CacheGroupSchemaKey         next_generation{1234, 1, 8};

    cache.refresh(first_generation, schema);
    EXPECT_FALSE(cache.rootPayloadFollows(first_generation, schema));
    EXPECT_TRUE(cache.rootPayloadFollows(next_generation, schema));
    EXPECT_ANY_THROW(cache.lookup(next_generation));

    cache.refresh(next_generation, schema);
    EXPECT_FALSE(cache.rootPayloadFollows(next_generation, schema));
    EXPECT_EQ(cache.lookup(next_generation)[0].tag, "full");
}

}  // namespace rtp_llm
