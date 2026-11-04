#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/models/logits_processor/FixedContinuationLogitsProcessor.h"

namespace rtp_llm {
namespace {

class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value): name_(name) {
        if (const char* old = std::getenv(name)) {
            previous_ = old;
        }
        value ? setenv(name, value, 1) : unsetenv(name);
    }
    ~ScopedEnv() {
        previous_ ? setenv(name_.c_str(), previous_->c_str(), 1) : unsetenv(name_.c_str());
    }
private:
    std::string                name_;
    std::optional<std::string> previous_;
};

class FixedContinuationTest: public testing::Test {
protected:
    void SetUp() override {
        char name[] = "/tmp/rtp-fixed-continuation-XXXXXX";
        auto path = mkdtemp(name);
        ASSERT_NE(path, nullptr);
        directory_ = path;
    }
    void TearDown() override {
        std::filesystem::remove_all(directory_);
    }
    FixedContinuationLogitsProcessor processor(int64_t request_id = 1) {
        return {{2, 3}, {4, 5}, 8, directory_.string(), "case", request_id};
    }
    SamplerInputs inputs(int32_t length = 2) {
        SamplerInputs result{};
        result.logits = torch::arange(8, torch::kFloat32).reshape({1, 8});
        result.token_ids = torch::tensor({{2, 3, 4, 0}}, torch::kInt32);
        result.sequence_lengths = torch::tensor({length}, torch::kInt32);
        return result;
    }
    std::filesystem::path directory_;
};

TEST_F(FixedContinuationTest, CapturesRawRowBeforeMaskAndChecksCommittedToken) {
    auto fixed = processor();
    auto sampler = inputs();
    auto original = sampler.logits.clone();
    ASSERT_FALSE(fixed.process(sampler, 0, 1).has_value());
    EXPECT_EQ(sampler.logits[0][4].item<float>(), 0.0f);
    EXPECT_TRUE(torch::isneginf(sampler.logits[0][7]).item<bool>());
    EXPECT_FALSE(std::filesystem::exists(directory_ / "case-request-1/row-0.f32"));
    ASSERT_FALSE(fixed.updateStatus(torch::tensor({{4}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(fixed.committedOutputLen(), 1);
    std::array<float, 8> values{};
    std::ifstream reader(directory_ / "case-request-1/row-0.f32", std::ios::binary);
    reader.read(reinterpret_cast<char*>(values.data()), sizeof(values));
    ASSERT_EQ(reader.gcount(), sizeof(values));
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], original[0][i].item<float>());
    }
}

TEST_F(FixedContinuationTest, DecodeBootstrapVerifiesFirstTokenWithoutInventingPrefillLogits) {
    auto fixed = processor();
    ASSERT_FALSE(fixed.updateStatus(torch::tensor({{4}}, torch::kInt32), 1).has_value());
    auto sampler = inputs(3);
    ASSERT_FALSE(fixed.process(sampler, 0, 1).has_value());
    ASSERT_FALSE(fixed.updateStatus(torch::tensor({{5}}, torch::kInt32), 1).has_value());
    EXPECT_FALSE(std::filesystem::exists(directory_ / "case-request-1/row-0.f32"));
    EXPECT_TRUE(std::filesystem::exists(directory_ / "case-request-1/row-1.f32"));
    EXPECT_EQ(fixed.committedOutputLen(), 2);
}

TEST_F(FixedContinuationTest, RejectsWrongHistoryLengthAndCommittedTokens) {
    auto fixed = processor();
    auto sampler = inputs(3);
    EXPECT_TRUE(fixed.process(sampler, 0, 1).has_value());
    sampler = inputs();
    sampler.token_ids[0][0] = 1;
    EXPECT_TRUE(fixed.process(sampler, 0, 1).has_value());
    EXPECT_TRUE(fixed.updateStatus(torch::tensor({{7}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(fixed.committedOutputLen(), 0);
    EXPECT_FALSE(std::filesystem::exists(directory_ / "case-request-1"));
}

TEST_F(FixedContinuationTest, RequiresOperatorOptInAndExactFrozenInput) {
    auto input = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    auto& config = *input->generate_config;
    config.task_id = "__rtp_v41_acceptance__/case";
    ScopedEnv disabled("RTP_LLM_FIXED_CONTINUATION_FIXTURE", nullptr);
    ASSERT_TRUE(FixedContinuationLogitsProcessor::fromGenerateInput(input).ok());
    EXPECT_FALSE(FixedContinuationLogitsProcessor::fromGenerateInput(input).value());

    auto fixture_path = directory_ / "fixture.json";
    std::ofstream(fixture_path)
        << R"({"schema_version":1,"vocab_size":8,"cases":{"case":{"input_ids":[2,3],"continuation_tokens":[4,5]}}})";
    ScopedEnv fixture("RTP_LLM_FIXED_CONTINUATION_FIXTURE", fixture_path.c_str());
    ScopedEnv output("RTP_LLM_FIXED_CONTINUATION_OUTPUT", directory_.c_str());
    config.return_logits = config.is_streaming = config.force_disable_sp_run = config.ignore_eos = true;
    config.reuse_cache = false;
    config.max_new_tokens = config.min_new_tokens = 2;
    input->input_ids = torch::tensor({2, 7}, torch::kInt32);
    EXPECT_FALSE(FixedContinuationLogitsProcessor::fromGenerateInput(input).ok());
    input->input_ids = torch::tensor({2, 3}, torch::kInt32);
    auto created = FixedContinuationLogitsProcessor::fromGenerateInput(input);
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    EXPECT_TRUE(created.value());
}

}  // namespace
}  // namespace rtp_llm
