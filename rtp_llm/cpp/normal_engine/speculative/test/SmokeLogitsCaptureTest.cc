#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>

#include "rtp_llm/cpp/normal_engine/speculative/SmokeLogitsCapture.h"

namespace rtp_llm {
namespace {

std::filesystem::path captureTestDirectory() {
    const auto temporary = std::getenv("TEST_TMPDIR");
    if (!temporary) {
        throw std::runtime_error("TEST_TMPDIR missing");
    }
    const auto root = std::filesystem::path(temporary) / ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::create_directories(root);
    return root;
}

TEST(SmokeLogitsCaptureTest, AcceptedRowsHaveStableTokenPositionsAndExactFloatBytes) {
    const auto root = captureTestDirectory();
    const auto logits = torch::arange(20, torch::kFloat32).view({4, 5});
    // Native K3.1 NoPE verify inputs have a prefix length, no position IDs,
    // and one padded/rejected draft row after these three accepted rows.
    GptModelInputs inputs;
    inputs.prefix_lengths = torch::tensor({3}, torch::kInt32);
    inputs.combo_tokens   = torch::tensor({6, 7, 8, 99}, torch::kInt32);
    ASSERT_FALSE(inputs.combo_position_ids.defined());
    const smoke_logits_capture::DecodeBatch batch(
        inputs, torch::tensor({3}, torch::kInt32), torch::tensor({{7, 8, 9, 0}}, torch::kInt32));
    batch.write(root, std::string(40, 'a'), 17, {1, 2, 3}, logits, 0, 4, 5);
    std::ifstream metadata(root / "17.decode.1.json");
    const std::string text((std::istreambuf_iterator<char>(metadata)), {});
    EXPECT_NE(text.find("\"positions\":[1,3]"), std::string::npos);
    EXPECT_NE(text.find("\"accepted_tokens\":[7,8,9]"), std::string::npos);
    EXPECT_NE(text.find("\"row_input_tokens\":[6,7,8]"), std::string::npos);
    EXPECT_EQ(std::filesystem::file_size(root / "17.input.i32"), 12u);
    std::ifstream data(root / "17.decode.1.f32", std::ios::binary);
    std::vector<float> actual(10);
    data.read(reinterpret_cast<char*>(actual.data()), actual.size() * sizeof(float));
    EXPECT_EQ(actual, (std::vector<float>{0, 1, 2, 3, 4, 10, 11, 12, 13, 14}));
    EXPECT_EQ(data.peek(), std::char_traits<char>::eof());
    EXPECT_THROW(batch.write(root, std::string(40, 'a'), 17, {1, 2, 3}, logits, 0, 4, 5),
                 std::runtime_error);
}

TEST(SmokeLogitsCaptureTest, UnacceptedDraftHistoryCannotProduceALogitsProof) {
    const auto root = captureTestDirectory();
    const auto logits = torch::zeros({3, 5});
    EXPECT_THROW(smoke_logits_capture::writeFrame(root, std::string(40, 'a'), 18, "decode", 1,
                    {1, 2, 3}, {3, 99, 8}, {7, 8, 9}, logits, 5), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(root / "18.decode.1.json"));
}

TEST(SmokeLogitsCaptureTest, PrefillPreservesRawLogitsAndRecordsAppendedSamplerToken) {
    const auto root = captureTestDirectory();
    auto logits = torch::tensor({{-2.0f, 5.0f, 1.0f}, {4.0f, 3.0f, 2.0f}});
    const smoke_logits_capture::PrefillBatch batch(logits);
    logits.zero_();
    const auto sampled = batch.sampledTokens(torch::tensor({{8, 9, 1}, {6, 7, 0}}, torch::kInt32));
    ASSERT_EQ(sampled, (std::vector<int64_t>{1, 0}));
    batch.write(root, std::string(40, 'a'), 20, {8, 9}, 0, sampled[0], 3);
    std::ifstream metadata(root / "20.prefill.0.json");
    const std::string text((std::istreambuf_iterator<char>(metadata)), {});
    EXPECT_NE(text.find("\"accepted_tokens\":[1]"), std::string::npos);
    std::ifstream data(root / "20.prefill.0.f32", std::ios::binary);
    std::vector<float> actual(3);
    data.read(reinterpret_cast<char*>(actual.data()), actual.size() * sizeof(float));
    EXPECT_EQ(actual, (std::vector<float>{-2.0f, 5.0f, 1.0f}));
}

TEST(SmokeLogitsCaptureTest, RequestIdentityCannotChangeItsInputTokens) {
    const auto root = captureTestDirectory();
    const auto logits = torch::zeros({1, 5});
    smoke_logits_capture::writeFrame(root, std::string(40, 'a'), 19, "prefill", 0,
                                     {1, 2, 3}, {3}, {7}, logits, 5);
    EXPECT_THROW(smoke_logits_capture::writeFrame(root, std::string(40, 'a'), 19, "decode", 1,
                    {1, 2, 4}, {7}, {8}, logits, 5), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(root / "19.decode.1.json"));
}

}  // namespace
}  // namespace rtp_llm
