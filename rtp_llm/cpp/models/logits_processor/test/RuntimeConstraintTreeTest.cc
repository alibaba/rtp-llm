#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"

namespace rtp_llm {
namespace {
void u32(std::string& out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) { out.push_back(static_cast<char>(value >> shift)); }
}
std::string artifact(uint64_t version) {
    std::string out("RTPCSR01", 8);
    u32(out, 1); u32(out, 48); u32(out, version); u32(out, version >> 32);
    u32(out, 225); u32(out, 2); u32(out, 3); u32(out, 4); u32(out, 2); u32(out, 0);
    for (int value : {0, 2, 3, 4, 10, 11, 2, 2, 1, 2, -1, -1}) { u32(out, value); }
    return out;
}
ConstraintTreeCsrSnapshotPtr load() {
    auto manager = ConstraintTreeCsrManager::instance();
    EXPECT_TRUE(manager->updateFromBinary(artifact(manager->currentVersion() + 1)).ok());
    return manager->snapshot();
}
}
TEST(RuntimeConstraintTreeTest, PinsSnapshotAndRejectsMalformedReplacement) {
    auto old = load();
    auto next = load();
    EXPECT_LT(old->version(), next->version());
    EXPECT_EQ(1, old->transition(0, 10));
    auto manager = ConstraintTreeCsrManager::instance();
    EXPECT_FALSE(manager->updateFromBinary("invalid").ok());
    EXPECT_EQ(next, manager->snapshot());
    EXPECT_EQ(ConstraintTreeCsrUpdateCode::STALE_VERSION, manager->updateFromBinary(artifact(old->version())).code);
}
TEST(RuntimeConstraintTreeTest, VariableBeamAdmissionAndNonFiniteScoresFailClosed) {
    auto snapshot = load();
    GenerateConfig config;
    config.num_beams = 2;
    config.variable_num_beams = {2, 4};
    EXPECT_TRUE(TreeLogitsProcessor::validateCsrRequest(snapshot, config, true).empty());
    config.variable_num_beams = {3, 4};
    EXPECT_FALSE(TreeLogitsProcessor::validateCsrRequest(snapshot, config, true).empty());
    EXPECT_FALSE(TreeLogitsProcessor::validateCsrRequest(nullptr, config, true).empty());
    TreeLogitsProcessor processor({StreamTreeInfo(true, 0, 0, true, snapshot)});
    EXPECT_FALSE(processor.validateBeamScores(torch::tensor({-1.0f, -2.0f}), 2).has_value());
    EXPECT_TRUE(processor.validateBeamScores(torch::tensor({-1.0f, -std::numeric_limits<float>::infinity()}), 2).has_value());
}
TEST(RuntimeConstraintTreeTest, MasksAndKeepsCompletedBeamsEosOnly) {
    auto snapshot = load();
    TreeLogitsProcessor processor({StreamTreeInfo(true, 0, 0, false, snapshot)});
    SamplerInputs inputs;
    inputs.logits = torch::zeros({1, 32}, torch::kFloat32);
    EXPECT_FALSE(processor.process(inputs, 0, 1).has_value());
    EXPECT_EQ(0.0f, inputs.logits[0][10].item<float>());
    EXPECT_TRUE(std::isinf(inputs.logits[0][9].item<float>()));
    EXPECT_FALSE(processor.updateStatus(torch::tensor({{10}}, torch::kInt32), 1).has_value());
    EXPECT_FALSE(processor.updateStatus(torch::tensor({{2}}, torch::kInt32), 1).has_value());
    inputs.logits.zero_();
    EXPECT_FALSE(processor.process(inputs, 0, 1).has_value());
    EXPECT_EQ(0.0f, inputs.logits[0][2].item<float>());
    EXPECT_TRUE(std::isinf(inputs.logits[0][10].item<float>()));
    EXPECT_FALSE(processor.updateStatus(torch::tensor({{2}}, torch::kInt32), 1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::tensor({{11}}, torch::kInt32), 1).has_value());
}
#if USING_CUDA || USING_ROCM
TEST(RuntimeConstraintTreeTest, NativeGpuMaskMatchesAllowedRowsAfterUpload) {
    auto manager = ConstraintTreeCsrManager::instance();
    ASSERT_TRUE(manager->updateFromBinary(artifact(manager->currentVersion() + 1), 0).ok());
    auto snapshot = manager->snapshot();
    ASSERT_TRUE(snapshot->deviceReady());
    TreeLogitsProcessor processor({StreamTreeInfo(true, 0, 0, false, snapshot)});
    SamplerInputs inputs;
    inputs.logits = torch::zeros({1, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    auto host = inputs.logits.cpu();
    for (int token = 0; token < 32; ++token) {
        EXPECT_EQ(token == 10 || token == 11, std::isfinite(host[0][token].item<float>()));
    }
}
#endif
}  // namespace rtp_llm
