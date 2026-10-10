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

// 64 first-level tokens, each with 16 children and a terminal EOS edge:
// enough distinct paths for the production 1 -> 64 -> 1024 beam schedule.
std::string variableBeamArtifact(uint64_t version) {
    std::vector<int32_t> rows{0}, tokens, next_states;
    for (int32_t parent = 0; parent < 64; ++parent) {
        tokens.push_back(10 + parent);
        next_states.push_back(1 + parent);
    }
    rows.push_back(tokens.size());
    for (int32_t parent = 0; parent < 64; ++parent) {
        for (int32_t child = 0; child < 16; ++child) {
            tokens.push_back(100 + child);
            next_states.push_back(65 + parent * 16 + child);
        }
        rows.push_back(tokens.size());
    }
    for (int32_t leaf = 0; leaf < 1024; ++leaf) {
        tokens.push_back(2);
        next_states.push_back(-1);
        rows.push_back(tokens.size());
    }
    std::string out("RTPCSR01", 8);
    u32(out, 1); u32(out, 48); u32(out, version); u32(out, version >> 32);
    u32(out, 225); u32(out, 2); u32(out, rows.size() - 1); u32(out, tokens.size());
    u32(out, 1024); u32(out, 0);
    for (int32_t value : rows) { u32(out, value); }
    for (int32_t value : tokens) { u32(out, value); }
    for (int32_t value : next_states) { u32(out, value); }
    return out;
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

TEST(RuntimeConstraintTreeTest, FixedBeamConsumesIncrementalTokensWithNonemptyPrompt) {
    auto snapshot = load();
    TreeLogitsProcessor processor({StreamTreeInfo(true, 831, 0, true, snapshot)});
    processor.updateMultiSeqStatus({0, 0});
    ASSERT_FALSE(processor.updateStatus(torch::tensor({{10}, {11}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(processor.committedOutputLen().value(), 1);
    const std::string prefix = "csr:v" + std::to_string(snapshot->version()) + ":state:";
    EXPECT_EQ(processor.getStatus(), (std::vector<std::string>{prefix + "1", prefix + "2"}));

    // Reorder existing beams, retaining their CSR state and committed length.
    processor.updateMultiSeqStatus({1, 0});
    EXPECT_EQ(processor.getStatus(), (std::vector<std::string>{prefix + "2", prefix + "1"}));
    ASSERT_FALSE(processor.updateStatus(torch::tensor({{2}, {2}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(processor.committedOutputLen().value(), 2);
    EXPECT_EQ(processor.getStatus(), (std::vector<std::string>{prefix + "-1", prefix + "-1"}));
    ASSERT_FALSE(processor.updateStatus(torch::tensor({{2}, {2}}, torch::kInt32), 1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::tensor({{10}, {2}}, torch::kInt32), 1).has_value());
}

TEST(RuntimeConstraintTreeTest, VariableBeam64To1024ConsumesIncrementalTokensAndPinsSnapshot) {
    auto manager = ConstraintTreeCsrManager::instance();
    int device_id = -1;
    auto logits_options = torch::TensorOptions().dtype(torch::kFloat32);
#if USING_CUDA || USING_ROCM
    device_id = 0;
    logits_options = logits_options.device(torch::kCUDA);
#endif
    ASSERT_TRUE(manager->updateFromBinary(variableBeamArtifact(manager->currentVersion() + 1), device_id).ok());
    auto snapshot = manager->snapshot();
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::zeros({831}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    input->generate_config->num_beams = 64;
    input->generate_config->variable_num_beams = {64, 1024};
    input->generate_config->num_return_sequences = 1024;
    input->generate_config->max_new_tokens = 2;
    ASSERT_TRUE(TreeLogitsProcessor::validateCsrRequest(snapshot, *input->generate_config, true).empty());
    auto processor = TreeLogitsProcessor::fromGenerateInput(input, 1, snapshot);
    ASSERT_NE(processor, nullptr);

    processor->updateMultiSeqStatus(std::vector<int>(64, 0));
    auto first_tokens = torch::empty({64, 1}, torch::kInt32);
    for (int32_t i = 0; i < 64; ++i) { first_tokens.data_ptr<int32_t>()[i] = 10 + i; }
    ASSERT_FALSE(processor->updateStatus(first_tokens, 1).has_value());
    ASSERT_EQ(processor->committedOutputLen().value(), 1);
    SamplerInputs expanded_inputs;
    expanded_inputs.logits = torch::zeros({64, 128}, logits_options);
    ASSERT_FALSE(processor->process(expanded_inputs, 0, 64).has_value());
    auto expanded_finite = torch::isfinite(expanded_inputs.logits).to(torch::kCPU);
    for (int32_t row = 0; row < 64; ++row) {
        for (int32_t token = 0; token < 128; ++token) {
            EXPECT_EQ(expanded_finite[row][token].item<bool>(), token >= 100 && token < 116);
        }
    }

    // A refresh between steps must not replace an in-flight request's snapshot.
    auto replacement = load();
    ASSERT_GT(replacement->version(), snapshot->version());
    std::vector<int> parents(1024);
    auto second_tokens = torch::empty({1024, 1}, torch::kInt32);
    for (int32_t i = 0; i < 1024; ++i) {
        parents[i] = 63 - i % 64;
        second_tokens.data_ptr<int32_t>()[i] = 100 + i / 64;
    }
    processor->updateMultiSeqStatus(parents);
    ASSERT_EQ(processor->size(), 1024u);
    ASSERT_FALSE(processor->updateStatus(second_tokens, 1).has_value());
    EXPECT_EQ(processor->committedOutputLen().value(), 2);
    const auto statuses = processor->getStatus();
    ASSERT_EQ(statuses.size(), 1024u);
    const std::string prefix = "csr:v" + std::to_string(snapshot->version()) + ":state:";
    for (int32_t i = 0; i < 1024; ++i) {
        EXPECT_EQ(statuses[i], prefix + std::to_string(65 + parents[i] * 16 + i / 64));
    }
    SamplerInputs inputs;
    inputs.logits = torch::zeros({1024, 128}, logits_options);
    ASSERT_FALSE(processor->process(inputs, 0, 1024).has_value());
    auto finite = torch::isfinite(inputs.logits);
    EXPECT_EQ(finite.sum().item<int64_t>(), 1024);
    EXPECT_TRUE(finite.select(1, 2).all().item<bool>());

    // A request admitted after activation uses the replacement, while the
    // expanded request above still uses its original GPU-backed snapshot.
    input->generate_config->num_beams = 2;
    input->generate_config->variable_num_beams = {2, 4};
    ASSERT_TRUE(TreeLogitsProcessor::validateCsrRequest(manager->snapshot(), *input->generate_config, true).empty());
    auto fresh_processor = TreeLogitsProcessor::fromGenerateInput(input, 1);
    ASSERT_NE(fresh_processor, nullptr);
    EXPECT_EQ(fresh_processor->getStatus(),
              (std::vector<std::string>{"csr:v" + std::to_string(replacement->version()) + ":state:0"}));
    fresh_processor->updateMultiSeqStatus({0, 0});
    ASSERT_FALSE(fresh_processor->updateStatus(torch::tensor({{10}, {11}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(fresh_processor->committedOutputLen().value(), 1);
    EXPECT_EQ(fresh_processor->getStatus(),
              (std::vector<std::string>{"csr:v" + std::to_string(replacement->version()) + ":state:1",
                                        "csr:v" + std::to_string(replacement->version()) + ":state:2"}));
}

TEST(RuntimeConstraintTreeTest, IncrementalBeamConsumesOnlyCommittedTokenPrefix) {
    auto snapshot = load();
    TreeLogitsProcessor processor({StreamTreeInfo(true, 831, 0, true, snapshot)});
    // Extra uncommitted tokens are not history and must not be consumed.
    ASSERT_FALSE(processor.updateStatus(torch::tensor({{10, 2, 11}}, torch::kInt32), 2).has_value());
    EXPECT_EQ(processor.committedOutputLen().value(), 2);
    EXPECT_EQ(processor.getStatus(),
              (std::vector<std::string>{"csr:v" + std::to_string(snapshot->version()) + ":state:-1"}));
}

TEST(RuntimeConstraintTreeTest, IncrementalBeamRejectsMalformedBuffersBeforeAdvancing) {
    auto snapshot = load();
    TreeLogitsProcessor processor({StreamTreeInfo(true, 831, 0, true, snapshot)});
    processor.updateMultiSeqStatus({0, 0});
    const auto initial_status = processor.getStatus();
    EXPECT_TRUE(processor.updateStatus(torch::tensor({10, 11}, torch::kInt32), 1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::tensor({{10}}, torch::kInt32), 1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::empty({2, 0}, torch::kInt32), 1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::zeros({2, 1}, torch::kInt32), -1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::tensor({{10}, {11}}, torch::kInt64), 1).has_value());
    auto noncontiguous = torch::zeros({2, 2}, torch::kInt32).transpose(0, 1);
    ASSERT_FALSE(noncontiguous.is_contiguous());
    EXPECT_TRUE(processor.updateStatus(noncontiguous, 1).has_value());
    EXPECT_EQ(processor.getStatus(), initial_status);
    EXPECT_EQ(processor.committedOutputLen().value(), 0);
    // Invalid transitions still fail closed, independently of buffer layout.
    EXPECT_TRUE(processor.updateStatus(torch::tensor({{99}, {11}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(processor.getStatus(), initial_status);
}
#if USING_CUDA || USING_ROCM
TEST(RuntimeConstraintTreeTest, NativeGpuMaskMatchesAllowedRowsAfterUpload) {
    auto manager = ConstraintTreeCsrManager::instance();
    ASSERT_TRUE(manager->updateFromBinary(artifact(manager->currentVersion() + 1), 0).ok());
    auto snapshot = manager->snapshot();
    ASSERT_TRUE(snapshot->deviceReady());
    for (const auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        auto root = StreamTreeInfo(true, 0, 0, false, snapshot);
        auto leaf = StreamTreeInfo(true, 0, 0, false, snapshot);
        leaf.csr_state = 1;
        TreeLogitsProcessor processor({root, leaf});
        SamplerInputs inputs;
        inputs.logits = torch::zeros({2, 32}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        ASSERT_FALSE(processor.process(inputs, 0, 2).has_value());
        auto host = inputs.logits.to(torch::kCPU, torch::kFloat32);
        for (int token = 0; token < 32; ++token) {
            EXPECT_EQ(token == 10 || token == 11, std::isfinite(host[0][token].item<float>()));
            EXPECT_EQ(token == 2, std::isfinite(host[1][token].item<float>()));
        }
    }
}
#endif
}  // namespace rtp_llm
