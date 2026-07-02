#include "gtest/gtest.h"

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

// 构造一个 StreamRecommendationInfo 便捷方法
static StreamRecommendationInfo makeInfo(int32_t                              combo_token_size,
                                         const std::vector<std::vector<int>>& banned,
                                         bool                                 is_beam_search = false,
                                         int32_t                              input_length   = 0) {
    std::set<std::vector<int>> banned_set(banned.begin(), banned.end());
    return StreamRecommendationInfo(combo_token_size, input_length, 0, is_beam_search, banned_set);
}

// 分配一个仅装载 logits 的 SamplerInputs（vocab_size 可控）
static SamplerInputs allocateSamplerInputs(size_t batch_size, size_t vocab_size, BaseLogitsProcessorPtr processor) {
    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits         = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    state_ptr->insert(processor, 0, batch_size);
    sampler_inputs.logits_processor_states_ptr = state_ptr;
    return sampler_inputs;
}

class RecommendationLogitsProcessorTest: public DeviceTestBase {};

// 场景 1：非 combo 末位时不做任何屏蔽
TEST_F(RecommendationLogitsProcessorTest, testProcessNoMaskWhenNotLastPosition) {
    const size_t                          batch_size = 1;
    const size_t                          vocab_size = 100;
    std::vector<StreamRecommendationInfo> infos;
    auto                                  info = makeInfo(3, {{10, 20, 30}});
    // 处于 combo 的第 0 位，不应该屏蔽任何 token
    info.pos_in_combo   = 0;
    info.current_prefix = {};
    infos.push_back(info);
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    auto sampler_inputs = allocateSamplerInputs(batch_size, vocab_size, processor);
    sampler_inputs.logits.fill_(1.0f);
    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();
    // 所有位置都应仍然是 1.0f，没有被置为 -inf
    for (size_t j = 0; j < vocab_size; ++j) {
        ASSERT_FLOAT_EQ(data[j], 1.0f);
    }
}

// 场景 2：在 combo 末位时，对命中前缀的 banned combo 的最后一个 token 做屏蔽
TEST_F(RecommendationLogitsProcessorTest, testProcessMaskAtLastPosition) {
    const size_t                          batch_size = 1;
    const size_t                          vocab_size = 100;
    std::vector<StreamRecommendationInfo> infos;
    // banned 有两组；前缀匹配第一组 (10, 20)，应屏蔽 30；不匹配第二组 (15, 25)
    auto info           = makeInfo(3, {{10, 20, 30}, {15, 25, 35}});
    info.pos_in_combo   = 2;
    info.current_prefix = {10, 20};
    infos.push_back(info);
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    auto sampler_inputs = allocateSamplerInputs(batch_size, vocab_size, processor);
    sampler_inputs.logits.fill_(1.0f);
    processor->process(sampler_inputs, 0, batch_size);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();
    ASSERT_TRUE(data[30] == -std::numeric_limits<float>::max() || data[30] == -INFINITY);
    ASSERT_FLOAT_EQ(data[29], 1.0f);
    ASSERT_FLOAT_EQ(data[31], 1.0f);
    // 未匹配前缀的 35 不应被屏蔽
    ASSERT_FLOAT_EQ(data[35], 1.0f);
}

// 场景 3：updateStatus 推进状态机；产生完整 combo 后自动加入 banned_combos（去重）
TEST_F(RecommendationLogitsProcessorTest, testUpdateStatusDedup) {
    std::vector<StreamRecommendationInfo> infos;
    infos.push_back(makeInfo(3, {}));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    auto step = [&](int token_id) {
        auto t = torch::tensor({{token_id}}, torch::kInt32);
        processor->updateStatus(t, 1);
    };

    // 生成 combo (7, 8, 9)
    step(7);
    EXPECT_EQ(1, processor->infos()[0].pos_in_combo);
    EXPECT_EQ(std::vector<int>({7}), processor->infos()[0].current_prefix);

    step(8);
    EXPECT_EQ(2, processor->infos()[0].pos_in_combo);
    EXPECT_EQ(std::vector<int>({7, 8}), processor->infos()[0].current_prefix);

    step(9);
    EXPECT_EQ(0, processor->infos()[0].pos_in_combo);
    EXPECT_TRUE(processor->infos()[0].current_prefix.empty());
    // 完整 combo 应加入 banned_combos
    ASSERT_EQ(1u, processor->infos()[0].banned_combos.size());
    EXPECT_EQ(std::vector<int>({7, 8, 9}), *processor->infos()[0].banned_combos.begin());
}

// 场景 4：去重闭环 —— 生成第一个 combo 后，再次到达同一前缀时该 combo 的最后一位被屏蔽
TEST_F(RecommendationLogitsProcessorTest, testDedupBlockRepeatedCombo) {
    const size_t                          vocab_size = 100;
    std::vector<StreamRecommendationInfo> infos;
    infos.push_back(makeInfo(3, {}));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // 先生成 (7, 8, 9)，让其进入 banned_combos
    auto step = [&](int token_id) {
        auto t = torch::tensor({{token_id}}, torch::kInt32);
        processor->updateStatus(t, 1);
    };
    step(7);
    step(8);
    step(9);

    // 下一轮：推进到 combo 末位，前缀再次为 (7, 8)
    step(7);
    step(8);
    ASSERT_EQ(2, processor->infos()[0].pos_in_combo);

    auto sampler_inputs = allocateSamplerInputs(1, vocab_size, processor);
    sampler_inputs.logits.fill_(1.0f);
    processor->process(sampler_inputs, 0, 1);

    auto logits_cpu = sampler_inputs.logits.cpu();
    auto data       = logits_cpu.data_ptr<float>();
    ASSERT_TRUE(data[9] == -std::numeric_limits<float>::max() || data[9] == -INFINITY);
    ASSERT_FLOAT_EQ(data[8], 1.0f);
}

// 场景 5：updateMultiSeqStatus 按 src_batch_indices 正确复制状态（独立副本）
TEST_F(RecommendationLogitsProcessorTest, testUpdateMultiSeqStatusCopy) {
    std::vector<StreamRecommendationInfo> infos;
    auto                                  info = makeInfo(3, {{1, 2, 3}});
    info.pos_in_combo                          = 1;
    info.current_prefix                        = {1};
    infos.push_back(info);
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // 将原有 1 条复制成 3 条（均来自索引 0）
    processor->updateMultiSeqStatus({0, 0, 0});
    ASSERT_EQ(3u, processor->size());

    // 修改副本 0 的状态不应影响副本 1/2
    auto t = torch::tensor({{9}, {0}, {0}}, torch::kInt32);
    processor->updateStatus(t, 1);

    EXPECT_EQ(std::vector<int>({1, 9}), processor->infos()[0].current_prefix);
    EXPECT_EQ(std::vector<int>({1, 0}), processor->infos()[1].current_prefix);
    EXPECT_EQ(std::vector<int>({1, 0}), processor->infos()[2].current_prefix);
}

// 场景 6：fromGenerateInput 在 combo_token_size<=0 时返回 nullptr
TEST_F(RecommendationLogitsProcessorTest, testFromGenerateInputDisabled) {
    auto generate_input                                     = std::make_shared<GenerateInput>();
    generate_input->generate_config                         = std::make_shared<GenerateConfig>();
    generate_input->generate_config->combo_token_size       = 0;
    generate_input->generate_config->banned_combo_token_ids = {};
    // 用 1 维的 input_ids 占位
    generate_input->input_ids = torch::zeros({1}, torch::kInt32);

    auto result = RecommendationLogitsProcessor::fromGenerateInput(generate_input, 2);
    ASSERT_EQ(nullptr, result);
}

// 场景 7：fromGenerateInput 在启用时正确初始化 banned_combos 并按 num 扩展 batch
TEST_F(RecommendationLogitsProcessorTest, testFromGenerateInputEnabled) {
    auto generate_input                                     = std::make_shared<GenerateInput>();
    generate_input->generate_config                         = std::make_shared<GenerateConfig>();
    generate_input->generate_config->combo_token_size       = 3;
    generate_input->generate_config->banned_combo_token_ids = {
        {1, 2, 3}, {4, 5, 6}, {7, 8}};  // 最后一个长度不等将被过滤
    generate_input->input_ids = torch::zeros({5}, torch::kInt32);

    auto p = RecommendationLogitsProcessor::fromGenerateInput(generate_input, 2);
    ASSERT_NE(nullptr, p);
    ASSERT_EQ(2u, p->size());
    for (const auto& info : p->infos()) {
        EXPECT_EQ(3, info.combo_token_size);
        EXPECT_EQ(2u, info.banned_combos.size());  // (7,8) 已被过滤
        EXPECT_EQ(5, info.input_length);
    }
}

}  // namespace rtp_llm
