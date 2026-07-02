#include "gtest/gtest.h"

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

// 构造一个 StreamRecommendationInfo 便捷方法
static StreamRecommendationInfo makeInfo(int32_t                                  combo_token_size,
                                         const std::vector<std::vector<int>>&     banned,
                                         bool                                     needs_token_offset = false,
                                         int32_t                                  input_length   = 0) {
    std::set<std::vector<int>> banned_set(banned.begin(), banned.end());
    return StreamRecommendationInfo(combo_token_size, input_length, 0, needs_token_offset, banned_set);
}

// 分配一个仅装载 logits 的 SamplerInputs（vocab_size 可控）
static SamplerInputs allocateSamplerInputs(size_t                                   batch_size,
                                           size_t                                   vocab_size,
                                           BaseLogitsProcessorPtr                   processor) {
    SamplerInputs sampler_inputs;
    sampler_inputs.batch_size     = batch_size;
    sampler_inputs.batch_size_out = batch_size;
    sampler_inputs.vocab_size     = vocab_size;
    sampler_inputs.logits =
        torch::zeros({(int64_t)batch_size, (int64_t)vocab_size},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    state_ptr->insert(processor, 0, batch_size);
    sampler_inputs.logits_processor_states_ptr = state_ptr;
    return sampler_inputs;
}

class RecommendationLogitsProcessorTest: public DeviceTestBase {};

// 场景 1：非 combo 末位时不做任何屏蔽
TEST_F(RecommendationLogitsProcessorTest, testProcessNoMaskWhenNotLastPosition) {
    const size_t batch_size = 1;
    const size_t vocab_size = 100;
    std::vector<StreamRecommendationInfo> infos;
    auto info = makeInfo(3, {{10, 20, 30}});
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
    const size_t batch_size = 1;
    const size_t vocab_size = 100;
    std::vector<StreamRecommendationInfo> infos;
    // banned 有两组；前缀匹配第一组 (10, 20)，应屏蔽 30；不匹配第二组 (15, 25)
    auto info = makeInfo(3, {{10, 20, 30}, {15, 25, 35}});
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
    const size_t vocab_size = 100;
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
    auto info = makeInfo(3, {{1, 2, 3}});
    info.pos_in_combo   = 1;
    info.current_prefix = {1};
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
    auto generate_input    = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->combo_token_size       = 0;
    generate_input->generate_config->banned_combo_token_ids = {};
    // 用 1 维的 input_ids 占位
    generate_input->input_ids = torch::zeros({1}, torch::kInt32);

    auto p = RecommendationLogitsProcessor::fromGenerateInput(generate_input, 2);
    ASSERT_EQ(nullptr, p);
}

// 场景 7：fromGenerateInput 在启用时正确初始化 banned_combos 并按 num 扩展 batch
TEST_F(RecommendationLogitsProcessorTest, testFromGenerateInputEnabled) {
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->combo_token_size       = 3;
    generate_input->generate_config->banned_combo_token_ids = {{1, 2, 3}, {4, 5, 6}, {7, 8}};  // 最后一个长度不等将被过滤
    generate_input->input_ids                               = torch::zeros({5}, torch::kInt32);

    auto p = RecommendationLogitsProcessor::fromGenerateInput(generate_input, 2);
    ASSERT_NE(nullptr, p);
    ASSERT_EQ(2u, p->size());
    for (const auto& info : p->infos()) {
        EXPECT_EQ(3, info.combo_token_size);
        EXPECT_EQ(2u, info.banned_combos.size());  // (7,8) 已被过滤
        EXPECT_EQ(5, info.input_length);
    }
}

// 场景 8：跨序列去重（非对称模式）—— 序列 0 不接收其他序列的 ban，补充序列接收所有
TEST_F(RecommendationLogitsProcessorTest, testCrossSequenceBanAsymmetric) {
    // 模拟 2 条序列，combo_token_size=3，开启跨序列去重
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> empty_set;
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // 序列 0 生成 combo (1,2,3)，序列 1 生成 combo (4,5,6)
    auto step = [&](int tok0, int tok1) {
        auto t = torch::tensor({{tok0}, {tok1}}, torch::kInt32);
        processor->updateStatus(t, 1);
    };
    step(1, 4);  // pos_in_combo: 0 → 1
    step(2, 5);  // pos_in_combo: 1 → 2
    step(3, 6);  // pos_in_combo: 2 → 0 (combo 完成)

    // 非对称广播：序列 0 只有自己的，序列 1 有所有的
    ASSERT_EQ(1u, processor->infos()[0].banned_combos.size());
    ASSERT_EQ(2u, processor->infos()[1].banned_combos.size());
    EXPECT_TRUE(processor->infos()[0].banned_combos.count({1, 2, 3}));
    EXPECT_FALSE(processor->infos()[0].banned_combos.count({4, 5, 6}));
    EXPECT_TRUE(processor->infos()[1].banned_combos.count({1, 2, 3}));
    EXPECT_TRUE(processor->infos()[1].banned_combos.count({4, 5, 6}));

    // 验证 completed_combo_count
    EXPECT_EQ(1, processor->infos()[0].completed_combo_count);
    EXPECT_EQ(1, processor->infos()[1].completed_combo_count);
}

// 场景 9：关闭跨序列去重时，各序列独立维护 banned_combos
TEST_F(RecommendationLogitsProcessorTest, testCrossSequenceBanDisabled) {
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> empty_set;
    // enable_cross_sequence_ban = false
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, false));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, false));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    auto step = [&](int tok0, int tok1) {
        auto t = torch::tensor({{tok0}, {tok1}}, torch::kInt32);
        processor->updateStatus(t, 1);
    };
    step(1, 4);
    step(2, 5);
    step(3, 6);

    // 关闭跨序列去重时，各序列只有自己的 combo
    ASSERT_EQ(1u, processor->infos()[0].banned_combos.size());
    ASSERT_EQ(1u, processor->infos()[1].banned_combos.size());
    EXPECT_TRUE(processor->infos()[0].banned_combos.count({1, 2, 3}));
    EXPECT_FALSE(processor->infos()[0].banned_combos.count({4, 5, 6}));
    EXPECT_TRUE(processor->infos()[1].banned_combos.count({4, 5, 6}));
    EXPECT_FALSE(processor->infos()[1].banned_combos.count({1, 2, 3}));
}

// 场景 10：top-K 分叉遮蔽——非主序列在 combo 起始位置被遮蔽 top-i
TEST_F(RecommendationLogitsProcessorTest, testTopKDivergeMasking) {
    // 3 条序列，combo_token_size=3，开启跨序列去重，diverge_start_combo=0
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> empty_set;
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // 通过 allocateSamplerInputs 在 CUDA 上分配 logits，与生产环境一致
    const size_t vocab_size = 5;
    auto sampler_inputs = allocateSamplerInputs(3, vocab_size, processor);
    // 填充 logits: 每行 = [1.0, 5.0, 3.0, 4.0, 2.0]
    // top-1=col1(5.0), top-2=col3(4.0)
    auto logits_cpu = torch::tensor({{1.0f, 5.0f, 3.0f, 4.0f, 2.0f},
                                      {1.0f, 5.0f, 3.0f, 4.0f, 2.0f},
                                      {1.0f, 5.0f, 3.0f, 4.0f, 2.0f}});
    sampler_inputs.logits.copy_(logits_cpu);

    // 所有序列 pos_in_combo=0 且 completed_combo_count=0 >= diverge_start=0
    processor->process(sampler_inputs, 0, 3);

    auto result = sampler_inputs.logits.cpu();
    // 序列 0（主）：不受影响，top-1 仍然是 col1
    EXPECT_FLOAT_EQ(5.0f, result[0][1].item<float>());
    EXPECT_FLOAT_EQ(4.0f, result[0][3].item<float>());

    // 序列 1：遮蔽 top-1 (col1)，col1 应为 -inf
    EXPECT_TRUE(std::isinf(result[1][1].item<float>()) && result[1][1].item<float>() < 0);
    EXPECT_FLOAT_EQ(4.0f, result[1][3].item<float>());  // top-2 不受影响

    // 序列 2：遮蔽 top-1,2 (col1, col3)，两者应为 -inf
    EXPECT_TRUE(std::isinf(result[2][1].item<float>()) && result[2][1].item<float>() < 0);
    EXPECT_TRUE(std::isinf(result[2][3].item<float>()) && result[2][3].item<float>() < 0);
}

// 场景 11：diverge_start_combo 延迟分叉——前 N 个商品不遮蔽
TEST_F(RecommendationLogitsProcessorTest, testDivergeStartComboDelay) {
    // 2 条序列，combo_token_size=3，diverge_start_combo=1（前 1 个商品不分叉）
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> empty_set;
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 1));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 1));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // 第 1 个 combo 开始时 completed_combo_count=0 < diverge_start=1，不应遮蔽
    const size_t vocab_size = 3;
    auto sampler_inputs1 = allocateSamplerInputs(2, vocab_size, processor);
    auto init_vals1 = torch::tensor({{1.0f, 5.0f, 3.0f}, {1.0f, 5.0f, 3.0f}});
    sampler_inputs1.logits.copy_(init_vals1);
    processor->process(sampler_inputs1, 0, 2);

    auto result1 = sampler_inputs1.logits.cpu();
    // 序列 1 的 top-1 不应被遮蔽（因为还没达到 diverge_start_combo）
    EXPECT_FLOAT_EQ(5.0f, result1[1][1].item<float>());

    // 生成完成第 1 个 combo
    auto step = [&](int tok0, int tok1) {
        auto t = torch::tensor({{tok0}, {tok1}}, torch::kInt32);
        processor->updateStatus(t, 1);
    };
    step(1, 1);
    step(2, 2);
    step(3, 3);
    // 现在 completed_combo_count=1 >= diverge_start=1

    // 第 2 个 combo 开始时，应该对序列 1 遮蔽
    auto sampler_inputs2 = allocateSamplerInputs(2, vocab_size, processor);
    auto init_vals2 = torch::tensor({{1.0f, 5.0f, 3.0f}, {1.0f, 5.0f, 3.0f}});
    sampler_inputs2.logits.copy_(init_vals2);
    processor->process(sampler_inputs2, 0, 2);

    auto result2 = sampler_inputs2.logits.cpu();
    // 序列 1 的 top-1 (col1) 现在应被遮蔽
    EXPECT_TRUE(std::isinf(result2[1][1].item<float>()) && result2[1][1].item<float>() < 0);
    // 序列 0 不受影响
    EXPECT_FLOAT_EQ(5.0f, result2[0][1].item<float>());
}

// 场景 12：combo_token_size==1 时跨序列去重被降级禁用（前置校验）
TEST_F(RecommendationLogitsProcessorTest, testCrossSeqBanDisabledWhenComboSize1) {
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->combo_token_size            = 1;
    generate_input->generate_config->enable_cross_sequence_ban   = true;
    generate_input->generate_config->num_return_sequences        = 4;
    generate_input->generate_config->banned_combo_token_ids      = {{5}};
    generate_input->input_ids                                    = torch::zeros({3}, torch::kInt32);

    auto p = RecommendationLogitsProcessor::fromGenerateInput(generate_input, 4);
    ASSERT_NE(nullptr, p);
    // combo_token_size==1 时 enable_cross_sequence_ban 应被降级为 false
    for (const auto& info : p->infos()) {
        EXPECT_FALSE(info.enable_cross_sequence_ban);
    }
}

// 场景 13：diverge 遮蔽与 banned combo 屏蔽在同一次 process() 中同时生效
TEST_F(RecommendationLogitsProcessorTest, testDivergeAndBanSimultaneous) {
    // 3 条序列，combo_token_size=3，开启跨序列去重
    // 序列 0: pos_in_combo=0 —— 主序列，不受 diverge 影响
    // 序列 1: pos_in_combo=0 —— 触发 diverge 遮蔽 (top-1)
    // 序列 2: pos_in_combo=2 且前缀匹配 banned combo —— 触发 ban 屏蔽
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> banned_with_match({{10, 20, 30}});
    std::set<std::vector<int>> empty_set;

    // 序列 0: pos=0, 主序列
    auto info0 = StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0);
    infos.push_back(info0);

    // 序列 1: pos=0, 待 diverge
    auto info1 = StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0);
    infos.push_back(info1);

    // 序列 2: pos=2, prefix={10,20}, banned={(10,20,30)} —— 待 ban
    auto info2 = StreamRecommendationInfo(3, 0, 0, false, banned_with_match, {}, true, 0);
    info2.pos_in_combo = 2;
    info2.current_prefix = {10, 20};
    infos.push_back(info2);

    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // vocab_size=50，在 CUDA 上分配
    const size_t vocab_size = 50;
    auto sampler_inputs = allocateSamplerInputs(3, vocab_size, processor);
    // 每行填充为 1.0，方便观察哪些位置被置为 -inf
    sampler_inputs.logits.fill_(1.0f);

    processor->process(sampler_inputs, 0, 3);

    auto result = sampler_inputs.logits.cpu();
    auto data0 = result[0].data_ptr<float>();
    auto data1 = result[1].data_ptr<float>();
    auto data2 = result[2].data_ptr<float>();

    // 序列 0：主序列，所有位置仍为 1.0（无 diverge，无 ban）
    for (size_t j = 0; j < vocab_size; ++j) {
        EXPECT_FLOAT_EQ(1.0f, data0[j]);
    }

    // 序列 1：diverge 遮蔽 top-1（所有值都是 1.0，topk 选出第一个），应有恰好 1 个位置被置为 -inf
    int inf_count_seq1 = 0;
    for (size_t j = 0; j < vocab_size; ++j) {
        if (std::isinf(data1[j]) && data1[j] < 0) inf_count_seq1++;
    }
    EXPECT_EQ(1, inf_count_seq1);  // top-1 遮蔽

    // 序列 2：ban 屏蔽 token 30（前缀 {10,20} 匹配 banned combo {10,20,30}）
    // 序列 2 在 pos=2 不触发 diverge（diverge 仅在 pos=0），仅触发 ban
    EXPECT_TRUE(std::isinf(data2[30]) && data2[30] < 0);
    // 其他位置不受影响
    EXPECT_FLOAT_EQ(1.0f, data2[29]);
    EXPECT_FLOAT_EQ(1.0f, data2[31]);
}

// 场景 14：updateMultiSeqStatus 与 cross-sequence ban 互斥——启用时调用应触发 assert
TEST_F(RecommendationLogitsProcessorTest, testUpdateMultiSeqStatusRejectsWhenCrossSeqBanEnabled) {
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> empty_set;
    // enable_cross_sequence_ban = true
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, true, 0));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // updateMultiSeqStatus 在 cross_sequence_ban 启用时应触发断言失败
    EXPECT_ANY_THROW(processor->updateMultiSeqStatus({0, 1}));
}

// 场景 15：updateMultiSeqStatus 在关闭 cross-sequence ban 时正常工作
TEST_F(RecommendationLogitsProcessorTest, testUpdateMultiSeqStatusWorksWhenCrossSeqBanDisabled) {
    std::vector<StreamRecommendationInfo> infos;
    std::set<std::vector<int>> empty_set;
    // enable_cross_sequence_ban = false
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, false, 0));
    infos.push_back(StreamRecommendationInfo(3, 0, 0, false, empty_set, {}, false, 0));
    auto processor = std::make_shared<RecommendationLogitsProcessor>(infos);

    // 关闭时正常重排，不抛异常
    EXPECT_NO_THROW(processor->updateMultiSeqStatus({1, 0}));
    // 验证重排生效（序列交换）
    EXPECT_EQ(processor->infos().size(), 2u);
}

// 场景 16：通过 fromGenerateInput 走生产路径（needs_token_offset=true, enable_cross_seq_ban=true）
// 验证 updateStatus 在 offset = current_output_length + input_length 路径下正确工作
TEST_F(RecommendationLogitsProcessorTest, testFromGenerateInputProductionPath) {
    // 配置：combo_token_size=3, num_return_sequences=3, enable_cross_sequence_ban=true
    // 生产中 needs_token_offset = hasNumBeams() || num_return_sequences > 1 = true
    // 注意：needs_token_offset 与 enable_cross_sequence_ban 不互斥，
    // 互斥的是 hasNumBeams()（即真正的 beam search），这里 num_return_sequences>1 触发 offset 但不是 beam search。
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->combo_token_size          = 3;
    generate_input->generate_config->num_return_sequences      = 3;
    generate_input->generate_config->enable_cross_sequence_ban = true;
    generate_input->generate_config->cross_seq_diverge_start_combo = 0;
    generate_input->generate_config->banned_combo_token_ids    = {};
    generate_input->input_ids = torch::zeros({4}, torch::kInt32);  // input_length=4

    auto p = RecommendationLogitsProcessor::fromGenerateInput(generate_input, 3);
    ASSERT_NE(nullptr, p);
    ASSERT_EQ(3u, p->size());

    // 验证 needs_token_offset=true 和 enable_cross_sequence_ban=true
    for (const auto& info : p->infos()) {
        EXPECT_TRUE(info.needs_token_offset);
        EXPECT_TRUE(info.enable_cross_sequence_ban);
        EXPECT_EQ(4, info.input_length);
    }

    // 模拟 updateStatus （needs_token_offset=true 路径）
    // tensor 形状 [N, input_length + total_output_length]
    // 第一步：output_length=0，offset=0+4=4，张量的第 4 列是新 token
    int input_len = 4;
    auto tokens_step1 = torch::zeros({3, input_len + 1}, torch::kInt32);
    // 序列 0 生成 token 10，序列 1 生成 token 20，序列 2 生成 token 30
    tokens_step1[0][input_len] = 10;
    tokens_step1[1][input_len] = 20;
    tokens_step1[2][input_len] = 30;
    p->updateStatus(tokens_step1, 1);

    // 每个序列 pos_in_combo 应该前进到 1
    for (const auto& info : p->infos()) {
        EXPECT_EQ(1, info.pos_in_combo);
    }

    // 第二步：output_length=1，offset=1+4=5
    auto tokens_step2 = torch::zeros({3, input_len + 2}, torch::kInt32);
    tokens_step2[0][input_len + 1] = 11;
    tokens_step2[1][input_len + 1] = 21;
    tokens_step2[2][input_len + 1] = 31;
    p->updateStatus(tokens_step2, 1);

    for (const auto& info : p->infos()) {
        EXPECT_EQ(2, info.pos_in_combo);
    }

    // 第三步：output_length=2，offset=2+4=6，combo 完成
    auto tokens_step3 = torch::zeros({3, input_len + 3}, torch::kInt32);
    tokens_step3[0][input_len + 2] = 12;
    tokens_step3[1][input_len + 2] = 22;
    tokens_step3[2][input_len + 2] = 32;
    p->updateStatus(tokens_step3, 1);

    // combo 完成，非对称广播：序列 0 仅有自己的，序列 1/2 有所有的
    EXPECT_EQ(1u, p->infos()[0].banned_combos.size());
    EXPECT_EQ(3u, p->infos()[1].banned_combos.size());
    EXPECT_EQ(3u, p->infos()[2].banned_combos.size());
    EXPECT_TRUE(p->infos()[0].banned_combos.count({10, 11, 12}));
    EXPECT_TRUE(p->infos()[1].banned_combos.count({10, 11, 12}));
    EXPECT_TRUE(p->infos()[1].banned_combos.count({20, 21, 22}));
    EXPECT_TRUE(p->infos()[1].banned_combos.count({30, 31, 32}));
}

// 场景 17：top-K 遮蔽深度上界保护 —— num_return_sequences=12 时，序列 11 的遮蔽深度
// 应被 kMaxDivergeDepth(=8) 钳制，至少保留足够可选 token
TEST_F(RecommendationLogitsProcessorTest, testDivergeDepthCappedByMaxLimit) {
    const int N = 12;  // 超过 kMaxDivergeDepth=8
    const int combo_size = 2;
    const int vocab = 20;

    std::vector<StreamRecommendationInfo> infos;
    for (int i = 0; i < N; ++i) {
        StreamRecommendationInfo info(combo_size, 0, 0, false, {}, {},
                                      /*enable_cross_sequence_ban=*/true,
                                      /*cross_seq_diverge_start_combo=*/0);
        infos.push_back(std::move(info));
    }
    auto p = std::make_shared<RecommendationLogitsProcessor>(std::move(infos));

    // 通过 allocateSamplerInputs 在 CUDA 上分配，与其他测试保持一致
    auto sampler_inputs = allocateSamplerInputs(N, vocab, p);
    // 填充 logits: 让 token 0~11 有高分，方便观察遮蔽
    sampler_inputs.logits.fill_(1.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sampler_inputs.logits[i][j] = 100.0f - j;
        }
    }

    p->process(sampler_inputs, 0, N);

    auto logits_cpu = sampler_inputs.logits.cpu();
    // 序列 0 不应被遮蔽
    for (int j = 0; j < vocab; ++j) {
        EXPECT_GT(logits_cpu[0][j].item<float>(), -1e30);
    }
    // 序列 11(i=11)：k = min(11, vocab-1=19, kMaxDivergeDepth=8) = 8
    // 应恰好有 8 个 token 被遮蔽为 -inf
    int masked_count = 0;
    for (int j = 0; j < vocab; ++j) {
        if (logits_cpu[11][j].item<float>() < -1e30) {
            masked_count++;
        }
    }
    EXPECT_EQ(8, masked_count);  // 受 kMaxDivergeDepth 钳制，而非 11

    // 序列 5(i=5)：k = min(5, 19, 8) = 5，正常不受 cap 影响
    int masked_count_5 = 0;
    for (int j = 0; j < vocab; ++j) {
        if (logits_cpu[5][j].item<float>() < -1e30) {
            masked_count_5++;
        }
    }
    EXPECT_EQ(5, masked_count_5);
}

// 场景 18：num_new_tokens > 1 批量推进（speculative decoding 场景）
// 一次推 3 个 token，combo_size=2，同一序列在一次 updateStatus 中完成 1 个 combo 并开始下一个
TEST_F(RecommendationLogitsProcessorTest, testBatchTokenAdvance) {
    const int N = 2;
    const int combo_size = 2;
    std::set<std::vector<int>> empty_set;

    std::vector<StreamRecommendationInfo> infos;
    for (int i = 0; i < N; ++i) {
        StreamRecommendationInfo info(combo_size, 0, 0, false, empty_set, {},
                                      /*enable_cross_sequence_ban=*/true,
                                      /*cross_seq_diverge_start_combo=*/0);
        infos.push_back(std::move(info));
    }
    auto p = std::make_shared<RecommendationLogitsProcessor>(std::move(infos));

    // 一次推 3 个 token: [A, B, C]
    // combo_size=2: token A+B 组成第 1 个 combo，token C 是第 2 个 combo 的第 1 位
    auto tokens = torch::zeros({N, 3}, torch::kInt32);
    tokens[0][0] = 10; tokens[0][1] = 11; tokens[0][2] = 20;
    tokens[1][0] = 30; tokens[1][1] = 31; tokens[1][2] = 40;

    p->updateStatus(tokens, 3);  // num_new_tokens=3

    // 序列 0: combo [10,11] 完成，然后 token 20 开始新 combo→pos_in_combo=1
    EXPECT_EQ(1, p->infos()[0].pos_in_combo);
    EXPECT_EQ(1, p->infos()[0].completed_combo_count);
    EXPECT_TRUE(p->infos()[0].banned_combos.count({10, 11}));

    // 序列 1: combo [30,31] 完成，然后 token 40 开始新 combo→pos_in_combo=1
    EXPECT_EQ(1, p->infos()[1].pos_in_combo);
    EXPECT_EQ(1, p->infos()[1].completed_combo_count);
    EXPECT_TRUE(p->infos()[1].banned_combos.count({30, 31}));

    // 跨序列广播验证：序列 1 应收到序列 0 的 combo [10,11]
    EXPECT_TRUE(p->infos()[1].banned_combos.count({10, 11}));
    // 序列 0 不接收序列 1 的 combo（primary-protected）
    EXPECT_FALSE(p->infos()[0].banned_combos.count({30, 31}));
}

// 场景 19：num_new_tokens=4，combo_size=2，同一序列一次 updateStatus 完成 2 个完整 combo
TEST_F(RecommendationLogitsProcessorTest, testBatchTokenMultipleCombos) {
    const int N = 2;
    const int combo_size = 2;
    std::set<std::vector<int>> empty_set;

    std::vector<StreamRecommendationInfo> infos;
    for (int i = 0; i < N; ++i) {
        StreamRecommendationInfo info(combo_size, 0, 0, false, empty_set, {},
                                      /*enable_cross_sequence_ban=*/true,
                                      /*cross_seq_diverge_start_combo=*/0);
        infos.push_back(std::move(info));
    }
    auto p = std::make_shared<RecommendationLogitsProcessor>(std::move(infos));

    // 一次推 4 个 token: combo_size=2 → 完成 2 个 combo
    // 序列 0: [A,B,C,D] → combo1=[A,B], combo2=[C,D]
    auto tokens = torch::zeros({N, 4}, torch::kInt32);
    tokens[0][0] = 1; tokens[0][1] = 2; tokens[0][2] = 3; tokens[0][3] = 4;
    tokens[1][0] = 5; tokens[1][1] = 6; tokens[1][2] = 7; tokens[1][3] = 8;

    p->updateStatus(tokens, 4);

    // 序列 0: 2 个 combo 完成，pos_in_combo=0 (刚好在边界上)
    EXPECT_EQ(0, p->infos()[0].pos_in_combo);
    EXPECT_EQ(2, p->infos()[0].completed_combo_count);
    EXPECT_TRUE(p->infos()[0].banned_combos.count({1, 2}));
    EXPECT_TRUE(p->infos()[0].banned_combos.count({3, 4}));

    // 序列 1: 同样 2 个 combo
    EXPECT_EQ(0, p->infos()[1].pos_in_combo);
    EXPECT_EQ(2, p->infos()[1].completed_combo_count);
    EXPECT_TRUE(p->infos()[1].banned_combos.count({5, 6}));
    EXPECT_TRUE(p->infos()[1].banned_combos.count({7, 8}));

    // 跨序列广播：序列 1 应收到序列 0 的两个 combo
    EXPECT_TRUE(p->infos()[1].banned_combos.count({1, 2}));
    EXPECT_TRUE(p->infos()[1].banned_combos.count({3, 4}));
    // 序列 0 不接收序列 1 的
    EXPECT_FALSE(p->infos()[0].banned_combos.count({5, 6}));
}

// 场景 20：think prelude 未完成时不进 combo，完成后正常进 combo + 跨序列广播
TEST_F(RecommendationLogitsProcessorTest, testThinkPreludeWithCrossSeqBan) {
    const int N = 2;
    const int combo_size = 2;
    std::set<std::vector<int>> empty_set;
    // end_think_token_ids = {99, 100}
    std::vector<int> end_think_ids = {99, 100};

    std::vector<StreamRecommendationInfo> infos;
    for (int i = 0; i < N; ++i) {
        StreamRecommendationInfo info(combo_size, 0, 0, false, empty_set, end_think_ids,
                                      /*enable_cross_sequence_ban=*/true,
                                      /*cross_seq_diverge_start_combo=*/0);
        infos.push_back(std::move(info));
    }
    auto p = std::make_shared<RecommendationLogitsProcessor>(std::move(infos));

    // 第 1 步：推送非 end_think token，think 未完成，combo 不应推进
    auto tokens1 = torch::zeros({N, 1}, torch::kInt32);
    tokens1[0][0] = 50;  tokens1[1][0] = 50;
    p->updateStatus(tokens1, 1);
    EXPECT_EQ(0, p->infos()[0].pos_in_combo);  // combo 未推进
    EXPECT_EQ(0, p->infos()[0].completed_combo_count);
    EXPECT_FALSE(p->infos()[0].think_done);

    // 第 2 步：推送 end_think 序列的第一个 token (99)
    auto tokens2 = torch::zeros({N, 1}, torch::kInt32);
    tokens2[0][0] = 99;  tokens2[1][0] = 99;
    p->updateStatus(tokens2, 1);
    EXPECT_FALSE(p->infos()[0].think_done);  // 仅匹配一半

    // 第 3 步：推送 end_think 序列的第二个 token (100)，think 完成
    auto tokens3 = torch::zeros({N, 1}, torch::kInt32);
    tokens3[0][0] = 100;  tokens3[1][0] = 100;
    p->updateStatus(tokens3, 1);
    EXPECT_TRUE(p->infos()[0].think_done);
    EXPECT_EQ(0, p->infos()[0].pos_in_combo);  // combo 仍未开始

    // 第 4-5 步：think 完成后推送 combo token，应形成完整 combo 并广播
    auto tokens4 = torch::zeros({N, 1}, torch::kInt32);
    tokens4[0][0] = 10;  tokens4[1][0] = 30;
    p->updateStatus(tokens4, 1);
    EXPECT_EQ(1, p->infos()[0].pos_in_combo);  // combo 推进到第 1 位

    auto tokens5 = torch::zeros({N, 1}, torch::kInt32);
    tokens5[0][0] = 11;  tokens5[1][0] = 31;
    p->updateStatus(tokens5, 1);
    EXPECT_EQ(0, p->infos()[0].pos_in_combo);  // combo 完成，复位
    EXPECT_EQ(1, p->infos()[0].completed_combo_count);
    EXPECT_TRUE(p->infos()[0].banned_combos.count({10, 11}));
    EXPECT_TRUE(p->infos()[1].banned_combos.count({30, 31}));
    // 跨序列广播
    EXPECT_TRUE(p->infos()[1].banned_combos.count({10, 11}));
    // primary-protected: 序列 0 不接收序列 1
    EXPECT_FALSE(p->infos()[0].banned_combos.count({30, 31}));
}

// 场景 21：安全降级路径——diverge + banned combo 叠加导致某行全 -inf，验证回退均匀分布
TEST_F(RecommendationLogitsProcessorTest, testSafetyFallbackAllMasked) {
    // 构造极端场景：vocab_size=3，combo_size=1，N=2
    // combo_size=1 时 pos_in_combo==0 同时满足 diverge条件(pos==0) 和 ban条件(pos==combo_size-1==0)
    // 序列 1 的 banned_combos 将封锁 token 1 和 2，diverge 封锁 topk=1 (token 0)
    // 导致序列 1 全部 3 个 token 都被遮蔽
    const int N = 2;
    const size_t vocab_size = 3;
    const int combo_size = 1;
    std::set<std::vector<int>> empty_set;
    // banned combo: {1} 和 {2} (combo_size=1，单元素 combo)
    std::set<std::vector<int>> banned = {{1}, {2}};

    std::vector<StreamRecommendationInfo> infos;
    // 序列 0: 主序列，无 ban，pos_in_combo=0
    StreamRecommendationInfo info0(combo_size, 0, 0, false, empty_set, {},
                                    /*enable_cross_sequence_ban=*/true,
                                    /*cross_seq_diverge_start_combo=*/0);
    infos.push_back(std::move(info0));

    // 序列 1: banned={1},{2}，pos_in_combo=0，completed_combo_count=1 (触发 diverge)
    StreamRecommendationInfo info1(combo_size, 0, 0, false, banned, {},
                                    /*enable_cross_sequence_ban=*/true,
                                    /*cross_seq_diverge_start_combo=*/0);
    info1.completed_combo_count = 1;
    infos.push_back(std::move(info1));

    auto processor = std::make_shared<RecommendationLogitsProcessor>(std::move(infos));
    auto inputs = allocateSamplerInputs(N, vocab_size, processor);

    // 初始 logits 全部为 1.0
    inputs.logits.fill_(1.0f);

    processor->process(inputs, 0, N);

    // 序列 1 应该触发安全降级：全部回退为 0.0 (均匀分布)
    auto logits_cpu = inputs.logits.cpu();
    auto acc = logits_cpu.accessor<float, 2>();
    for (size_t v = 0; v < vocab_size; ++v) {
        EXPECT_FLOAT_EQ(0.0f, acc[1][v]);
    }
    // 序列 0 不应被影响（主序列无 diverge，无 ban）
    for (size_t v = 0; v < vocab_size; ++v) {
        EXPECT_FLOAT_EQ(1.0f, acc[0][v]);
    }
}

// 场景 22：num_return_sequences=1 + enable_cross_sequence_ban=true 的 no-op 边界
TEST_F(RecommendationLogitsProcessorTest, testSingleSequenceCrossSeqBanNoOp) {
    // N=1 时 updateStatus 跳过广播(size()>1 不满足)，process 跳过 diverge(i>0 不满足)
    const int N = 1;
    const size_t vocab_size = 10;
    const int combo_size = 2;
    std::set<std::vector<int>> banned = {{1, 2}};

    std::vector<StreamRecommendationInfo> infos;
    StreamRecommendationInfo info(combo_size, 0, 0, false, banned, {},
                                   /*enable_cross_sequence_ban=*/true,
                                   /*cross_seq_diverge_start_combo=*/0);
    info.pos_in_combo = 1;
    info.current_prefix = {1};
    info.completed_combo_count = 1;
    infos.push_back(std::move(info));

    auto processor = std::make_shared<RecommendationLogitsProcessor>(std::move(infos));
    auto inputs = allocateSamplerInputs(N, vocab_size, processor);
    inputs.logits.fill_(1.0f);

    processor->process(inputs, 0, N);

    // ban 应该正常工作：token 2 被封锁
    auto logits_cpu = inputs.logits.cpu();
    auto acc = logits_cpu.accessor<float, 2>();
    EXPECT_FLOAT_EQ(-std::numeric_limits<float>::infinity(), acc[0][2]);
    // 其他 token 不受影响
    EXPECT_FLOAT_EQ(1.0f, acc[0][0]);
    EXPECT_FLOAT_EQ(1.0f, acc[0][1]);
    EXPECT_FLOAT_EQ(1.0f, acc[0][3]);

    // updateStatus 不应崩溃，且无广播发生
    auto tokens = torch::zeros({N, 1}, torch::kInt32);
    tokens[0][0] = 2;  // 完成 combo [1,2]
    processor->updateStatus(tokens, 1);
    EXPECT_EQ(2, processor->infos()[0].completed_combo_count);
    EXPECT_TRUE(processor->infos()[0].banned_combos.count({1, 2}));
}

}  // namespace rtp_llm
