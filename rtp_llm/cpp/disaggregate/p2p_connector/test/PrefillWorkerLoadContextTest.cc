#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillWorkerLoadContext.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class PrefillWorkerLoadContextTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // 创建测试用的 AsymmetricTPContext
    std::vector<AsymmetricTPContext> createAsymmetricTPContexts(int count) {
        std::vector<AsymmetricTPContext> contexts;
        for (int i = 0; i < count; ++i) {
            contexts.emplace_back("192.168.1." + std::to_string(10 + i), 8080 + i, 1, 0, 1, 0);
        }
        return contexts;
    }

    // 获取当前时间（毫秒）+ 偏移量
    int64_t getDeadlineMs(int64_t offset_ms = 1000) {
        return currentTimeMs() + offset_ms;
    }

protected:
};

// ==================== PrefillWorkerLoadContext 测试 ====================

// 测试构造函数和基本属性
TEST_F(PrefillWorkerLoadContextTest, Constructor) {
    int64_t     request_id             = 1001;
    std::string unique_key             = "test_key_1";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(2);
    int         num_layers             = 3;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    EXPECT_EQ(context.requestId(), request_id);
    EXPECT_EQ(context.asymmetricTPContexts().size(), 2);
    EXPECT_FALSE(context.isDone());
    EXPECT_FALSE(context.isCanceled());
    EXPECT_FALSE(context.isTimeout());
    EXPECT_TRUE(context.isAllSuccess());

    // 初始时所有 layer 都需要传输
    EXPECT_TRUE(context.needTransfer(0));
    EXPECT_TRUE(context.needTransfer(1));
    EXPECT_TRUE(context.needTransfer(2));
    EXPECT_FALSE(context.needTransfer(3));  // 不存在的 layer

    // 开始传输 layer 0
    context.startTransfer(0);
    EXPECT_FALSE(context.needTransfer(0));
    EXPECT_TRUE(context.needTransfer(1));
    EXPECT_TRUE(context.needTransfer(2));

    // 完成所有传输
    // id = layer_id * asymmetric_tp_contexts.size()
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
        for (size_t ctx_idx = 0; ctx_idx < asymmetric_tp_contexts.size(); ++ctx_idx) {
            int id = layer_id * static_cast<int>(asymmetric_tp_contexts.size()) + static_cast<int>(ctx_idx);
            context.notifyDone(id, true);
        }
    }

    EXPECT_TRUE(context.isDone());
}

// 测试 notifyDone 的 success 参数
TEST_F(PrefillWorkerLoadContextTest, NotifyDoneSuccess) {
    int64_t     request_id             = 1004;
    std::string unique_key             = "test_key_4";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(1);
    int         num_layers             = 2;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    EXPECT_TRUE(context.isAllSuccess());

    // 第一个传输成功
    context.notifyDone(0, true);
    EXPECT_TRUE(context.isAllSuccess());

    // 第二个传输失败
    context.notifyDone(1, false);
    EXPECT_FALSE(context.isAllSuccess());
}

// 测试 setCanceled 和 isCanceled
TEST_F(PrefillWorkerLoadContextTest, SetCanceled) {
    int64_t     request_id             = 1005;
    std::string unique_key             = "test_key_5";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(1);
    int         num_layers             = 2;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    EXPECT_FALSE(context.isCanceled());

    context.setCanceled();
    EXPECT_TRUE(context.isCanceled());
}

// 测试 isTimeout
TEST_F(PrefillWorkerLoadContextTest, IsTimeout) {
    int64_t     request_id             = 1006;
    std::string unique_key             = "test_key_6";
    int64_t     deadline_ms            = getDeadlineMs(100);  // 100ms 后过期
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(1);
    int         num_layers             = 2;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    // 立即检查，应该未超时
    EXPECT_FALSE(context.isTimeout());

    // 等待 150ms 后检查，应该超时
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_TRUE(context.isTimeout());
}

// 测试 isAllTransfersDone
TEST_F(PrefillWorkerLoadContextTest, IsAllTransfersDone) {
    int64_t     request_id             = 1007;
    std::string unique_key             = "test_key_7";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(2);  // 2 个 context
    int         num_layers             = 3;                              // 3 个 layer

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    // 初始时没有传输未完成
    EXPECT_TRUE(context.isAllTransfersDone());

    // 开始传输 layer 0，但还没有完成
    context.startTransfer(0);
    EXPECT_FALSE(context.isAllTransfersDone());

    // 完成 layer 0 的所有传输 (2 个)
    context.notifyDone(0, true);  // layer 0, context 0
    context.notifyDone(1, true);  // layer 0, context 1
    // 此时 transferred_ids_.size() = 2
    // need_transfer_layer_ids_.size() = 2 (layer 1, 2)
    // (num_layers - need_transfer_layer_ids_.size()) * asymmetric_tp_contexts.size() = (3 - 2) * 2 = 2
    EXPECT_TRUE(context.isAllTransfersDone());

    // 开始传输 layer 1
    context.startTransfer(1);
    // 此时 need_transfer_layer_ids_.size() = 1 (layer 2)
    // (num_layers - need_transfer_layer_ids_.size()) * asymmetric_tp_contexts.size() = (3 - 1) * 2 = 4
    // transferred_ids_.size() = 2 < 4，所以应该返回 false
    EXPECT_FALSE(context.isAllTransfersDone());

    // 完成 layer 1 的所有传输
    context.notifyDone(2, true);  // layer 1, context 0
    context.notifyDone(3, true);  // layer 1, context 1
    // 此时 transferred_ids_.size() = 4
    // need_transfer_layer_ids_.size() = 1 (layer 2)
    // (num_layers - need_transfer_layer_ids_.size()) * asymmetric_tp_contexts.size() = (3 - 1) * 2 = 4
    EXPECT_TRUE(context.isAllTransfersDone());
}

// ==================== PrefillWorkerLoadContextStore 测试 ====================

// 测试 addContext 和 getContext
TEST_F(PrefillWorkerLoadContextTest, StoreAddAndGetContext) {
    PrefillWorkerLoadContextStore store;

    int64_t     request_id             = 2001;
    std::string unique_key             = "test_key_store_1";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(2);
    int         num_layers             = 3;

    auto context = store.addContext(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->requestId(), request_id);

    auto retrieved = store.getContext(request_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->requestId(), request_id);
    EXPECT_EQ(retrieved, context);

    EXPECT_FALSE(store.getContext(9999) != nullptr);

    store.removeContext(request_id);
    retrieved = store.getContext(request_id);
    EXPECT_EQ(retrieved, nullptr);
}

// 测试 checkTimeout - 部分过期，部分未过期
TEST_F(PrefillWorkerLoadContextTest, StoreCheckTimeoutPartialExpired) {
    PrefillWorkerLoadContextStore store;

    int64_t     request_id1            = 4003;
    int64_t     request_id2            = 4004;
    int64_t     request_id3            = 4005;
    std::string unique_key1            = "test_key_store_6_1";
    std::string unique_key2            = "test_key_store_6_2";
    std::string unique_key3            = "test_key_store_6_3";
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(1);
    int         num_layers             = 2;

    // request_id1: 已过期
    int64_t deadline_ms1 = currentTimeMs() - 100;
    store.addContext(request_id1, unique_key1, deadline_ms1, asymmetric_tp_contexts, num_layers);

    // request_id2: 未过期
    int64_t deadline_ms2 = getDeadlineMs(1000);
    store.addContext(request_id2, unique_key2, deadline_ms2, asymmetric_tp_contexts, num_layers);

    // request_id3: 已过期
    int64_t deadline_ms3 = currentTimeMs() - 50;
    store.addContext(request_id3, unique_key3, deadline_ms3, asymmetric_tp_contexts, num_layers);

    // 检查超时
    store.checkTimeout();

    // 验证结果
    auto retrieved1 = store.getContext(request_id1);
    EXPECT_EQ(retrieved1, nullptr);

    auto retrieved2 = store.getContext(request_id2);
    ASSERT_NE(retrieved2, nullptr);
    EXPECT_EQ(retrieved2->requestId(), request_id2);

    auto retrieved3 = store.getContext(request_id3);
    EXPECT_EQ(retrieved3, nullptr);
}

// 测试完整的传输流程
TEST_F(PrefillWorkerLoadContextTest, CompleteTransferFlow) {
    int64_t     request_id             = 5001;
    std::string unique_key             = "test_key_complete";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(2);  // 2 个 context
    int         num_layers             = 3;                              // 3 个 layer

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    // 初始状态
    EXPECT_FALSE(context.isDone());
    EXPECT_FALSE(context.isCanceled());
    EXPECT_TRUE(context.isAllSuccess());
    EXPECT_TRUE(context.needTransfer(0));
    EXPECT_TRUE(context.needTransfer(1));
    EXPECT_TRUE(context.needTransfer(2));

    // 开始传输 layer 0
    context.startTransfer(0);
    EXPECT_FALSE(context.needTransfer(0));

    // 完成 layer 0 的所有传输
    context.notifyDone(0, true);  // layer 0, context 0
    context.notifyDone(1, true);  // layer 0, context 1

    // 开始传输 layer 1
    context.startTransfer(1);
    context.notifyDone(2, true);  // layer 1, context 0
    context.notifyDone(3, true);  // layer 1, context 1

    // 开始传输 layer 2
    context.startTransfer(2);
    context.notifyDone(4, true);  // layer 2, context 0
    context.notifyDone(5, true);  // layer 2, context 1

    // 所有传输完成
    EXPECT_TRUE(context.isDone());
    EXPECT_TRUE(context.isAllSuccess());
}

// 测试取消流程
TEST_F(PrefillWorkerLoadContextTest, CancelFlow) {
    int64_t     request_id             = 5002;
    std::string unique_key             = "test_key_cancel";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(1);
    int         num_layers             = 2;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);

    EXPECT_FALSE(context.isCanceled());

    // 开始传输
    context.startTransfer(0);

    // 取消
    context.setCanceled();
    EXPECT_TRUE(context.isCanceled());

    EXPECT_FALSE(context.isDone());
    EXPECT_FALSE(context.isAllTransfersDone());

    // 完成传输
    context.notifyDone(0, true);
}

}  // namespace rtp_llm
