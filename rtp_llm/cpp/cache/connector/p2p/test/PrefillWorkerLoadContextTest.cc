#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "rtp_llm/cpp/cache/connector/p2p/PrefillWorkerLoadContext.h"
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

TEST_F(PrefillWorkerLoadContextTest, basicTest) {
    int64_t     request_id     = 1001;
    std::string unique_key     = "test_key_1";
    int64_t     deadline_ms    = getDeadlineMs();
    int         transfer_count = 6;  // 2 contexts * 3 layers

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, transfer_count);

    // default value test
    EXPECT_EQ(context.requestId(), request_id);
    EXPECT_FALSE(context.done());
    EXPECT_FALSE(context.canceled());
    EXPECT_FALSE(context.timeout());
    EXPECT_FALSE(context.success());  // 还未完成，所以 success 是 false

    ASSERT_EQ(context.getNeedTransferIds().size(), 6);

    EXPECT_TRUE(context.startTransfer(0));
    EXPECT_EQ(context.getNeedTransferIds().size(), transfer_count - 1);
    EXPECT_FALSE(context.getNeedTransferIds().count(0) > 0);

    // 尝试再次开始传输 id 0 应该失败
    EXPECT_FALSE(context.startTransfer(0));

    for (int i = 1; i < transfer_count; ++i) {
        context.startTransfer(i);
    }
    ASSERT_TRUE(context.isAllTransferStarted());

    context.notifyDone(0, ErrorCode::NONE_ERROR, "");
    for (int i = 1; i < transfer_count; ++i) {
        ASSERT_FALSE(context.isAllTransfersDone());
        context.notifyDone(i, ErrorCode::NONE_ERROR, "");
    }
    ASSERT_TRUE(context.isAllTransfersDone());
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
}

TEST_F(PrefillWorkerLoadContextTest, SetCanceled) {
    int64_t     request_id     = 1005;
    std::string unique_key     = "test_key_5";
    int64_t     deadline_ms    = getDeadlineMs();
    int         transfer_count = 1;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, transfer_count);

    ASSERT_EQ(context.getNeedTransferIds().size(), 1);
    EXPECT_FALSE(context.canceled());

    EXPECT_TRUE(context.startTransfer(0));
    EXPECT_EQ(context.getNeedTransferIds().size(), transfer_count - 1);
    EXPECT_FALSE(context.getNeedTransferIds().count(0) > 0);

    context.setCanceled();
    EXPECT_TRUE(context.canceled());

    EXPECT_TRUE(context.isAllTransferStarted());
    EXPECT_FALSE(context.isAllTransfersDone());

    context.notifyDone(0, ErrorCode::NONE_ERROR, "");
    EXPECT_TRUE(context.isAllTransfersDone());
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
}

// 测试 timeout
TEST_F(PrefillWorkerLoadContextTest, IsTimeout) {
    int64_t     request_id     = 1006;
    std::string unique_key     = "test_key_6";
    int64_t     deadline_ms    = getDeadlineMs(100);  // 100ms 后过期
    int         transfer_count = 2;

    PrefillWorkerLoadContext context(request_id, unique_key, deadline_ms, transfer_count);

    // 立即检查，应该未超时
    EXPECT_FALSE(context.timeout());

    // 等待 150ms 后检查，应该超时
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_TRUE(context.timeout());
}

// ==================== PrefillWorkerLoadContextStore 测试 ====================

// 测试 addContext 和 getContext
TEST_F(PrefillWorkerLoadContextTest, StoreAddAndGetContext) {
    PrefillWorkerLoadContextStore store;

    EXPECT_EQ(store.getContextsCount(), 0);

    int64_t     request_id             = 2001;
    std::string unique_key             = "test_key_store_1";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(2);
    int         num_layers             = 3;

    auto context = store.addContext(request_id, unique_key, deadline_ms, asymmetric_tp_contexts, num_layers);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->requestId(), request_id);
    EXPECT_EQ(store.getContextsCount(), 1);

    // 验证 transfer_count = asymmetric_tp_contexts.size() * num_layers
    EXPECT_EQ(context->getNeedTransferIds().size(), 6);

    auto retrieved = store.getContext(request_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->requestId(), request_id);
    EXPECT_EQ(retrieved, context);

    EXPECT_EQ(store.getContext(9999), nullptr);

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

    EXPECT_EQ(store.getContextsCount(), 3);

    // 检查超时
    store.checkTimeout();

    EXPECT_EQ(store.getContextsCount(), 1);

    // 验证结果
    auto retrieved1 = store.getContext(request_id1);
    EXPECT_EQ(retrieved1, nullptr);

    auto retrieved2 = store.getContext(request_id2);
    ASSERT_NE(retrieved2, nullptr);
    EXPECT_EQ(retrieved2->requestId(), request_id2);

    auto retrieved3 = store.getContext(request_id3);
    EXPECT_EQ(retrieved3, nullptr);
}

// ==================== cancelByUniqueKey 测试 ====================

// 测试 cancelByUniqueKey - context 不存在
TEST_F(PrefillWorkerLoadContextTest, StoreCancelByUniqueKey_ContextNotFound) {
    PrefillWorkerLoadContextStore store;

    // 不添加任何 context，直接调用 cancelByUniqueKey
    bool result = store.cancelByUniqueKey("non_existent_unique_key");

    // 验证返回 true（因为 cancel 是 best-effort）
    EXPECT_TRUE(result);
}

// 测试 cancelByUniqueKey - 多个 context，取消指定的一个
TEST_F(PrefillWorkerLoadContextTest, StoreCancelByUniqueKey_MultipleContexts) {
    PrefillWorkerLoadContextStore store;

    int64_t     request_id1            = 5002;
    int64_t     request_id2            = 5003;
    std::string unique_key1            = "test_cancel_multi_1";
    std::string unique_key2            = "test_cancel_multi_2";
    int64_t     deadline_ms            = getDeadlineMs();
    auto        asymmetric_tp_contexts = createAsymmetricTPContexts(1);
    int         num_layers             = 2;

    auto context1 = store.addContext(request_id1, unique_key1, deadline_ms, asymmetric_tp_contexts, num_layers);
    auto context2 = store.addContext(request_id2, unique_key2, deadline_ms, asymmetric_tp_contexts, num_layers);

    ASSERT_NE(context1, nullptr);
    ASSERT_NE(context2, nullptr);
    EXPECT_FALSE(context1->canceled());
    EXPECT_FALSE(context2->canceled());

    // 只取消 unique_key1 对应的 context
    bool result = store.cancelByUniqueKey(unique_key1);

    // 验证只有 context1 被取消
    EXPECT_TRUE(result);
    EXPECT_TRUE(context1->canceled());
    EXPECT_FALSE(context2->canceled());
}

}  // namespace rtp_llm
