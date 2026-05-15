#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"

namespace rtp_llm {

class P2PConnectorResourceStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        stream_store_ = std::make_unique<P2PConnectorResourceStore>(nullptr, 100);
        ASSERT_TRUE(stream_store_->init());
    }

    void TearDown() override {
        stream_store_.reset();
    }

    // Helper to get current time + offset
    int64_t getDeadlineMs(int64_t offset_ms = 5000) {
        return currentTimeMs() + offset_ms;
    }

    /// Create a MockMeta with routing context configured
    std::shared_ptr<MockMeta> createMockMeta(const std::string& unique_key, int64_t request_id, int64_t deadline_ms) {
        auto meta = std::make_shared<MockMeta>();
        meta->setUniqueKey(unique_key);
        meta->setRequestId(request_id);
        meta->setDeadlineMs(deadline_ms);
        meta->setPrefillAddr("127.0.0.1", 12345);
        meta->setPrefillTpSize(1);
        return meta;
    }

    // Create a mock KV cache resource
    KVCacheResourcePtr createMockKVCacheResource() {
        return std::make_shared<KVCacheResource>();
    }

protected:
    std::unique_ptr<P2PConnectorResourceStore> stream_store_;
};

// ==================== addResource & stealResource 基础测试 ====================

TEST_F(P2PConnectorResourceStoreTest, AddAndStealResource_Success) {
    std::string unique_key  = "test_key_1";
    int64_t     request_id  = 1001;
    int64_t     deadline_ms = getDeadlineMs();
    auto        meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto        resource    = createMockKVCacheResource();

    stream_store_->addResource(meta, resource);

    // Steal resource (use waitAndStealResource with current time as deadline for immediate return)
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100);

    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->request_id, request_id);
    EXPECT_EQ(entry->kv_cache_resource, resource);
    EXPECT_EQ(entry->deadline_ms, deadline_ms);
}

TEST_F(P2PConnectorResourceStoreTest, StealResource_NotFound) {
    // Use short timeout to avoid waiting
    auto entry = stream_store_->waitAndStealResource("non_existent_key", currentTimeMs() + 10);

    EXPECT_EQ(entry, nullptr);
}

TEST_F(P2PConnectorResourceStoreTest, StealResource_CanOnlyStealOnce) {
    std::string unique_key  = "test_key_2";
    int64_t     request_id  = 1002;
    int64_t     deadline_ms = getDeadlineMs();
    auto        meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto        resource    = createMockKVCacheResource();

    stream_store_->addResource(meta, resource);

    // First steal should succeed
    auto entry1 = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100);
    ASSERT_NE(entry1, nullptr);

    // Second steal should fail (already removed)
    auto entry2 = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 10);
    EXPECT_EQ(entry2, nullptr);
}

// ==================== waitAndStealResource 测试 ====================

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_ImmediateReturn) {
    std::string unique_key  = "test_key_3";
    int64_t     request_id  = 1003;
    int64_t     deadline_ms = getDeadlineMs();
    auto        meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto        resource    = createMockKVCacheResource();

    stream_store_->addResource(meta, resource);

    // waitAndStealResource should return immediately since resource exists
    auto start_time = currentTimeMs();
    auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms);
    auto elapsed_ms = currentTimeMs() - start_time;

    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->request_id, request_id);
    EXPECT_LT(elapsed_ms, 100);  // Should return almost immediately
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_WaitForResource) {
    std::string unique_key  = "test_key_4";
    int64_t     request_id  = 1004;
    int64_t     deadline_ms = getDeadlineMs(5000);
    auto        meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto        resource    = createMockKVCacheResource();

    std::atomic<bool> resource_added{false};

    std::thread add_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        stream_store_->addResource(meta, resource);
        resource_added.store(true);
    });

    auto start_time = currentTimeMs();
    auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms);
    auto elapsed_ms = currentTimeMs() - start_time;

    add_thread.join();

    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->request_id, request_id);
    EXPECT_TRUE(resource_added.load());
    // Should wait for the resource (around 200ms, but allow some tolerance)
    EXPECT_GE(elapsed_ms, 150);
    EXPECT_LT(elapsed_ms, 1000);  // Should not wait too long
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_Timeout) {
    std::string unique_key  = "test_key_5";
    int64_t     deadline_ms = currentTimeMs() + 200;  // Short timeout

    auto start_time = currentTimeMs();
    auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms);
    auto elapsed_ms = currentTimeMs() - start_time;

    EXPECT_EQ(entry, nullptr);
    // Should wait until timeout (around 200ms, allow some tolerance)
    EXPECT_GE(elapsed_ms, 150);
    EXPECT_LT(elapsed_ms, 500);
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_PastDeadline_ReturnsNullImmediately) {
    const std::string unique_key  = "test_key_past_deadline";
    const int64_t     deadline_ms = currentTimeMs() - 1;

    const auto start_time = currentTimeMs();
    const auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms);
    const auto elapsed_ms = currentTimeMs() - start_time;

    EXPECT_EQ(entry, nullptr);
    EXPECT_LT(elapsed_ms, 50);
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_CancelledWhileWaiting_ReturnsNullWithoutSteal) {
    const std::string unique_key  = "test_key_cancel_while_wait";
    const int64_t     deadline_ms = currentTimeMs() + 10000;

    std::atomic<bool>     cancel{false};
    std::function<bool()> is_cancelled = [&cancel]() { return cancel.load(); };

    std::thread cancel_thread([&cancel]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cancel.store(true);
    });

    const auto start_time = currentTimeMs();
    const auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, is_cancelled);
    const auto elapsed_ms = currentTimeMs() - start_time;

    cancel_thread.join();

    EXPECT_EQ(entry, nullptr);
    EXPECT_GE(elapsed_ms, 80);
    EXPECT_LT(elapsed_ms, 2000);

    // Store 仍可用：补资源后应能正常 steal
    const int64_t request_id = 2001;
    auto          meta       = createMockMeta(unique_key, request_id, getDeadlineMs(5000));
    auto          resource   = createMockKVCacheResource();
    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry_after = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 500);
    ASSERT_NE(entry_after, nullptr);
    EXPECT_EQ(entry_after->request_id, request_id);
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_CancelledBeforeResourceAppears) {
    const std::string unique_key  = "test_key_cancel_before_add";
    const int64_t     request_id  = 2002;
    const int64_t     deadline_ms = getDeadlineMs(5000);
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    std::atomic<bool>     cancel{false};
    std::function<bool()> is_cancelled = [&cancel]() { return cancel.load(); };

    std::thread cancel_early([&cancel]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        cancel.store(true);
    });

    std::thread late_add([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        stream_store_->addResource(meta, resource);
    });

    const auto start_time = currentTimeMs();
    const auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, is_cancelled);
    const auto elapsed_ms = currentTimeMs() - start_time;

    cancel_early.join();
    late_add.join();

    EXPECT_EQ(entry, nullptr);
    EXPECT_LT(elapsed_ms, 250);

    // 资源已入 store，未被取消路径取走，应仍可 steal
    auto stolen = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 500);
    ASSERT_NE(stolen, nullptr);
    EXPECT_EQ(stolen->request_id, request_id);
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_OnlyWakeUpCorrectWaiter) {
    std::string unique_key_1 = "test_key_6_a";
    std::string unique_key_2 = "test_key_6_b";
    int64_t     request_id_1 = 1006;
    int64_t     request_id_2 = 1007;
    int64_t     deadline_ms  = getDeadlineMs(5000);
    auto        meta1        = createMockMeta(unique_key_1, request_id_1, deadline_ms);
    auto        meta2        = createMockMeta(unique_key_2, request_id_2, deadline_ms);
    auto        resource     = createMockKVCacheResource();

    std::atomic<bool>                          waiter1_done{false};
    std::atomic<bool>                          waiter2_done{false};
    std::shared_ptr<P2PConnectorResourceEntry> entry1;
    std::shared_ptr<P2PConnectorResourceEntry> entry2;

    // Start two waiters for different keys
    std::thread waiter1([&]() {
        entry1 = stream_store_->waitAndStealResource(unique_key_1, deadline_ms);
        waiter1_done.store(true);
    });

    std::thread waiter2([&]() {
        entry2 = stream_store_->waitAndStealResource(unique_key_2, deadline_ms);
        waiter2_done.store(true);
    });

    // Wait for both threads to start waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    stream_store_->addResource(meta1, resource);

    // Wait for waiter1 to complete
    waiter1.join();

    EXPECT_TRUE(waiter1_done.load());
    ASSERT_NE(entry1, nullptr);
    EXPECT_EQ(entry1->request_id, request_id_1);

    // waiter2 should still be waiting (or we add resource for it)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    stream_store_->addResource(meta2, resource);

    waiter2.join();

    EXPECT_TRUE(waiter2_done.load());
    ASSERT_NE(entry2, nullptr);
    EXPECT_EQ(entry2->request_id, request_id_2);
}

TEST_F(P2PConnectorResourceStoreTest, WaitAndStealResource_MultipleWaitersForSameKey) {
    std::string unique_key  = "test_key_7";
    int64_t     request_id  = 1008;
    int64_t     deadline_ms = getDeadlineMs(5000);
    auto        meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto        resource    = createMockKVCacheResource();

    std::atomic<int>                           success_count{0};
    std::shared_ptr<P2PConnectorResourceEntry> entry1;
    std::shared_ptr<P2PConnectorResourceEntry> entry2;

    // Start two waiters for the same key
    std::thread waiter1([&]() {
        entry1 = stream_store_->waitAndStealResource(unique_key, deadline_ms);
        if (entry1) {
            success_count.fetch_add(1);
        }
    });

    std::thread waiter2([&]() {
        entry2 = stream_store_->waitAndStealResource(unique_key, deadline_ms);
        if (entry2) {
            success_count.fetch_add(1);
        }
    });

    // Wait for both threads to start waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    stream_store_->addResource(meta, resource);

    waiter1.join();
    waiter2.join();

    // Only one waiter should get the resource (since steal removes it)
    EXPECT_EQ(success_count.load(), 1);
}

// ==================== 超时清理测试 ====================

TEST_F(P2PConnectorResourceStoreTest, ResourceTimeout_AutoRemoval) {
    std::string unique_key  = "test_key_8";
    int64_t     request_id  = 1009;
    int64_t     deadline_ms = currentTimeMs() + 50;
    auto        meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto        resource    = createMockKVCacheResource();

    stream_store_->addResource(meta, resource);

    // Wait for the timeout check to run (check interval is 100ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // Resource should have been removed due to timeout
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 10);
    EXPECT_EQ(entry, nullptr);
}

TEST_F(P2PConnectorResourceStoreTest, SideChannelTimeout_AutoRemoval) {
    const std::string unique_key  = "test_side_channel_timeout";
    const int64_t     deadline_ms = currentTimeMs() + 50;

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 42;
    stream_store_->notifySideChannelReady(unique_key, deadline_ms, side_data);

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    EXPECT_FALSE(stream_store_->consumeSideChannelData(unique_key, consumed_data));
}

TEST_F(P2PConnectorResourceStoreTest, ClearSideChannelData_RemovesIndependentEntry) {
    const std::string unique_key  = "test_side_channel_clear";
    const int64_t     deadline_ms = getDeadlineMs(5000);

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 7;
    stream_store_->notifySideChannelReady(unique_key, deadline_ms, side_data);
    stream_store_->clearSideChannelData(unique_key);

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    EXPECT_FALSE(stream_store_->consumeSideChannelData(unique_key, consumed_data));
}

TEST_F(P2PConnectorResourceStoreTest, StolenEntry_NotifySideChannelWithAbsoluteDeadline_PreservesData) {
    const std::string unique_key  = "test_side_channel_after_steal";
    const int64_t     request_id  = 1010;
    const int64_t     deadline_ms = currentTimeMs() + 1000;
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100);
    ASSERT_NE(entry, nullptr);

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 88;
    stream_store_->notifySideChannelReady(unique_key, deadline_ms, side_data);

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    ASSERT_TRUE(stream_store_->consumeSideChannelData(unique_key, consumed_data));
    EXPECT_TRUE(consumed_data.has_first_token);
    EXPECT_EQ(consumed_data.first_token_id, 88);
}

// ==================== markCancelled 测试 ====================

// markCancelled when resource is already in store → removes it immediately so blocks are freed
TEST_F(P2PConnectorResourceStoreTest, MarkCancelled_ResourceAlreadyInStore_RemovesImmediately) {
    const std::string unique_key  = "test_cancel_existing";
    const int64_t     request_id  = 3001;
    const int64_t     deadline_ms = getDeadlineMs(5000);
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    ASSERT_TRUE(stream_store_->addResource(meta, resource));

    // Cancel while resource is sitting in store
    stream_store_->markCancelled(unique_key);

    // Resource should be gone — steal fails immediately
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 50);
    EXPECT_EQ(entry, nullptr);
}

// markCancelled before resource arrives → addResource rejects the resource on arrival
TEST_F(P2PConnectorResourceStoreTest, MarkCancelled_ResourceNotYetInStore_RejectsSubsequentAdd) {
    const std::string unique_key  = "test_cancel_before_add";
    const int64_t     request_id  = 3002;
    const int64_t     deadline_ms = getDeadlineMs(5000);
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    // Cancel before prefill adds the resource
    stream_store_->markCancelled(unique_key);

    // Resource arrives later (prefill finished inference after decode already timed out)
    bool added = stream_store_->addResource(meta, resource);
    EXPECT_FALSE(added);

    // Resource should not be stealable
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 50);
    EXPECT_EQ(entry, nullptr);
}

// markCancelled is idempotent and does not block subsequent keys with different names
TEST_F(P2PConnectorResourceStoreTest, MarkCancelled_DoesNotAffectOtherKeys) {
    const std::string key_cancelled = "test_cancel_only_this";
    const std::string key_normal    = "test_cancel_other_key";
    const int64_t     request_id    = 3003;
    const int64_t     deadline_ms   = getDeadlineMs(5000);
    auto              meta          = createMockMeta(key_normal, request_id, deadline_ms);
    auto              resource      = createMockKVCacheResource();

    stream_store_->markCancelled(key_cancelled);

    // A different key should still work normally
    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry = stream_store_->waitAndStealResource(key_normal, currentTimeMs() + 200);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->request_id, request_id);
}

// After markCancelled rejects addResource(), the cancel record is consumed and the key can be reused
TEST_F(P2PConnectorResourceStoreTest, MarkCancelled_CancelRecordConsumedAfterRejection) {
    const std::string unique_key  = "test_cancel_record_consumed";
    const int64_t     request_id  = 3004;
    const int64_t     deadline_ms = getDeadlineMs(5000);
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    stream_store_->markCancelled(unique_key);

    // First add is rejected (consumes the cancel record)
    EXPECT_FALSE(stream_store_->addResource(meta, resource));

    // Second add for the same key (e.g. a new request with the same key) succeeds
    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 200);
    ASSERT_NE(entry, nullptr);
}

}  // namespace rtp_llm
