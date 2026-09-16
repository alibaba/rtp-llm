#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <limits>
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
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100, deadline_ms);

    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->request_id, request_id);
    EXPECT_EQ(entry->kv_cache_resource, resource);
    EXPECT_EQ(entry->request_deadline_ms, deadline_ms);
}

TEST_F(P2PConnectorResourceStoreTest, StealResource_NotFound) {
    const int64_t deadline_ms = currentTimeMs() + 5000;
    // Use short timeout to avoid waiting
    auto entry = stream_store_->waitAndStealResource("non_existent_key", currentTimeMs() + 10, deadline_ms);

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
    auto entry1 = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100, deadline_ms);
    ASSERT_NE(entry1, nullptr);

    // Second steal should fail (already removed)
    auto entry2 = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 10, deadline_ms);
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
    auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms);
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
    auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms);
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
    auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms);
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
    const auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms);
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
    const auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms, is_cancelled);
    const auto elapsed_ms = currentTimeMs() - start_time;

    cancel_thread.join();

    EXPECT_EQ(entry, nullptr);
    EXPECT_GE(elapsed_ms, 80);
    EXPECT_LT(elapsed_ms, 2000);

    // Store 仍可用：补资源后应能正常 steal
    const int64_t request_id = 2001;
    auto          meta       = createMockMeta(unique_key, request_id, deadline_ms);
    auto          resource   = createMockKVCacheResource();
    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry_after = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 500, deadline_ms);
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
    const auto entry      = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms, is_cancelled);
    const auto elapsed_ms = currentTimeMs() - start_time;

    cancel_early.join();
    late_add.join();

    EXPECT_EQ(entry, nullptr);
    EXPECT_LT(elapsed_ms, 250);

    // 资源已入 store，未被取消路径取走，应仍可 steal
    auto stolen = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 500, deadline_ms);
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
        entry1 = stream_store_->waitAndStealResource(unique_key_1, deadline_ms, deadline_ms);
        waiter1_done.store(true);
    });

    std::thread waiter2([&]() {
        entry2 = stream_store_->waitAndStealResource(unique_key_2, deadline_ms, deadline_ms);
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
        entry1 = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms);
        if (entry1) {
            success_count.fetch_add(1);
        }
    });

    std::thread waiter2([&]() {
        entry2 = stream_store_->waitAndStealResource(unique_key, deadline_ms, deadline_ms);
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

    // Wait past the request deadline for the deadline waiter to release the resource.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // Resource should have been removed due to timeout
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 10, deadline_ms);
    EXPECT_EQ(entry, nullptr);
}

TEST(P2PConnectorResourceStoreDeadlineTest, ResourceTimeoutUsesRequestDeadlineNotPollingInterval) {
    std::mutex              mutex;
    std::condition_variable released_cv;
    bool                    released = false;
    {
        P2PConnectorResourceStore store(nullptr, 10 * 1000);
        ASSERT_TRUE(store.init());
        store.setOnRequestReleased([&](int64_t, int64_t) {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
            released_cv.notify_one();
        });

        const std::string unique_key  = "event_driven_timeout";
        const int64_t     deadline_ms = currentTimeMs() + 50;
        auto              meta        = std::make_shared<MockMeta>();
        meta->setUniqueKey(unique_key);
        meta->setRequestId(1010);
        meta->setDeadlineMs(deadline_ms);
        meta->setPrefillAddr("127.0.0.1", 12345);
        meta->setPrefillTpSize(1);
        ASSERT_TRUE(store.addResource(meta, std::make_shared<KVCacheResource>()));

        std::unique_lock<std::mutex> lock(mutex);
        EXPECT_TRUE(released_cv.wait_for(lock, std::chrono::milliseconds(500), [&]() { return released; }));
    }
}

TEST_F(P2PConnectorResourceStoreTest, SideChannelTimeout_AutoRemoval) {
    const std::string unique_key  = "test_side_channel_timeout";
    const int64_t     deadline_ms = currentTimeMs() + 50;

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 42;
    stream_store_->publishPrefillPayload(unique_key, deadline_ms, std::move(side_data));

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    EXPECT_FALSE(stream_store_->takePrefillPayload(unique_key, consumed_data));
}

TEST_F(P2PConnectorResourceStoreTest, ClearPrefillPayload_RemovesIndependentEntry) {
    const std::string unique_key  = "test_side_channel_clear";
    const int64_t     deadline_ms = getDeadlineMs(5000);

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 7;
    stream_store_->publishPrefillPayload(unique_key, deadline_ms, std::move(side_data));
    stream_store_->clearPrefillPayload(unique_key);

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    EXPECT_FALSE(stream_store_->takePrefillPayload(unique_key, consumed_data));
}

// Regression: 5/25 .199 prefill produced 50 "side-channel timeout" WARNs at
// 22:08-22:14 that all traced back to a cancel burst at 21:08 — handleRead
// cancelled the keys, but the engine still produced first-tokens shortly
// after and called publishPrefillPayload, which wrote to side_channel_data_map_
// with no consumer. Those entries sat for a full 1h business deadline before
// checkTimeout reaped them. After the fix, publishPrefillPayload checks
// cancelled_keys_ and skips the write so nothing leaks into the map.
TEST_F(P2PConnectorResourceStoreTest, PublishPrefillPayload_SkipsWriteIfKeyAlreadyCancelled) {
    const int64_t deadline_ms = currentTimeMs() + 5000;
    const std::string unique_key = "test_notify_after_cancel_skips_write";

    stream_store_->markCancelled(unique_key, deadline_ms);

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 42;
    stream_store_->publishPrefillPayload(unique_key, currentTimeMs() + 5000, std::move(side_data));

    P2PConnectorResourceEntry::SideChannelData consumed;
    EXPECT_FALSE(stream_store_->takePrefillPayload(unique_key, consumed))
        << "Side-channel entry should not be written when key is already cancelled, "
        << "otherwise it leaks into side_channel_data_map_ and produces a 1h-delayed "
        << "timeout WARN at checkTimeout() (see 5/25 .199 incident).";
}

// Counter-check: when the key was never cancelled, publishPrefillPayload
// still writes normally (otherwise the cancellation check has overshot).
TEST_F(P2PConnectorResourceStoreTest, PublishPrefillPayload_WritesNormallyWhenNotCancelled) {
    const int64_t deadline_ms = currentTimeMs() + 5000;
    const std::string unique_key = "test_notify_without_cancel_writes_normally";

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 99;
    stream_store_->publishPrefillPayload(unique_key, deadline_ms, std::move(side_data));

    P2PConnectorResourceEntry::SideChannelData consumed;
    ASSERT_TRUE(stream_store_->takePrefillPayload(unique_key, consumed));
    EXPECT_TRUE(consumed.has_first_token);
    EXPECT_EQ(consumed.first_token_id, 99);
}

TEST_F(P2PConnectorResourceStoreTest, StolenEntry_SideChannelUsesTransferDeadline) {
    const std::string unique_key  = "test_side_channel_after_steal";
    const int64_t     request_id  = 1010;
    const int64_t     deadline_ms = currentTimeMs() + 1000;
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100, deadline_ms);
    ASSERT_NE(entry, nullptr);

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 88;
    stream_store_->publishPrefillPayload(unique_key, deadline_ms, std::move(side_data));

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    EXPECT_FALSE(stream_store_->takePrefillPayload(unique_key, consumed_data));
}

TEST_F(P2PConnectorResourceStoreTest, StolenEntry_TerminalRejectsLateSideChannel) {
    const std::string unique_key  = "test_terminal_after_steal";
    const int64_t     request_id  = 1011;
    const int64_t     deadline_ms = currentTimeMs() + 5000;
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    ASSERT_NE(stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 100, deadline_ms), nullptr);
    stream_store_->markTerminal(unique_key, deadline_ms);

    P2PConnectorResourceEntry::SideChannelData side_data;
    side_data.has_first_token = true;
    side_data.first_token_id  = 89;
    stream_store_->publishPrefillPayload(unique_key, deadline_ms, std::move(side_data));

    P2PConnectorResourceEntry::SideChannelData consumed_data;
    EXPECT_FALSE(stream_store_->takePrefillPayload(unique_key, consumed_data));
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
    stream_store_->markCancelled(unique_key, deadline_ms);

    // Resource should be gone — steal fails immediately
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 50, deadline_ms);
    EXPECT_EQ(entry, nullptr);
    EXPECT_TRUE(stream_store_->isMarkedCancelled(unique_key));
}

// markCancelled before resource arrives → addResource rejects the resource on arrival
TEST_F(P2PConnectorResourceStoreTest, MarkCancelled_ResourceNotYetInStore_RejectsSubsequentAdd) {
    const std::string unique_key  = "test_cancel_before_add";
    const int64_t     request_id  = 3002;
    const int64_t     deadline_ms = getDeadlineMs(5000);
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    // Cancel before prefill adds the resource
    stream_store_->markCancelled(unique_key, deadline_ms);

    // Resource arrives later (prefill finished inference after decode already timed out)
    bool added = stream_store_->addResource(meta, resource);
    EXPECT_FALSE(added);

    // Resource should not be stealable
    auto entry = stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 50, deadline_ms);
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

    stream_store_->markCancelled(key_cancelled, deadline_ms);

    // A different key should still work normally
    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    auto entry = stream_store_->waitAndStealResource(key_normal, currentTimeMs() + 200, deadline_ms);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->request_id, request_id);
}

// A rejected late resource must not consume the terminal record. Otherwise a
// second late add/start pair can recreate a request that has already ended.
TEST_F(P2PConnectorResourceStoreTest, MarkCancelled_TombstoneRetainedAfterRejection) {
    const std::string unique_key  = "test_cancel_record_consumed";
    const int64_t     request_id  = 3004;
    const int64_t     deadline_ms = getDeadlineMs(5000);
    auto              meta        = createMockMeta(unique_key, request_id, deadline_ms);
    auto              resource    = createMockKVCacheResource();

    stream_store_->markCancelled(unique_key, deadline_ms);

    // Every add for the same terminal request is rejected until its deadline.
    EXPECT_FALSE(stream_store_->addResource(meta, resource));
    EXPECT_FALSE(stream_store_->addResource(meta, resource));
    EXPECT_TRUE(stream_store_->isMarkedCancelled(unique_key));
}

TEST_F(P2PConnectorResourceStoreTest, AddResource_ExpiredRequestIsRejectedAfterTombstoneExpiry) {
    const int64_t deadline_ms = currentTimeMs() + 5000;
    const std::string unique_key = "test_expired_request_add";
    auto meta = createMockMeta(unique_key, 3005, currentTimeMs() - 1);

    EXPECT_FALSE(stream_store_->addResource(meta, createMockKVCacheResource()));
    EXPECT_EQ(stream_store_->waitAndStealResource(unique_key, currentTimeMs() + 10, deadline_ms), nullptr);
}


TEST_F(P2PConnectorResourceStoreTest, NoStartLoadRetainsResourceUntilRequestDeadline) {
    const int64_t request_deadline_ms = currentTimeMs() + 7200000;
    auto meta = createMockMeta("request_hold", 5001, request_deadline_ms);
    auto resource = createMockKVCacheResource();
    ASSERT_TRUE(stream_store_->addResource(meta, resource));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const int64_t load_deadline_ms = currentTimeMs() + 1000;
    auto entry = stream_store_->waitAndStealResource("request_hold", load_deadline_ms, request_deadline_ms);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->kv_cache_resource, resource);
    EXPECT_EQ(entry->deadline_ms, load_deadline_ms);
    EXPECT_EQ(entry->request_deadline_ms, request_deadline_ms);
}

TEST_F(P2PConnectorResourceStoreTest, ExpiredLoadRejectsLateResourceAndSideChannel) {
    const int64_t request_deadline_ms = currentTimeMs() + 5000;
    const int64_t load_deadline_ms = currentTimeMs() + 30;
    EXPECT_EQ(stream_store_->waitAndStealResource("late_load", load_deadline_ms, request_deadline_ms), nullptr);
    auto meta = createMockMeta("late_load", 5002, request_deadline_ms);
    EXPECT_FALSE(stream_store_->addResource(meta, createMockKVCacheResource()));
    P2PConnectorResourceEntry::SideChannelData data;
    stream_store_->publishPrefillPayload("late_load", request_deadline_ms, std::move(data));
    EXPECT_FALSE(stream_store_->takePrefillPayload("late_load", data));
    EXPECT_EQ(stream_store_->waitAndStealResource("late_load", currentTimeMs() + 1000, request_deadline_ms), nullptr);
}

TEST_F(P2PConnectorResourceStoreTest, DuplicateStartLoadCannotExtendSideChannelDeadline) {
    const int64_t request_deadline_ms = currentTimeMs() + 5000;
    const int64_t load_deadline_ms = currentTimeMs() + 30;
    auto meta = createMockMeta("duplicate_load", 5003, request_deadline_ms);
    ASSERT_TRUE(stream_store_->addResource(meta, createMockKVCacheResource()));
    ASSERT_NE(stream_store_->waitAndStealResource("duplicate_load", load_deadline_ms, request_deadline_ms), nullptr);
    EXPECT_EQ(stream_store_->waitAndStealResource("duplicate_load", request_deadline_ms, request_deadline_ms), nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    P2PConnectorResourceEntry::SideChannelData data;
    stream_store_->publishPrefillPayload("duplicate_load", request_deadline_ms, std::move(data));
    EXPECT_FALSE(stream_store_->takePrefillPayload("duplicate_load", data));
}

TEST_F(P2PConnectorResourceStoreTest, MissingOrInfiniteDeadlineDoesNotCreateResource) {
    for (int64_t invalid : {int64_t{0}, std::numeric_limits<int64_t>::max()}) {
        auto meta = createMockMeta("invalid_deadline", 5004, invalid);
        EXPECT_FALSE(stream_store_->addResource(meta, createMockKVCacheResource()));
        EXPECT_EQ(stream_store_->waitAndStealResource("invalid_deadline", currentTimeMs() + 100, invalid), nullptr);
    }
}

TEST(P2PRequestTimeoutTest, FirstArrivalWinsAndTerminalCannotRenew) {
    P2PConnectorResourceStore store(nullptr, 10);
    const auto before = currentTimeMs();
    const auto deadline = store.requestDeadline("relative-timeout", 5000);
    EXPECT_GE(deadline, before + 5000);
    EXPECT_LE(deadline, currentTimeMs() + 5000);
    EXPECT_EQ(store.requestDeadline("relative-timeout", 60000), deadline);
    store.markTerminal("relative-timeout", deadline);
    EXPECT_EQ(store.requestDeadline("relative-timeout", 60000), deadline);
    EXPECT_TRUE(store.isMarkedCancelled("relative-timeout"));
}

TEST(P2PRequestTimeoutTest, RegistrationWaitReadsExistingDeadline) {
    P2PConnectorResourceStore store(nullptr, 10);
    const auto                deadline = store.requestDeadline("registered", 5000);
    EXPECT_EQ(store.waitForRequestDeadline("registered", currentTimeMs() + 1000), deadline);
    EXPECT_EQ(store.waitForRequestDeadline("registered", currentTimeMs() + 1000, [] { return true; }), 0);
    store.markTerminal("registered", deadline);
    EXPECT_EQ(store.waitForRequestDeadline("registered", currentTimeMs() + 1000), 0);
}

TEST(P2PRequestTimeoutTest, RegistrationWaitRequiresMatchingGenerateStream) {
    P2PConnectorResourceStore store(nullptr, 10);
    store.requestDeadline("other-key", 5000);
    std::promise<void> waiting;
    auto               started = waiting.get_future();
    std::atomic<bool>  signalled{false};
    auto               result = std::async(std::launch::async, [&] {
        return store.waitForRequestDeadline("matching-key", currentTimeMs() + 5000, [&] {
            if (!signalled.exchange(true)) {
                waiting.set_value();
            }
            return false;
        });
    });
    ASSERT_EQ(started.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    const auto deadline = store.requestDeadline("matching-key", 1000);
    ASSERT_EQ(result.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(result.get(), deadline);
}

TEST(P2PRequestTimeoutTest, RegistrationWaitDoesNotCreateOrRenewDeadline) {
    P2PConnectorResourceStore store(nullptr, 10);
    const auto                load_deadline = currentTimeMs() + 20;
    EXPECT_EQ(store.waitForRequestDeadline("not-registered", load_deadline), 0);
    EXPECT_GE(currentTimeMs(), load_deadline);
    EXPECT_EQ(store.waitForRequestDeadline("cancelled", currentTimeMs() + 5000, [] { return true; }), 0);
    const auto before_registration = currentTimeMs();
    const auto deadline            = store.requestDeadline("not-registered", 5000);
    EXPECT_GE(deadline, before_registration + 5000);
    EXPECT_EQ(store.waitForRequestDeadline("not-registered", currentTimeMs() + 1000), deadline);
    EXPECT_EQ(store.requestDeadline("not-registered", 60000), deadline);
}

TEST_F(P2PConnectorResourceStoreTest, ResourcePublicationDoesNotReplaceGenerateStreamRegistration) {
    const auto deadline = currentTimeMs() + 5000;
    ASSERT_TRUE(
        stream_store_->addResource(createMockMeta("resource-only", 5021, deadline), createMockKVCacheResource()));
    EXPECT_EQ(stream_store_->waitForRequestDeadline("resource-only", currentTimeMs() + 20), 0);
    const auto registered = stream_store_->requestDeadline("resource-only", 5000);
    EXPECT_EQ(stream_store_->waitForRequestDeadline("resource-only", currentTimeMs() + 1000), registered);
}

TEST(P2PRequestTimeoutTest, InvalidDurationsDoNotCreateRequest) {
    P2PConnectorResourceStore store(nullptr, 10);
    EXPECT_EQ(store.requestDeadline("", 1000), 0);
    EXPECT_EQ(store.requestDeadline("key", 0), 0);
    EXPECT_EQ(store.requestDeadline("key", -1), 0);
    EXPECT_EQ(store.requestDeadline("key", std::numeric_limits<int64_t>::max()), 0);
    EXPECT_GT(store.requestDeadline("key", 1000), currentTimeMs());
}

// Drive cleanup with an explicit time instead of sleeping or racing the deadline thread.
class P2PResourceStoreDeadlineIndexTest: public P2PConnectorResourceStoreTest {
protected:
    void SetUp() override {
        stream_store_ = std::make_unique<P2PConnectorResourceStore>(nullptr, 100);
    }
};

TEST_F(P2PResourceStoreDeadlineIndexTest, TerminalStateExpiresAtOriginalRequestDeadline) {
    auto& store = *stream_store_;
    const std::string key = "terminal";
    const auto deadline = store.requestDeadline(key, 60000);
    ASSERT_TRUE(store.addResource(createMockMeta(key, 6101, deadline), createMockKVCacheResource()));
    const auto load_deadline = currentTimeMs() + 10000;
    ASSERT_NE(store.waitAndStealResource(key, load_deadline, deadline), nullptr);
    EXPECT_EQ(store.nextDeadlineMsLocked(), load_deadline);
    store.markTerminal(key, deadline);
    EXPECT_EQ(store.nextDeadlineMsLocked(), deadline);
    store.checkTimeout(load_deadline);
    EXPECT_EQ(store.request_states_.size(), 1u);

    store.checkTimeout(deadline - 1);
    EXPECT_TRUE(store.isMarkedCancelled(key));
    EXPECT_EQ(store.requestDeadline(key, 120000), deadline);
    EXPECT_FALSE(store.addResource(createMockMeta(key, 6101, deadline), createMockKVCacheResource()));
    P2PConnectorResourceEntry::SideChannelData data;
    store.publishPrefillPayload(key, deadline, std::move(data));
    EXPECT_FALSE(store.takePrefillPayload(key, data));
    ASSERT_EQ(store.deadline_index_.size(), 1u);
    EXPECT_EQ(store.nextDeadlineMsLocked(), deadline);

    // No extra hour of retention after the original request deadline.
    store.checkTimeout(deadline);
    EXPECT_TRUE(store.request_states_.empty());
    EXPECT_TRUE(store.deadline_index_.empty());
    EXPECT_FALSE(store.nextDeadlineMsLocked().has_value());
}

TEST_F(P2PResourceStoreDeadlineIndexTest, LoadExpiryReleasesResourcesAndKeepsTerminalUntilRequestExpiry) {
    auto& store = *stream_store_;
    // Insert in the opposite order to expiry, including an unrelated live request.
    const auto later_deadline = store.requestDeadline("later", 120000);
    const auto request_deadline = store.requestDeadline("loading", 60000);
    const auto load_deadline = currentTimeMs() + 10000;
    auto resource = createMockKVCacheResource();
    std::weak_ptr<KVCacheResource> weak_resource = resource;
    ASSERT_TRUE(store.addResource(createMockMeta("loading", 6102, request_deadline), resource));
    resource.reset();
    int released = 0;
    store.setOnRequestReleased([&](int64_t request_id, int64_t deadline) {
        ++released;
        EXPECT_EQ(request_id, 6102);
        EXPECT_EQ(deadline, request_deadline);
        // The callback may re-enter the store; it must run outside the store lock.
        EXPECT_TRUE(store.isMarkedCancelled("loading"));
    });

    const auto generation = store.deadline_generation_;
    // Establish the shorter load deadline without consuming the stored resource.
    EXPECT_EQ(store.waitAndStealResource("loading", load_deadline, request_deadline, [] { return true; }), nullptr);
    EXPECT_GT(store.deadline_generation_, generation);
    EXPECT_EQ(store.nextDeadlineMsLocked(), load_deadline);
    EXPECT_EQ(store.deadline_index_.size(), 2u);
    P2PConnectorResourceEntry::SideChannelData data;
    store.publishPrefillPayload("loading", request_deadline, std::move(data));

    store.checkTimeout(load_deadline - 1);
    EXPECT_FALSE(weak_resource.expired());
    EXPECT_EQ(released, 0);
    store.checkTimeout(load_deadline);
    EXPECT_TRUE(weak_resource.expired());
    EXPECT_EQ(released, 1);
    EXPECT_TRUE(store.isMarkedCancelled("loading"));
    EXPECT_FALSE(store.takePrefillPayload("loading", data));
    EXPECT_FALSE(store.addResource(createMockMeta("loading", 6102, request_deadline), createMockKVCacheResource()));
    EXPECT_EQ(released, 2);  // A rejected late add also releases its computed layers.
    EXPECT_EQ(store.nextDeadlineMsLocked(), request_deadline);
    EXPECT_EQ(store.deadline_index_.size(), 2u);

    store.checkTimeout(request_deadline - 1);
    store.checkTimeout(request_deadline);
    EXPECT_EQ(released, 2);
    EXPECT_EQ(store.request_states_.count("loading"), 0u);
    EXPECT_EQ(store.request_states_.count("later"), 1u);
    EXPECT_EQ(store.nextDeadlineMsLocked(), later_deadline);
    store.checkTimeout(later_deadline);
    EXPECT_TRUE(store.request_states_.empty());
    EXPECT_TRUE(store.deadline_index_.empty());
}

TEST_F(P2PResourceStoreDeadlineIndexTest, PublicationsShareDeadlineWithoutAccumulatingTimers) {
    auto& store = *stream_store_;
    const auto deadline = currentTimeMs() + 60000;
    for (const std::string key : {"first", "second"}) {
        for (int i = 0; i < 16; ++i) {
            P2PConnectorResourceEntry::SideChannelData data;
            data.has_first_token = true;
            data.first_token_id = i;
            store.publishPrefillPayload(key, deadline, std::move(data));
        }
    }
    EXPECT_EQ(store.deadline_index_.size(), 2u);
    EXPECT_EQ(store.request_states_.size(), 2u);
    const auto generation = store.deadline_generation_;
    EXPECT_EQ(store.requestDeadline("first", 120000), deadline);
    // Completing a request with this same expiry needs no timer wakeup.
    store.markTerminal("first", deadline);
    EXPECT_EQ(store.deadline_generation_, generation);
    EXPECT_EQ(store.deadline_index_.size(), 2u);
    P2PConnectorResourceEntry::SideChannelData consumed;
    ASSERT_TRUE(store.takePrefillPayload("second", consumed));
    EXPECT_EQ(consumed.first_token_id, 15);

    store.checkTimeout(deadline);
    EXPECT_TRUE(store.request_states_.empty());
    EXPECT_TRUE(store.deadline_index_.empty());
}

TEST_F(P2PResourceStoreDeadlineIndexTest, ExpiredCallbacksDoNotRecreateCollectedState) {
    auto& store = *stream_store_;
    const auto expired_deadline = currentTimeMs() - 1;
    EXPECT_FALSE(store.addResource(createMockMeta("expired", 6103, expired_deadline), createMockKVCacheResource()));
    P2PConnectorResourceEntry::SideChannelData data;
    store.publishPrefillPayload("expired", expired_deadline, std::move(data));
    store.markTerminal("expired", expired_deadline);
    EXPECT_EQ(store.waitAndStealResource("expired", expired_deadline, expired_deadline), nullptr);
    EXPECT_TRUE(store.request_states_.empty());
    EXPECT_TRUE(store.resource_map_.empty());
    EXPECT_TRUE(store.deadline_index_.empty());
}

}  // namespace rtp_llm
