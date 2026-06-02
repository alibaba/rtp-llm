#include <gtest/gtest.h>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

using transfer::TransferTask;
using transfer::TransferTaskStore;

class TransferTaskStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        store_ = std::make_unique<TransferTaskStore>();
    }

    std::shared_ptr<TransferTask> addTask(const std::string& key, int64_t deadline_offset_ms = 5000) {
        return store_->addTask(key, {}, currentTimeMs() + deadline_offset_ms);
    }

protected:
    std::unique_ptr<TransferTaskStore> store_;
};

// ==================== 基础 CRUD ====================

TEST_F(TransferTaskStoreTest, AddAndGet_ReturnsSameTask) {
    auto task = addTask("key1");
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(store_->getTask("key1"), task);
}

TEST_F(TransferTaskStoreTest, Get_NonExistent_ReturnsNull) {
    EXPECT_EQ(store_->getTask("no_such_key"), nullptr);
}

TEST_F(TransferTaskStoreTest, Steal_RemovesFromStore) {
    auto task   = addTask("key1");
    auto stolen = store_->stealTask("key1");

    EXPECT_EQ(stolen, task);
    EXPECT_EQ(store_->getTask("key1"), nullptr);
}

TEST_F(TransferTaskStoreTest, Steal_NonExistent_ReturnsNull) {
    EXPECT_EQ(store_->stealTask("no_such_key"), nullptr);
}

TEST_F(TransferTaskStoreTest, Steal_SecondTime_ReturnsNull) {
    addTask("key1");
    store_->stealTask("key1");
    EXPECT_EQ(store_->stealTask("key1"), nullptr);
}

TEST_F(TransferTaskStoreTest, GetTaskCount_Accurate) {
    EXPECT_EQ(store_->getTaskCount(), 0);
    addTask("k1");
    EXPECT_EQ(store_->getTaskCount(), 1);
    addTask("k2");
    addTask("k3");
    EXPECT_EQ(store_->getTaskCount(), 3);
    store_->stealTask("k2");
    EXPECT_EQ(store_->getTaskCount(), 2);
}

// ==================== 边界与覆盖 ====================

/// Adding the same key twice must return nullptr and leave the original task intact.
TEST_F(TransferTaskStoreTest, AddDuplicateKey_ReturnsNullAndKeepsOriginal) {
    auto task1 = addTask("key1");
    ASSERT_NE(task1, nullptr);

    auto task2 = addTask("key1");  // duplicate → must fail
    EXPECT_EQ(task2, nullptr);

    // Original task must still be retrievable and unchanged.
    EXPECT_EQ(store_->getTask("key1"), task1);
    EXPECT_EQ(store_->getTaskCount(), 1);
}

/// After stealing the original task, adding the same key again must succeed.
TEST_F(TransferTaskStoreTest, AddDuplicateKey_AfterSteal_Succeeds) {
    auto task1 = addTask("key1");
    store_->stealTask("key1");

    auto task2 = addTask("key1");
    ASSERT_NE(task2, nullptr);
    EXPECT_NE(task2, task1);
    EXPECT_EQ(store_->getTask("key1"), task2);
}

/// getTask does not remove the task; subsequent getTask returns the same object.
TEST_F(TransferTaskStoreTest, Get_IsNonDestructive) {
    auto task = addTask("key1");
    EXPECT_EQ(store_->getTask("key1"), task);
    EXPECT_EQ(store_->getTask("key1"), task);
    EXPECT_EQ(store_->getTaskCount(), 1);
}

// ==================== 并发安全 ====================

/// Each thread adds a unique key and immediately steals it; no crash, final count == 0.
TEST_F(TransferTaskStoreTest, Concurrent_AddSteal_NoCrash) {
    constexpr int            num_threads = 8;
    std::vector<std::thread> threads;
    std::atomic<int>         steal_success{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i, &steal_success]() {
            std::string key = "key_" + std::to_string(i);
            store_->addTask(key, {}, currentTimeMs() + 5000);
            if (store_->stealTask(key)) {
                steal_success.fetch_add(1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(steal_success.load(), num_threads);
    EXPECT_EQ(store_->getTaskCount(), 0);
}

}  // namespace rtp_llm
