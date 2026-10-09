#include "rtp_llm/cpp/rdma_transport/DeferredReadQueue.h"
#include <gtest/gtest.h>
#include <thread>

using rtp_llm::rdma_transport::RdmaDeferredReadQueue;

TEST(DeferredReadQueueTest, CompletionBeforeOwnershipHandoffDoesNotRelease) {
    RdmaDeferredReadQueue queue;
    auto                  completed = std::make_shared<std::atomic<bool>>(true);
    auto                  storage   = std::make_shared<int>(1);
    std::weak_ptr<int>    weak      = storage;
    int                   released  = 0;
    auto                  ticket    = queue.prepare(completed, storage, [&] { ++released; });
    storage.reset();
    queue.reclaimCompleted();
    EXPECT_EQ(released, 0);
    EXPECT_FALSE(weak.expired());
    ticket->store(RdmaDeferredReadQueue::Ownership::DEFERRED, std::memory_order_release);
    queue.reclaimCompleted();
    queue.reclaimCompleted();
    EXPECT_EQ(released, 1);
    EXPECT_TRUE(weak.expired());
}

TEST(DeferredReadQueueTest, TimeoutRetainsStorageUntilConfirmedLateCompletion) {
    RdmaDeferredReadQueue queue;
    auto                  completed = std::make_shared<std::atomic<bool>>(false);
    auto                  storage   = std::make_shared<int>(1);
    std::weak_ptr<int>    weak      = storage;
    int                   released  = 0;
    queue.defer(completed, storage, [&] { ++released; });
    storage.reset();
    queue.reclaimCompleted();
    EXPECT_EQ(released, 0);
    EXPECT_FALSE(weak.expired());
    std::thread callback([completed] { completed->store(true, std::memory_order_release); });
    callback.join();
    queue.reclaimCompleted();
    EXPECT_EQ(released, 1);
    EXPECT_TRUE(weak.expired());
}

TEST(DeferredReadQueueTest, SuccessfulCallerAndShutdownNeverDoubleRelease) {
    RdmaDeferredReadQueue queue;
    auto                  completed = std::make_shared<std::atomic<bool>>(true);
    int                   released  = 0;
    auto                  ticket    = queue.prepare(completed, std::make_shared<int>(1), [&] { ++released; });
    ticket->store(RdmaDeferredReadQueue::Ownership::CALLER);
    queue.reclaimCompleted();
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(released, 0);
    completed->store(false);
    queue.defer(completed, std::make_shared<int>(2), [&] { ++released; });
    queue.clearAfterTransportShutdown();
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(released, 0);
}
