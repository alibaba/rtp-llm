#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

#include <chrono>
#include <future>
#include <stdexcept>

namespace rtp_llm {
namespace {

class PendingStore: public NormalCacheStore {
public:
    PendingStore() {
        request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    }

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        callbacks.push_back(std::move(callback));
        if (throw_after == callbacks.size()) {
            throw std::runtime_error("dispatch interrupted");
        }
    }

    size_t                                  throw_after = 0;
    std::vector<CacheStoreLoadDoneCallback> callbacks;
};

struct PendingLoad {
    explicit PendingLoad(bool rdma): context(std::make_shared<LoadContext>(store, rdma)) {
        auto lifetime = std::make_shared<int>(0);
        owner         = lifetime;
        buffers.front()->setLoadWriterLifetime(std::move(lifetime));
    }

    void submit() {
        context->load(buffers, "peer", 1, 1, 60000, [] { return true; }, 1, 0);
    }

    std::shared_ptr<PendingStore>                    store = std::make_shared<PendingStore>();
    std::shared_ptr<LoadContext>                     context;
    std::vector<std::shared_ptr<RequestBlockBuffer>> buffers{std::make_shared<RequestBlockBuffer>("first"),
                                                             std::make_shared<RequestBlockBuffer>("second")};
    std::weak_ptr<int>                               owner;
};

TEST(LoadContextCompletionTest, TcpCancellationWaitsForEveryLocalCopyIncludingErrors) {
    PendingLoad load(false);
    load.submit();
    ASSERT_EQ(load.store->callbacks.size(), 2u);
    int observers = 0;
    load.context->onDestinationWritesStopped([&] { ++observers; });
    load.context->waitDone();
    EXPECT_EQ(load.context->getErrorInfo().code(), ErrorCode::CANCELLED);
    EXPECT_FALSE(load.context->waitForDestinationWritesStopped(std::chrono::steady_clock::now()));
    load.store->callbacks[0].destinationWritesStopped();
    load.store->callbacks[0](true, CacheStoreErrorCode::None);
    EXPECT_FALSE(load.owner.expired());
    EXPECT_EQ(observers, 0);
    load.store->callbacks[1](false, CacheStoreErrorCode::LoadErrorUnknown);
    EXPECT_FALSE(load.owner.expired());
    load.store->callbacks[1].destinationWritesStopped();
    EXPECT_TRUE(load.context->waitForDestinationWritesStopped(std::chrono::steady_clock::now()));
    EXPECT_TRUE(load.owner.expired());
    EXPECT_EQ(observers, 1);
    EXPECT_EQ(load.context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextCompletionTest, RdmaBusinessCompletionAloneDoesNotReleaseTarget) {
    for (bool success : {false, true}) {
        PendingLoad load(true);
        load.submit();
        ASSERT_EQ(load.store->callbacks.size(), 1u);
        auto& callback = load.store->callbacks.front();
        callback(success, success ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadRdmaConnectFailed);
        load.context->waitDone();
        EXPECT_FALSE(load.context->destinationWritesStopped());
        EXPECT_FALSE(load.owner.expired());
        auto stop = callback.destinationStopNotifier();
        load.store->callbacks.clear();
        std::weak_ptr<LoadContext> weak_context = load.context;
        int                        observers    = 0;
        load.context->onDestinationWritesStopped([&] { ++observers; });
        load.context.reset();
        EXPECT_TRUE(weak_context.expired());
        EXPECT_FALSE(load.owner.expired());
        stop();
        stop();
        EXPECT_TRUE(load.owner.expired());
        EXPECT_EQ(observers, 1);
    }
}

TEST(LoadContextCompletionTest, InterruptedDispatchRetainsEveryIssuedTarget) {
    PendingLoad load(false);
    load.store->throw_after = 2;
    EXPECT_THROW(load.submit(), std::runtime_error);
    ASSERT_EQ(load.store->callbacks.size(), 2u);
    load.store->callbacks[1](false, CacheStoreErrorCode::LoadErrorUnknown);
    load.store->callbacks[1].destinationWritesStopped();
    EXPECT_FALSE(load.context->destinationWritesStopped());
    EXPECT_FALSE(load.owner.expired());
    load.store->callbacks[0](true, CacheStoreErrorCode::None);
    load.store->callbacks[0].destinationWritesStopped();
    EXPECT_TRUE(load.context->destinationWritesStopped());
    EXPECT_TRUE(load.owner.expired());
    bool stopped = false;
    load.context->onDestinationWritesStopped([&] { stopped = true; });
    EXPECT_TRUE(stopped);
}

TEST(LoadContextCompletionTest, IsolationDeadlineRequestsStopAndStillWaitsForItsReceipt) {
    PendingLoad load(true);
    load.submit();
    auto& callback  = load.store->callbacks.front();
    auto  requested = std::make_shared<std::promise<void>>();
    auto  ready     = requested->get_future();
    callback.setDestinationFence([requested] { requested->set_value(); });
    load.context->recoverDestinationWritesAfterTimeout(std::chrono::milliseconds(1));
    load.context->destinationRecoveryCallback()();
    ASSERT_EQ(ready.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(load.owner.expired());
    callback.destinationWritesStopped();
    EXPECT_TRUE(load.owner.expired());
}

TEST(LoadContextCompletionTest, DeadlineAlsoFencesWritesWhoseTransportWasNotReadyYet) {
    PendingLoad load(false);
    load.submit();
    auto requested = std::make_shared<std::promise<void>>();
    auto ready     = requested->get_future();
    load.store->callbacks[0].setDestinationFence([requested] { requested->set_value(); });
    load.context->recoverDestinationWritesAfterTimeout(std::chrono::milliseconds(0));
    ASSERT_EQ(ready.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    load.store->callbacks[0].destinationWritesStopped();
    bool late_fenced = false;
    load.store->callbacks[1].setDestinationFence([&] { late_fenced = true; });
    EXPECT_TRUE(late_fenced);
    EXPECT_FALSE(load.owner.expired());
    load.store->callbacks[1].destinationWritesStopped();
    EXPECT_TRUE(load.owner.expired());
}

}  // namespace
}  // namespace rtp_llm
