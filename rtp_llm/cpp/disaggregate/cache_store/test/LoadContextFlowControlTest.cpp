#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include <gtest/gtest.h>
#include <future>
#include <thread>

namespace rtp_llm {
namespace {
class FakeStore: public CacheStore {
public:
    std::function<void(CacheStoreLoadDoneCallback)> on_load;
    std::atomic<int>                                calls{0};
    std::vector<uint32_t>                           timeouts;
    void                                            load(const std::shared_ptr<RequestBlockBuffer>&,
                                                         CacheStoreLoadDoneCallback cb,
                                                         const std::string&,
                                                         uint32_t,
                                                         uint32_t,
                                                         uint32_t timeout,
                                                         int,
                                                         int) override {
        ++calls;
        timeouts.push_back(timeout);
        on_load(std::move(cb));
    }
    void store(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreStoreDoneCallback) override {}
    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&,
                                             uint32_t,
                                             uint32_t,
                                             int64_t,
                                             LoadContext::CheckCancelFunc,
                                             int,
                                             int) override {
        return {};
    }
    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                               int64_t) override {
        return {};
    }
    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return {};
    }
    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}
    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return false;
    }
    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return {};
    }
    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return memory_;
    }
    void debugInfo() override {}

private:
    std::shared_ptr<MemoryUtil> memory_;
};

std::vector<std::shared_ptr<RequestBlockBuffer>> buffers(int n) {
    std::vector<std::shared_ptr<RequestBlockBuffer>> result;
    for (int i = 0; i < n; ++i)
        result.push_back(std::make_shared<RequestBlockBuffer>(std::to_string(i)));
    return result;
}

TEST(LoadContextFlowControlTest, BoundsOutstandingCallsAndRefillsAfterCompletion) {
    auto                                   store = std::make_shared<FakeStore>();
    auto                                   ctx   = std::make_shared<LoadContext>(store, false);
    std::mutex                             mutex;
    std::condition_variable                cv;
    std::deque<CacheStoreLoadDoneCallback> callbacks;
    size_t                                 peak = 0;
    store->on_load                              = [&](auto cb) {
        std::lock_guard<std::mutex> lock(mutex);
        callbacks.push_back(std::move(cb));
        peak = std::max(peak, callbacks.size());
        cv.notify_one();
    };
    ctx->load(buffers(40), "peer", 1, 0, 2000, nullptr, 1, 0);
    ASSERT_EQ(store->calls, 2);
    auto waiter = std::async(std::launch::async, [&] { ctx->waitDone(); });
    for (int i = 0; i < 40; ++i) {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&] { return !callbacks.empty(); }));
        auto cb = std::move(callbacks.front());
        callbacks.pop_front();
        lock.unlock();
        cb(true, CacheStoreErrorCode::None);
    }
    waiter.get();
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(store->calls, 40);
    EXPECT_LE(peak, 2);
}

TEST(LoadContextFlowControlTest, RetriesOnlyQueueRejectionWithRemainingDeadline) {
    auto store     = std::make_shared<FakeStore>();
    store->on_load = [&](auto cb) {
        cb(store->calls > 2, store->calls > 2 ? CacheStoreErrorCode::None : CacheStoreErrorCode::PushWorkerItemFailed);
    };
    auto ctx = std::make_shared<LoadContext>(store, false);
    ctx->load(buffers(1), "peer", 1, 0, 1000, nullptr, 1, 0);
    ctx->waitDone();
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(store->calls, 3);
    EXPECT_LT(store->timeouts.back(), store->timeouts.front());
    EXPECT_TRUE(ctx->failedBlockDebugInfos().empty());

    store->calls   = 0;
    store->on_load = [](auto cb) { cb(false, CacheStoreErrorCode::InvalidParams); };
    ctx            = std::make_shared<LoadContext>(store, false);
    ctx->load(buffers(1), "peer", 1, 0, 1000, nullptr, 1, 0);
    ctx->waitDone();
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(store->calls, 1);
}

TEST(LoadContextFlowControlTest, QueueRetriesAreBounded) {
    auto store     = std::make_shared<FakeStore>();
    store->on_load = [](auto cb) { cb(false, CacheStoreErrorCode::PushWorkerItemFailed); };
    auto ctx       = std::make_shared<LoadContext>(store, false);
    ctx->load(buffers(1), "peer", 1, 0, 2000, nullptr, 1, 0);
    ctx->waitDone();
    EXPECT_EQ(store->calls, 6);
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(ctx->failedBlockDebugInfos().size(), 1);
}

TEST(LoadContextFlowControlTest, CancellationAndDeadlineStopUnsentCallsAndPreserveTerminalError) {
    for (bool cancel : {false, true}) {
        auto                                    store = std::make_shared<FakeStore>();
        std::vector<CacheStoreLoadDoneCallback> callbacks;
        store->on_load = [&](auto cb) { callbacks.push_back(std::move(cb)); };
        auto ctx       = std::make_shared<LoadContext>(store, false);
        bool cancelled = false;
        ctx->load(buffers(40), "peer", 1, 0, cancel ? 1000 : 50, [&] { return cancelled; }, 1, 0);
        cancelled = cancel;
        ctx->waitDone();
        auto expected = cancel ? ErrorCode::CANCELLED : ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT;
        EXPECT_EQ(ctx->getErrorInfo().code(), expected);
        EXPECT_EQ(store->calls, 2);
        for (auto& cb : callbacks)
            cb(false, CacheStoreErrorCode::PushWorkerItemFailed);
        EXPECT_EQ(ctx->getErrorInfo().code(), expected);
        EXPECT_EQ(store->calls, 2);
    }
}

TEST(LoadContextFlowControlTest, CombinedRdmaCallIsNotRetried) {
    auto store     = std::make_shared<FakeStore>();
    store->on_load = [](auto cb) { cb(false, CacheStoreErrorCode::PushWorkerItemFailed); };
    auto ctx       = std::make_shared<LoadContext>(store, true);
    ctx->load(buffers(40), "peer", 1, 2, 1000, nullptr, 1, 0);
    ctx->waitDone();
    EXPECT_EQ(store->calls, 1);
    EXPECT_FALSE(ctx->success());
}
}  // namespace
}  // namespace rtp_llm
