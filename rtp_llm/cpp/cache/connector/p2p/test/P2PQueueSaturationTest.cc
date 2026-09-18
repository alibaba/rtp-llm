#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace {

class QueueGate {
public:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
    }
    bool entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return entered_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false, released_ = false;
};

class P2PQueueSaturationTest: public ::testing::Test {
protected:
    template<class Predicate>
    bool waitFor(Predicate predicate) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!predicate()) {
            if (std::chrono::steady_clock::now() >= end) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return true;
    }
    void SetUp() override {
        for (int rank = 0; rank < 2; ++rank) {
            auto server = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
            ASSERT_TRUE(server->start());
            server->service()->setLeaseStatus(true, 1, 1, true);
            workers_.push_back(std::move(server));
        }
        prefill_ = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
        ASSERT_TRUE(prefill_->start());
        config_.topology = test::makeTestCacheTopology(1, 1, {{0}});
        config_.load_cache_timeout_ms = 10000;
        for (const auto& worker : workers_) {
            config_.worker_grpc_addrs.push_back("127.0.0.1:" + std::to_string(worker->listenPort()));
            config_.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(worker->listenPort()));
        }
        client_ = std::make_shared<P2PBroadcastClient>(config_.worker_grpc_addrs);
        ASSERT_TRUE(client_->init());
        scheduler_ = std::make_unique<P2PConnectorSchedulerDecode>(config_, nullptr, client_);
        ASSERT_TRUE(scheduler_->init("saturation"));
        // Replace both references before any request exists. Keep production
        // kickoff and control dispatch sharing one bounded pool, with no new API.
        scheduler_->checker_->stop();
        scheduler_->async_read_pool_->stop();
        scheduler_->async_read_pool_->join();
        pool_ = std::make_shared<autil::LockFreeThreadPool>(1, 2, nullptr, "SmallKickoffPool");
        ASSERT_TRUE(pool_->start());
        scheduler_->async_read_pool_ = pool_;
        scheduler_->checker_ = std::make_shared<P2PConnectorAsyncReadContextChecker>();
        ASSERT_TRUE(scheduler_->checker_->init(nullptr, client_, pool_));
        gate_ = std::make_shared<QueueGate>();
        ASSERT_EQ(pool_->pushTask([gate = gate_] { gate->wait(); }, false), autil::ThreadPoolBase::ERROR_NONE);
        ASSERT_TRUE(gate_->entered());
    }
    void TearDown() override {
        if (gate_) gate_->release();
        scheduler_.reset();
        pool_.reset();
        client_.reset();
        workers_.clear();
        prefill_.reset();
    }
    void fillQueue() {
        for (int i = 0; i < 16; ++i) {
            const auto error = pool_->pushTask([] {}, false, false);
            if (error == autil::ThreadPoolBase::ERROR_POOL_QUEUE_FULL) return;
            ASSERT_EQ(error, autil::ThreadPoolBase::ERROR_NONE);
        }
        FAIL() << "test queue never saturated";
    }
    KVCacheResourcePtr resource() {
        auto result = std::make_shared<KVCacheResource>();
        result->initGroups(config_.topology);
        result->mutableBlockIds(0).assign({1, 2});
        result->cacheKeys() = {101, 102};
        return result;
    }
    std::shared_ptr<MockMeta> meta(const std::string& key, int64_t timeout = 10000) {
        auto result = std::make_shared<MockMeta>();
        result->setUniqueKey(key);
        result->setRequestId(1);
        result->setDeadlineMs(currentTimeMs() + timeout);
        result->setPrefillAddr("127.0.0.1", prefill_->listenPort());
        result->setPrefillTpSize(1);
        return result;
    }
    void expectNoRpc() {
        EXPECT_EQ(prefill_->service()->getStartLoadCallCount(), 0);
        for (const auto& worker : workers_) EXPECT_EQ(worker->service()->getBroadcastTpCallCount(), 0);
    }
    void expectRecovery() {
        gate_->release();
        ASSERT_TRUE(waitFor([&] { return pool_->getItemCount() == 0; }));
        auto result = scheduler_->asyncRead(resource(), meta("recovery"), {2, 0}, true);
        ASSERT_TRUE(result.ok());
        ASSERT_NE(result.context, nullptr);
        ASSERT_TRUE(waitFor([&] { return result.context->done(); }));
        EXPECT_TRUE(result.context->success()) << result.context->errorInfo().ToString();
    }
    void queuedTermination(bool cancel) {
        auto held = resource();
        std::weak_ptr<KVCacheResource> weak = held;
        auto result = scheduler_->asyncRead(held, meta("queued", cancel ? 10000 : 100), {0, 2});
        held.reset();
        ASSERT_TRUE(result.ok());
        ASSERT_NE(result.context, nullptr);
        if (cancel) scheduler_->cancel(result.context);
        ASSERT_TRUE(waitFor([&] { return result.context->done(); }));
        EXPECT_EQ(result.context->errorInfo().code(), cancel ? ErrorCode::CANCELLED : ErrorCode::GENERATE_TIMEOUT);
        ASSERT_NO_FATAL_FAILURE(expectNoRpc());
        result.context.reset();
        gate_->release();
        // The queued closure may retain the context until dequeued, but cannot
        // create RPCs for a request which already terminated.
        ASSERT_TRUE(waitFor([&] { return weak.expired(); }));
        ASSERT_NO_FATAL_FAILURE(expectNoRpc());
        ASSERT_NO_FATAL_FAILURE(expectRecovery());
    }
    P2PConnectorSchedulerConfig config_;
    std::vector<std::unique_ptr<TestRpcServer>> workers_;
    std::unique_ptr<TestRpcServer> prefill_;
    std::shared_ptr<P2PBroadcastClient> client_;
    std::shared_ptr<autil::LockFreeThreadPool> pool_;
    std::unique_ptr<P2PConnectorSchedulerDecode> scheduler_;
    std::shared_ptr<QueueGate> gate_;
};

TEST_F(P2PQueueSaturationTest, FullKickoffQueueRejectsWithoutRpcOrRetainedResource) {
    ASSERT_NO_FATAL_FAILURE(fillQueue());
    auto held = resource();
    std::weak_ptr<KVCacheResource> weak = held;
    auto submitting = std::async(std::launch::async, [&] {
        return scheduler_->asyncRead(held, meta("rejected"), {0, 2});
    });
    // Declared after the future so even a failing assertion unblocks enqueue
    // before std::future's destructor joins its thread.
    auto cleanup = std::shared_ptr<void>(nullptr, [gate = gate_](void*) { gate->release(); });
    ASSERT_EQ(submitting.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = submitting.get();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.context, nullptr);
    EXPECT_EQ(result.error_info.code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
    EXPECT_EQ(scheduler_->checker_->inflightContextCount(), 0u);
    held.reset();
    EXPECT_TRUE(weak.expired());
    ASSERT_NO_FATAL_FAILURE(expectNoRpc());
    ASSERT_NO_FATAL_FAILURE(expectRecovery());
}

TEST_F(P2PQueueSaturationTest, QueuedCancelNeverStartsRpcAfterCapacityReturns) {
    ASSERT_NO_FATAL_FAILURE(queuedTermination(true));
}

TEST_F(P2PQueueSaturationTest, QueuedDeadlineNeverStartsRpcAfterCapacityReturns) {
    ASSERT_NO_FATAL_FAILURE(queuedTermination(false));
}

TEST_F(P2PQueueSaturationTest, SaturatedControlPoolStillExpiresAndEventuallyDrainsEveryKey) {
    ASSERT_NO_FATAL_FAILURE(fillQueue());
    std::vector<std::weak_ptr<P2PConnectorAsyncReadContext>> pending;
    for (int i = 0; i < 8; ++i) {
        auto broadcast = std::make_shared<P2PBroadcastClient::Result>("control_" + std::to_string(i));
        auto server = std::make_shared<DecodeLoadHelper::Result>();  // An unfinished StartLoad.
        // Exercise deadline expiry and lease cleanup without collecting metrics.
        auto context = std::make_shared<P2PConnectorAsyncReadContext>(
            resource(), broadcast, server, nullptr,
            10000, false, currentTimeMs() + 10000, currentTimeMs() + 100);
        scheduler_->checker_->addContext(context);
        pending.push_back(context);
    }
    ASSERT_TRUE(waitFor([&] {
        for (const auto& weak : pending) {
            auto context = weak.lock();
            if (!context || !context->done()) return false;
        }
        return true;
    }));
    for (const auto& weak : pending) {
        auto context = weak.lock();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
        EXPECT_TRUE(context->resourceHoldPending());
    }
    ASSERT_NO_FATAL_FAILURE(expectNoRpc());
    gate_->release();
    ASSERT_TRUE(waitFor([&] {
        for (const auto& weak : pending) if (!weak.expired()) return false;
        return true;
    }));
    for (const auto& worker : workers_) {
        EXPECT_EQ(worker->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::CANCEL_READ), 8);
        EXPECT_GE(worker->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::QUERY_LEASE_STATUS), 8);
    }
    EXPECT_EQ(scheduler_->checker_->inflightContextCount(), 0u);
    ASSERT_NO_FATAL_FAILURE(expectRecovery());
}

}  // namespace
}  // namespace rtp_llm
