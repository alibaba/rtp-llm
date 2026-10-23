// Explicit transport stop must join ANet reporters before their dependencies
// are destroyed, even when cache-store owners outlive engine shutdown.

#include "gtest/gtest.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "aios/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"
#include <atomic>
#include <future>

namespace rtp_llm {

class TcpMessagerShutdownTest: public CacheStoreTestBase {};

namespace {
struct ReporterBarrier {
    std::promise<void>       entered;
    std::promise<void>       stopping;
    std::promise<void>       release;
    std::shared_future<void> released = release.get_future().share();
    std::atomic<bool>        callback_done{false};
    std::atomic<bool>        dependency_alive{true};
    std::atomic<bool>        callback_saw_live_dependency{false};
    std::atomic<bool>        first_callback{true};
};

class BarrierANetReporter final: public anet::KMonitorANetMetricReporter {
public:
    explicit BarrierANetReporter(std::shared_ptr<ReporterBarrier> barrier):
        anet::KMonitorANetMetricReporter(anet::ANetMetricReporterConfig{}, kmonitor::FATAL),
        barrier_(std::move(barrier)) {}

    ~BarrierANetReporter() override {
        barrier_->stopping.set_value();
        // Exercise the real ANet release/LoopThread join, not a mock join.
        release();
    }

private:
    std::shared_ptr<ReporterBarrier> barrier_;
};

class ShutdownProbeMessager final: public Messager {
public:
    using Messager::Messager;
    bool init(MessagerInitParams) override {
        return true;
    }
    void stop() override {
        stopped = true;
    }
    void load(const std::shared_ptr<LoadRequest>& request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>&) override {
        load_saw_live_transport = !stopped.load();
        ++loads;
        request->callback(true, CacheStoreErrorCode::None);
    }
    bool generateBlockInfo(::BlockBufferInfo*, const std::shared_ptr<BlockBuffer>&, uint32_t, uint32_t) override {
        return true;
    }
    std::atomic<bool> stopped{false};
    std::atomic<bool> load_saw_live_transport{false};
    std::atomic<int>  loads{0};
};
}  // namespace

TEST_F(TcpMessagerShutdownTest, QueuedLoadsDrainBeforeTransportStopsAndNewWorkIsRejected) {
    using namespace std::chrono_literals;
    CacheStoreInitParams params;
    params.listen_port  = autil::NetUtil::randomPort();
    params.rdma_mode    = false;
    params.thread_count = 2;
    params.memory_util  = memory_util_;
    auto store          = NormalCacheStore::createNormalCacheStore(params);
    ASSERT_NE(store, nullptr);
    auto messager = std::make_shared<ShutdownProbeMessager>(
        memory_util_, store->request_block_buffer_store_, kmonitor::MetricsReporterPtr());
    store->messager_ = messager;

    std::promise<void> entered;
    std::promise<void> release;
    auto               released = release.get_future().share();
    ASSERT_EQ(store->thread_pool_->pushTask([&]() {
        entered.set_value();
        released.wait();
    }),
              autil::ThreadPoolBase::ERROR_NONE);
    EXPECT_EQ(entered.get_future().wait_for(5s), std::future_status::ready);

    auto buffer = std::make_shared<RequestBlockBuffer>("queued-load-shutdown");
    buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", 16, 'x', false));
    std::promise<bool> loaded;
    auto               loaded_result = loaded.get_future();
    store->load(buffer, [&](bool ok, CacheStoreErrorCode) { loaded.set_value(ok); }, "127.0.0.1", 1, 0);
    auto       stopping = std::async(std::launch::async, [&]() { store->stopTransport(); });
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!store->thread_pool_close_ && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    EXPECT_TRUE(store->thread_pool_close_);
    EXPECT_EQ(stopping.wait_for(20ms), std::future_status::timeout);
    EXPECT_FALSE(messager->stopped);
    release.set_value();
    EXPECT_EQ(stopping.wait_for(5s), std::future_status::ready);
    stopping.get();
    ASSERT_EQ(loaded_result.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(loaded_result.get());
    EXPECT_TRUE(messager->load_saw_live_transport);
    EXPECT_TRUE(messager->stopped);

    int  rejected = 0;
    auto reject   = [&](bool ok, CacheStoreErrorCode ec) {
        EXPECT_FALSE(ok);
        EXPECT_EQ(ec, CacheStoreErrorCode::PushWorkerItemFailed);
        ++rejected;
    };
    store->load(buffer, reject, "127.0.0.1", 1, 0);
    store->store(buffer, reject);
    EXPECT_EQ(rejected, 2);
    EXPECT_EQ(messager->loads, 1);
}

TEST_F(TcpMessagerShutdownTest, StopTransportJoinsReporterBeforeRetainedStoreCanOutliveDependencies) {
    using namespace std::chrono_literals;
    CacheStoreInitParams params;
    params.listen_port      = autil::NetUtil::randomPort();
    params.rdma_listen_port = 0;
    params.rdma_mode        = false;
    params.thread_count     = 2;
    auto retained_store     = NormalCacheStore::createNormalCacheStore(params);
    ASSERT_NE(retained_store, nullptr);
    auto messager = std::dynamic_pointer_cast<TcpMessager>(retained_store->messager_);
    ASSERT_NE(messager, nullptr);
    auto wrapper = std::dynamic_pointer_cast<arpc::KMonitorANetServerMetricReporter>(
        messager->tcp_server_->rpc_server_->_metricReporter);
    ASSERT_NE(wrapper, nullptr);
    ASSERT_NE(wrapper->_anetReporter, nullptr);
    // Retain the real store/server owners but replace the reporter callback with
    // a barrier. Reporter destruction must happen at explicit stop, not dtor.
    auto barrier                                     = std::make_shared<ReporterBarrier>();
    auto reporter                                    = std::make_shared<BarrierANetReporter>(barrier);
    wrapper->_anetReporter                           = reporter;
    std::weak_ptr<BarrierANetReporter> weak_reporter = reporter;
    reporter->_reportThread                          = autil::LoopThread::createLoopThread(
        [barrier]() {
            if (!barrier->first_callback.exchange(false)) {
                return;
            }
            barrier->entered.set_value();
            barrier->released.wait();
            barrier->callback_saw_live_dependency = barrier->dependency_alive.load();
            barrier->callback_done                = true;
        },
        1000000,
        "ANetStopBarrier");
    EXPECT_NE(reporter->_reportThread, nullptr);
    reporter.reset();
    wrapper.reset();
    EXPECT_EQ(barrier->entered.get_future().wait_for(5s), std::future_status::ready);
    auto stopping = std::async(std::launch::async, [&]() { retained_store->stopTransport(); });
    EXPECT_EQ(barrier->stopping.get_future().wait_for(5s), std::future_status::ready);
    EXPECT_EQ(stopping.wait_for(0s), std::future_status::timeout);
    EXPECT_FALSE(barrier->callback_done.load());
    // Always release before any fatal assertion so a negative mutation cannot
    // strand a callback or an async future in test cleanup.
    barrier->release.set_value();
    EXPECT_EQ(stopping.wait_for(5s), std::future_status::ready);
    stopping.get();
    EXPECT_TRUE(barrier->callback_done.load());
    EXPECT_TRUE(barrier->callback_saw_live_dependency.load());
    EXPECT_TRUE(weak_reporter.expired());
    barrier->dependency_alive = false;
    EXPECT_NE(retained_store, nullptr);
    retained_store->stopTransport();
}

TEST_F(TcpMessagerShutdownTest, StopIsExplicitAndIdempotent) {
    auto buffer_store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto messager     = std::make_shared<TcpMessager>(memory_util_, buffer_store, kmonitor::MetricsReporterPtr());
    MessagerInitParams params{autil::NetUtil::randomPort()};
    ASSERT_TRUE(messager->init(params));

    // The explicit shutdown path must be safe to call, and safe to repeat.
    messager->stop();
    messager->stop();
    SUCCEED();
}

TEST_F(TcpMessagerShutdownTest, StopWithoutInitIsSafe) {
    auto buffer_store = std::make_shared<RequestBlockBufferStore>(memory_util_);
    auto messager     = std::make_shared<TcpMessager>(memory_util_, buffer_store, kmonitor::MetricsReporterPtr());
    // Never init'ed: stop must not touch uninitialized members.
    messager->stop();
    SUCCEED();
}

TEST_F(TcpMessagerShutdownTest, NormalCacheStoreStopTransportChain) {
    CacheStoreInitParams params;
    params.listen_port      = autil::NetUtil::randomPort();
    params.rdma_listen_port = 0;
    params.rdma_mode        = false;
    params.thread_count     = 2;
    auto store              = NormalCacheStore::createNormalCacheStore(params);
    ASSERT_TRUE(store != nullptr);

    store->stopTransport();
    store->stopTransport();  // idempotent
    SUCCEED();
}

}  // namespace rtp_llm
