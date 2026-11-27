#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <array>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace rtp_llm {
namespace {

class Collector: public ::accl::barex::AcclMetricReporter {
public:
    Collector(): AcclMetricReporter(::accl::barex::AcclMetricReporterConfig(false)) {}

    void reportRdmaMetric(const std::string&, uint64_t, const std::map<std::string, std::string>&) override {}
};

class MemoryStub: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool, uint64_t = 0) override {
        return true;
    }
    bool deregUserMr(void*, bool) override {
        ++deregistrations;
        return true;
    }
    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return true;
    }
    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return true;
    }
    bool isRdmaMode() override {
        return false;
    }
    int deregistrations{0};
};

struct Collectors {
    std::mutex              mutex;
    std::condition_variable changed;
    std::array<bool, 2>     started{};
    std::atomic<bool>       monitor_alive{true};
    std::atomic<bool>       used_stopped_monitor{false};

    std::shared_ptr<Collector> start(size_t index) {
        auto reporter               = std::make_shared<Collector>();
        reporter->_collectThreadPtr = autil::LoopThread::createLoopThread(
            [this, index]() {
                if (!monitor_alive.load()) {
                    used_stopped_monitor = true;
                }
                std::lock_guard<std::mutex> lock(mutex);
                started[index] = true;
                changed.notify_all();
            },
            1000,
            "rdma_metrics_test");
        return reporter;
    }

    bool waitUntilRunning() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(2), [this]() { return started[0] && started[1]; });
    }
};

TEST(RdmaMetricsShutdownTest, StopsTransportWhileRetainedCacheOwnersCanReleaseMemory) {
    // Keep the state alive until every retained collector has been destroyed,
    // including when an assertion fails before explicit shutdown.
    Collectors state;
    auto       server_reporter = state.start(0);
    auto       client_reporter = state.start(1);
    ASSERT_TRUE(server_reporter->_collectThreadPtr);
    ASSERT_TRUE(client_reporter->_collectThreadPtr);
    ASSERT_TRUE(state.waitUntilRunning());

    auto store                         = std::shared_ptr<NormalCacheStore>(new NormalCacheStore);
    auto memory                        = std::make_shared<MemoryStub>();
    store->memory_util_                = memory;
    store->request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory);
    auto messager          = std::make_shared<RdmaMessager>(nullptr, store->request_block_buffer_store_, nullptr);
    messager->rdma_server_ = std::make_shared<RdmaServer>(nullptr);
    messager->rdma_client_ = std::make_shared<RdmaClient>(nullptr);
    messager->rdma_server_->metrics_reporter_ = server_reporter;
    messager->rdma_client_->metrics_reporter_ = client_reporter;
    store->messager_                          = messager;
    std::weak_ptr<RdmaMessager> transport     = messager;
    messager.reset();

    // Real shutdown can race with a repeated stop, and retained cache/tensor
    // owners must not postpone collector shutdown until their destructors.
    auto                     retained_store = store;
    std::vector<std::thread> stops;
    for (size_t index = 0; index < 8; ++index) {
        stops.emplace_back([store]() { store->stop(); });
    }
    for (auto& thread : stops) {
        thread.join();
    }
    ASSERT_FALSE(server_reporter->_collectThreadPtr);
    ASSERT_FALSE(client_reporter->_collectThreadPtr);
    EXPECT_GT(server_reporter.use_count(), 0);
    EXPECT_GT(client_reporter.use_count(), 0);
    EXPECT_EQ(retained_store.get(), store.get());
    EXPECT_TRUE(transport.expired());
    ASSERT_EQ(retained_store->getMemoryUtil(), memory);
    EXPECT_TRUE(retained_store->getMemoryUtil()->deregUserMr(nullptr, true));
    EXPECT_EQ(memory->deregistrations, 1);

    state.monitor_alive = false;
    store->stop();
    EXPECT_FALSE(state.used_stopped_monitor.load());
}

TEST(RdmaMetricsShutdownTest, StopJoinsAnAlreadyRunningCollector) {
    std::mutex              mutex;
    std::condition_variable changed;
    bool                    entered = false, release = false;
    auto                    reporter = std::make_shared<Collector>();
    std::future<void>       stopped;
    struct Release {
        std::mutex&              mutex;
        std::condition_variable& changed;
        bool&                    release;
        ~Release() {
            std::lock_guard<std::mutex> lock(mutex);
            release = true;
            changed.notify_all();
        }
    } unblock{mutex, changed, release};
    reporter->_collectThreadPtr = autil::LoopThread::createLoopThread(
        [&]() {
            std::unique_lock<std::mutex> lock(mutex);
            entered = true;
            changed.notify_all();
            changed.wait(lock, [&]() { return release; });
        },
        1000,
        "blocked_metric");
    ASSERT_TRUE(reporter->_collectThreadPtr);
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(changed.wait_for(lock, std::chrono::seconds(2), [&]() { return entered; }));
    }
    RdmaClient client(nullptr);
    client.metrics_reporter_ = reporter;
    std::promise<void> stopping;
    auto               started = stopping.get_future();
    stopped                    = std::async(std::launch::async, [&]() {
        stopping.set_value();
        client.stopMetricsReporter();
    });
    started.wait();
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
        changed.notify_all();
    }
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    stopped.get();
    EXPECT_FALSE(reporter->_collectThreadPtr);
}

TEST(RdmaMetricsShutdownTest, AcceptsUninitializedAndExpiredReporters) {
    RdmaServer server(nullptr);
    RdmaClient client(nullptr);
    server.stopMetricsReporter();
    client.stopMetricsReporter();
    auto reporter            = std::make_shared<Collector>();
    server.metrics_reporter_ = reporter;
    client.metrics_reporter_ = reporter;
    reporter.reset();
    server.stopMetricsReporter();
    client.stopMetricsReporter();
}

}  // namespace
}  // namespace rtp_llm
