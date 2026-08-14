#include "gtest/gtest.h"

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

class RecordingMessager final: public Messager {
public:
    RecordingMessager(const std::shared_ptr<MemoryUtil>&              memory_util,
                      const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store):
        Messager(memory_util, request_block_buffer_store, nullptr) {}

    bool init(MessagerInitParams) override {
        return true;
    }

    void load(const std::shared_ptr<LoadRequest>&                          request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) override {
        seen_timeout_ms_.store(request->timeout_ms);
        load_count_.fetch_add(1);
        collector->markEnd(true);
        request->callback(true, CacheStoreErrorCode::None);
    }

    int loadCount() const {
        return load_count_.load();
    }

    uint32_t seenTimeoutMs() const {
        return seen_timeout_ms_.load();
    }

private:
    bool generateBlockInfo(::BlockBufferInfo*, const std::shared_ptr<BlockBuffer>&, uint32_t, uint32_t) override {
        return true;
    }

    std::atomic<int>      load_count_{0};
    std::atomic<uint32_t> seen_timeout_ms_{0};
};

class RejectingThreadPool final: public autil::LockFreeThreadPool {
public:
    RejectingThreadPool(): autil::LockFreeThreadPool(1, 1, nullptr, "RejectingLoadTest") {}

    ERROR_TYPE pushWorkItem(autil::WorkItem*, bool is_blocked) override {
        push_count_.fetch_add(1);
        was_blocked_.store(is_blocked);
        return ERROR_POOL_QUEUE_FULL;
    }

    int pushCount() const {
        return push_count_.load();
    }

    bool wasBlocked() const {
        return was_blocked_.load();
    }

private:
    std::atomic<int>  push_count_{0};
    std::atomic<bool> was_blocked_{true};
};

struct WorkerGate {
    std::mutex              mutex;
    std::condition_variable condition;
    std::promise<void>      started;
    bool                    open{false};
};

class CallbackState {
public:
    CacheStoreLoadDoneCallback callback() {
        return [this](bool ok, CacheStoreErrorCode error_code) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++count_;
            ok_         = ok;
            error_code_ = error_code;
            condition_.notify_all();
        };
    }

    bool waitForCount(int expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [this, expected] { return count_ >= expected; });
    }

    int count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    bool ok() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ok_;
    }

    CacheStoreErrorCode errorCode() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_code_;
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable condition_;
    int                     count_{0};
    bool                    ok_{false};
    CacheStoreErrorCode     error_code_{CacheStoreErrorCode::None};
};

class NormalCacheStoreLoadAdmissionTest: public ::testing::Test {
protected:
    void SetUp() override {
        memory_util_               = createMemoryUtilImpl(false);
        request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_);
        messager_                   = std::make_shared<RecordingMessager>(memory_util_, request_block_buffer_store_);
        thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(1, 1, nullptr, "LoadAdmissionTest");
        ASSERT_TRUE(thread_pool_->start());

        cache_store_.reset(new NormalCacheStore);
        cache_store_->memory_util_               = memory_util_;
        cache_store_->request_block_buffer_store_ = request_block_buffer_store_;
        cache_store_->messager_                  = messager_;
        cache_store_->thread_pool_               = thread_pool_;
    }

    void TearDown() override {
        openWorkerGate();
        cache_store_.reset();
    }

    std::shared_ptr<RequestBlockBuffer> makeLoadBuffer(const std::string& request_id) {
        auto byte = std::shared_ptr<void>(new char(0), [](void* ptr) { delete static_cast<char*>(ptr); });
        auto buffer = std::make_shared<RequestBlockBuffer>(request_id);
        buffer->addBlock("block", byte, 1, false, true);
        return buffer;
    }

    void blockWorker() {
        worker_gate_        = std::make_shared<WorkerGate>();
        auto started_future = worker_gate_->started.get_future();
        ASSERT_EQ(autil::ThreadPoolBase::ERROR_NONE,
                  thread_pool_->pushTask([gate = worker_gate_]() {
                      gate->started.set_value();
                      std::unique_lock<std::mutex> lock(gate->mutex);
                      gate->condition.wait(lock, [gate] { return gate->open; });
                  },
                                         false));
        ASSERT_EQ(std::future_status::ready, started_future.wait_for(1s));
    }

    void openWorkerGate() {
        if (worker_gate_ == nullptr) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(worker_gate_->mutex);
            worker_gate_->open = true;
        }
        worker_gate_->condition.notify_all();
    }

protected:
    std::shared_ptr<MemoryUtil>              memory_util_;
    std::shared_ptr<RequestBlockBufferStore> request_block_buffer_store_;
    std::shared_ptr<RecordingMessager>       messager_;
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::shared_ptr<NormalCacheStore>          cache_store_;
    std::shared_ptr<WorkerGate>                worker_gate_;
};

TEST_F(NormalCacheStoreLoadAdmissionTest, FullQueueRejectsLoadWithoutBlocking) {
    thread_pool_->stop();
    auto rejecting_thread_pool = std::make_shared<RejectingThreadPool>();
    cache_store_->thread_pool_  = rejecting_thread_pool;

    CallbackState callback_state;
    cache_store_->load(makeLoadBuffer("overloaded"), callback_state.callback(), "127.0.0.1", 1, 0, 1000, 1, 0);

    EXPECT_EQ(1, rejecting_thread_pool->pushCount());
    EXPECT_FALSE(rejecting_thread_pool->wasBlocked());
    EXPECT_EQ(1, callback_state.count());
    EXPECT_FALSE(callback_state.ok());
    EXPECT_EQ(CacheStoreErrorCode::PushWorkerItemFailed, callback_state.errorCode());
    EXPECT_EQ(0, messager_->loadCount());
}

TEST_F(NormalCacheStoreLoadAdmissionTest, ExpiredQueuedLoadTimesOutWithoutCallingMessager) {
    blockWorker();
    CallbackState callback_state;
    cache_store_->load(makeLoadBuffer("expired"), callback_state.callback(), "127.0.0.1", 1, 0, 5, 1, 0);

    std::this_thread::sleep_for(25ms);
    openWorkerGate();

    ASSERT_TRUE(callback_state.waitForCount(1, 1s));
    EXPECT_EQ(1, callback_state.count());
    EXPECT_FALSE(callback_state.ok());
    EXPECT_EQ(CacheStoreErrorCode::LoadBufferTimeout, callback_state.errorCode());
    EXPECT_EQ(0, messager_->loadCount());
}

TEST_F(NormalCacheStoreLoadAdmissionTest, ZeroTimeoutExpiresBeforeCallingMessager) {
    CallbackState callback_state;
    cache_store_->load(makeLoadBuffer("zero-timeout"), callback_state.callback(), "127.0.0.1", 1, 0, 0, 1, 0);

    ASSERT_TRUE(callback_state.waitForCount(1, 1s));
    EXPECT_EQ(1, callback_state.count());
    EXPECT_FALSE(callback_state.ok());
    EXPECT_EQ(CacheStoreErrorCode::LoadBufferTimeout, callback_state.errorCode());
    EXPECT_EQ(0, messager_->loadCount());
}

TEST_F(NormalCacheStoreLoadAdmissionTest, PositiveSubmillisecondBudgetRoundsUp) {
    const auto now = std::chrono::steady_clock::time_point(std::chrono::seconds(1));
    uint32_t   remaining_timeout_ms = 0;

    EXPECT_TRUE(getCacheStoreLoadRemainingTimeoutMs(
        now + std::chrono::nanoseconds(1), now, remaining_timeout_ms));
    EXPECT_EQ(1, remaining_timeout_ms);
    EXPECT_FALSE(getCacheStoreLoadRemainingTimeoutMs(now, now, remaining_timeout_ms));
    EXPECT_EQ(0, remaining_timeout_ms);
}

TEST_F(NormalCacheStoreLoadAdmissionTest, QueuedLoadForwardsOnlyRemainingTimeout) {
    blockWorker();
    CallbackState callback_state;
    cache_store_->load(makeLoadBuffer("remaining"), callback_state.callback(), "127.0.0.1", 1, 0, 500, 1, 0);

    std::this_thread::sleep_for(50ms);
    openWorkerGate();

    ASSERT_TRUE(callback_state.waitForCount(1, 1s));
    EXPECT_EQ(1, callback_state.count());
    EXPECT_TRUE(callback_state.ok());
    EXPECT_EQ(CacheStoreErrorCode::None, callback_state.errorCode());
    EXPECT_EQ(1, messager_->loadCount());
    EXPECT_GT(messager_->seenTimeoutMs(), 0);
    EXPECT_LT(messager_->seenTimeoutMs(), 500);
}

}  // namespace
}  // namespace rtp_llm
