#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace rtp_llm {
namespace {

class TestSyncContext: public SyncContext {
public:
    TestSyncContext(): SyncContext(nullptr, false) {}

    void preparePendingResult(CacheStoreLoadDeadline deadline, CheckCancelFunc check_cancel_func = nullptr) {
        deadline_ = deadline;
        if (check_cancel_func == nullptr) {
            check_cancel_func_.reset();
        } else {
            check_cancel_func_ = std::make_shared<const CheckCancelFunc>(std::move(check_cancel_func));
        }
        expect_layer_cnt_ = 1;
        done_layer_cnt_   = 0;
    }

    void setDeadline(CacheStoreLoadDeadline deadline) {
        deadline_ = deadline;
    }

    bool finalizeAt(bool cancellation_requested, CacheStoreLoadDeadline now) {
        CheckCancelFuncHolder retired_check_cancel_func;
        bool                  finalized = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finalized = finalizeDeadlineOrCancellationLocked(
                cancellation_requested, retired_check_cancel_func, now);
        }
        retired_check_cancel_func.reset();
        return finalized;
    }

private:
    bool doCall(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreLoadDeadline) override {
        return true;
    }
};

struct ReentrantProbeState {
    std::mutex              mutex;
    std::condition_variable cond;
    bool                    armed                          = false;
    bool                    entered                        = false;
    bool                    context_call_finished          = false;
    bool                    context_call_finished_in_scope = false;
};

class ReentrantDestructorProbe {
public:
    explicit ReentrantDestructorProbe(std::shared_ptr<ReentrantProbeState> state): state_(std::move(state)) {}

    ~ReentrantDestructorProbe() {
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (!state_->armed || state_->entered) {
            return;
        }
        state_->entered = true;
        state_->cond.notify_all();
        state_->context_call_finished_in_scope = state_->cond.wait_for(
            lock, std::chrono::seconds(5), [this] { return state_->context_call_finished; });
    }

    bool operator()() const {
        return false;
    }

private:
    std::shared_ptr<ReentrantProbeState> state_;
};

class CopyCountingCancelProbe {
public:
    explicit CopyCountingCancelProbe(std::shared_ptr<int> copy_count): copy_count_(std::move(copy_count)) {}

    CopyCountingCancelProbe(const CopyCountingCancelProbe& other): copy_count_(other.copy_count_) {
        ++*copy_count_;
    }

    CopyCountingCancelProbe(CopyCountingCancelProbe&&) noexcept = default;

    bool operator()() const {
        return true;
    }

private:
    std::shared_ptr<int> copy_count_;
};

static_assert(std::is_same_v<decltype(std::declval<const TestSyncContext&>().getErrorInfo()), ErrorInfo>,
              "getErrorInfo must return an owned snapshot");

TEST(LoadContextConcurrencyTest, pastDeadlineSuccessCallbackFinalizesAsTimeout) {
    auto context = std::make_shared<TestSyncContext>();
    context->preparePendingResult(CacheStoreLoadClock::now() - std::chrono::milliseconds(1));

    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(true, CacheStoreErrorCode::None, request);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
}

TEST(LoadContextConcurrencyTest, timeoutResultCannotBeOverwrittenByLateCallback) {
    auto context = std::make_shared<TestSyncContext>();
    context->preparePendingResult(CacheStoreLoadClock::now() - std::chrono::milliseconds(1));
    context->waitDone();

    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(false, CacheStoreErrorCode::LoadRdmaWriteFailed, request);

    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
}

TEST(LoadContextConcurrencyTest, cancelledResultCannotBeOverwrittenByLateCallback) {
    auto context = std::make_shared<TestSyncContext>();
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(1), []() { return true; });
    context->waitDone();

    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(false, CacheStoreErrorCode::LoadRdmaWriteFailed, request);

    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextConcurrencyTest, cancelledSuccessCallbackFinalizesAsCancelled) {
    auto context = std::make_shared<TestSyncContext>();
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(1), []() { return true; });

    context->waitDone();

    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(true, CacheStoreErrorCode::None, request);

    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextConcurrencyTest, cancellationRequestedBeforeFinalSuccessWins) {
    auto context   = std::make_shared<TestSyncContext>();
    auto cancelled = std::make_shared<std::atomic_bool>(false);
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(1),
                                  [cancelled]() { return cancelled->load(std::memory_order_acquire); });

    cancelled->store(true, std::memory_order_release);
    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(true, CacheStoreErrorCode::None, request);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextConcurrencyTest, completionWakeRechecksCancellationBeforeSuccess) {
    auto context   = std::make_shared<TestSyncContext>();
    auto cancelled = std::make_shared<std::atomic_bool>(false);
    std::mutex              observation_mutex;
    std::condition_variable observation_cond;
    bool                    observed_not_cancelled = false;
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(5), [&]() {
        const bool current = cancelled->load(std::memory_order_acquire);
        if (!current) {
            {
                std::lock_guard<std::mutex> lock(observation_mutex);
                observed_not_cancelled = true;
            }
            observation_cond.notify_all();
        }
        return current;
    });

    std::thread waiter([&]() { context->waitDone(); });
    bool        observed = false;
    {
        std::unique_lock<std::mutex> lock(observation_mutex);
        observed = observation_cond.wait_for(
            lock, std::chrono::seconds(5), [&]() { return observed_not_cancelled; });
    }

    cancelled->store(true, std::memory_order_release);
    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(true, CacheStoreErrorCode::None, request);
    waiter.join();

    ASSERT_TRUE(observed);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextConcurrencyTest, cancelCheckerRunsWithoutContextMutex) {
    auto context = std::make_shared<TestSyncContext>();
    auto state   = std::make_shared<ReentrantProbeState>();

    std::thread context_caller([&]() {
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (!state->cond.wait_for(lock, std::chrono::seconds(5), [&]() { return state->entered; })) {
                return;
            }
        }
        (void)context->success();
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->context_call_finished = true;
        }
        state->cond.notify_all();
    });

    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(1), [&]() {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->entered = true;
        state->cond.notify_all();
        state->context_call_finished_in_scope = state->cond.wait_for(
            lock, std::chrono::seconds(5), [&]() { return state->context_call_finished; });
        return true;
    });

    context->waitDone();
    context_caller.join();

    EXPECT_TRUE(state->context_call_finished_in_scope);
    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextConcurrencyTest, cancelCheckerSnapshotDoesNotCopyCallableUnderMutex) {
    auto context    = std::make_shared<TestSyncContext>();
    auto copy_count = std::make_shared<int>(0);
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(1),
                                  CopyCountingCancelProbe(copy_count));
    *copy_count = 0;

    context->waitDone();

    EXPECT_EQ(*copy_count, 0);
    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
}

TEST(LoadContextConcurrencyTest, cancelTargetDestructorRunsWithoutContextMutex) {
    auto context = std::make_shared<TestSyncContext>();
    auto state   = std::make_shared<ReentrantProbeState>();
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::seconds(1),
                                  ReentrantDestructorProbe(state));
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->armed = true;
    }

    std::thread context_caller([&]() {
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (!state->cond.wait_for(lock, std::chrono::seconds(5), [&]() { return state->entered; })) {
                return;
            }
        }
        (void)context->success();
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->context_call_finished = true;
        }
        state->cond.notify_all();
    });

    const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");
    context->updateResult(true, CacheStoreErrorCode::None, request);
    context->waitDone();
    context_caller.join();

    EXPECT_TRUE(state->entered);
    EXPECT_TRUE(state->context_call_finished_in_scope);
    EXPECT_TRUE(context->success());
}

TEST(LoadContextConcurrencyTest, terminalFinalizerUsesExactDeadlineAndDeadlinePrecedence) {
    const auto deadline = CacheStoreLoadDeadline(std::chrono::seconds(2));
    const auto tick     = CacheStoreLoadClock::duration(1);

    auto before_deadline = std::make_shared<TestSyncContext>();
    before_deadline->preparePendingResult(deadline);
    EXPECT_FALSE(before_deadline->finalizeAt(false, deadline - tick));

    auto at_deadline = std::make_shared<TestSyncContext>();
    at_deadline->preparePendingResult(deadline);
    EXPECT_TRUE(at_deadline->finalizeAt(false, deadline));
    EXPECT_EQ(at_deadline->getErrorInfo().code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);

    auto after_deadline = std::make_shared<TestSyncContext>();
    after_deadline->preparePendingResult(deadline);
    EXPECT_TRUE(after_deadline->finalizeAt(false, deadline + tick));
    EXPECT_EQ(after_deadline->getErrorInfo().code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);

    auto cancelled_before_deadline = std::make_shared<TestSyncContext>();
    cancelled_before_deadline->preparePendingResult(deadline);
    EXPECT_TRUE(cancelled_before_deadline->finalizeAt(true, deadline - tick));
    EXPECT_EQ(cancelled_before_deadline->getErrorInfo().code(), ErrorCode::CANCELLED);

    auto cancelled_at_deadline = std::make_shared<TestSyncContext>();
    cancelled_at_deadline->preparePendingResult(deadline);
    EXPECT_TRUE(cancelled_at_deadline->finalizeAt(true, deadline));
    EXPECT_EQ(cancelled_at_deadline->getErrorInfo().code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
}

TEST(LoadContextConcurrencyTest, callbackWaitRacePreservesPastDeadline) {
    for (int iteration = 0; iteration < 256; ++iteration) {
        auto context = std::make_shared<TestSyncContext>();
        context->preparePendingResult(CacheStoreLoadClock::now() - std::chrono::milliseconds(1));
        const auto request = std::make_shared<RequestBlockBuffer>("request", "layer");

        std::thread callback_thread(
            [&]() { context->updateResult(true, CacheStoreErrorCode::None, request); });
        std::thread waiter_thread([&]() { context->waitDone(); });
        callback_thread.join();
        waiter_thread.join();

        EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
    }
}

TEST(LoadContextConcurrencyTest, repeatedWaitPreservesCancelledTerminalResult) {
    auto context       = std::make_shared<TestSyncContext>();
    int  cancel_checks = 0;
    context->preparePendingResult(CacheStoreLoadClock::now() + std::chrono::milliseconds(150),
                                  [&cancel_checks]() { return cancel_checks++ == 0; });

    context->waitDone();
    context->setDeadline(CacheStoreLoadClock::now() - std::chrono::milliseconds(1));
    context->waitDone();

    EXPECT_EQ(context->getErrorInfo().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(cancel_checks, 1);
}

}  // namespace
}  // namespace rtp_llm
