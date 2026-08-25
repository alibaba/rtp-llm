#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/test/MockAsyncContext.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace test {

namespace {

class BlockingAsyncContext final: public AsyncContext {
public:
    void setDone(bool done) {
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = done;
            if (done_) {
                callbacks.swap(callbacks_);
            }
        }
        cv_.notify_all();
        for (auto& callback : callbacks) {
            callback(success_.load(std::memory_order_relaxed) ?
                         ErrorInfo::OkStatus() :
                         ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "blocking context failed"));
        }
    }

    void setSuccess(bool success) {
        success_.store(success, std::memory_order_relaxed);
    }

    void waitDone() override {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return done_; });
    }

    void onDone(DoneCallback callback) override {
        if (!callback) {
            return;
        }
        bool run_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (done_) {
                run_now = true;
            } else {
                callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback(success() ? ErrorInfo::OkStatus() :
                                 ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "blocking context failed"));
        }
    }

    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_;
    }

    bool success() const override {
        return success_.load(std::memory_order_relaxed);
    }

private:
    mutable std::mutex              mutex_;
    mutable std::condition_variable cv_;
    bool                            done_{false};
    std::atomic<bool>               success_{true};
    std::vector<DoneCallback>       callbacks_;
};

}  // namespace

TEST(AsyncContextTest, CompletedAsyncContext_SuccessStatus) {
    CompletedAsyncContext context(ErrorInfo::OkStatus());
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());

    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());
}

TEST(AsyncContextTest, CompletedAsyncContext_FailureStatusPreservesErrorInfo) {
    CompletedAsyncContext context(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_FALSE(context.errorInfo().ok());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.errorInfo().ToString(), "invalid block transfer request");

    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.errorInfo().ToString(), "invalid block transfer request");
}

TEST(AsyncContextTest, CompletedAsyncContext_OnDoneRunsImmediately) {
    CompletedAsyncContext context(ErrorInfo(ErrorCode::INVALID_PARAMS, "already done"));
    size_t                callback_count = 0;

    context.onDone([&](ErrorInfo error) {
        EXPECT_EQ(error.code(), ErrorCode::INVALID_PARAMS);
        ++callback_count;
    });

    EXPECT_EQ(callback_count, 1u);
}

TEST(AsyncContextTest, FusedAsyncContext_DoneTrue_WhenEmptyOrAllDoneOrNull) {
    auto c1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto c2 = std::shared_ptr<AsyncContext>{nullptr};
    ON_CALL(*c1, done()).WillByDefault(testing::Return(true));

    FusedAsyncContext fused_empty({});
    EXPECT_TRUE(fused_empty.done());

    FusedAsyncContext fused({c1, c2});
    EXPECT_TRUE(fused.done());
}

TEST(AsyncContextTest, FusedAsyncContext_DoneFalse_WhenAnyNotDone) {
    auto done_ctx     = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto not_done_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*done_ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*not_done_ctx, done()).WillByDefault(testing::Return(false));

    FusedAsyncContext fused({done_ctx, not_done_ctx});
    EXPECT_FALSE(fused.done());
}

TEST(AsyncContextTest, FusedAsyncContext_SuccessTrue_WhenAllSuccessOrNull) {
    auto ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ok, success()).WillByDefault(testing::Return(true));

    FusedAsyncContext fused({ok, nullptr});
    EXPECT_TRUE(fused.success());
}

TEST(AsyncContextTest, FusedAsyncContext_SuccessFalse_WhenAnyFail) {
    auto ok  = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto bad = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ok, success()).WillByDefault(testing::Return(true));
    ON_CALL(*bad, success()).WillByDefault(testing::Return(false));

    FusedAsyncContext fused({ok, bad});
    EXPECT_FALSE(fused.success());
}

TEST(AsyncContextTest, FusedAsyncContext_OnDoneWaitsForAllChildrenAndKeepsFirstError) {
    auto first  = std::make_shared<BlockingAsyncContext>();
    auto second = std::make_shared<BlockingAsyncContext>();
    first->setSuccess(false);
    FusedAsyncContext fused({first, second});

    size_t    callback_count = 0;
    ErrorCode result_code    = ErrorCode::NONE_ERROR;
    fused.onDone([&](ErrorInfo error) {
        result_code = error.code();
        ++callback_count;
    });

    first->setDone(true);
    EXPECT_EQ(callback_count, 0u);
    second->setDone(true);
    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(result_code, ErrorCode::EXECUTION_EXCEPTION);
}

}  // namespace test
}  // namespace rtp_llm
