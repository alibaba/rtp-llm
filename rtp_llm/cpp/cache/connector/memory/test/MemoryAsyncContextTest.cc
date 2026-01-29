// Copyright (c) RTP-LLM

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>

#include "grpcpp/alarm.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {

// --------------------------------- MemoryAsyncContextTest ---------------------------------

class MemoryAsyncContextTest: public ::testing::Test {
protected:
    // NOTE: This test file needs a "completed" BroadcastResult without running real RPCs.
    // We achieve this by scheduling a grpc::Alarm event onto each worker's CompletionQueue,
    // then calling BroadcastResult::waitDone() once to finalize its internal success flag.
    using MemoryBroadcastResultT = rtp_llm::BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    using MemoryWorkerCtxT       = typename MemoryBroadcastResultT::WorkerRpcContext;

    static std::shared_ptr<MemoryBroadcastResultT>
    makeCompletedBroadcastResult(const std::vector<std::shared_ptr<MemoryWorkerCtxT>>& workers) {
        // BroadcastResult::waitDone() may call TryCancel() on all contexts when any status is not OK.
        for (const auto& w : workers) {
            if (w && !w->client_context) {
                w->client_context = std::make_shared<grpc::ClientContext>();
            }
        }

        auto result = std::make_shared<MemoryBroadcastResultT>(workers);

        // Post one event per worker so BroadcastResult::waitDone() can finish immediately.
        std::vector<std::unique_ptr<grpc::Alarm>> alarms;
        alarms.reserve(workers.size());
        for (size_t i = 0; i < workers.size(); ++i) {
            alarms.emplace_back(std::make_unique<grpc::Alarm>());
            alarms.back()->Set(&workers[i]->completion_queue,
                               std::chrono::system_clock::now(),
                               reinterpret_cast<void*>(static_cast<intptr_t>(i)));
        }

        result->waitDone();
        return result;
    }
};

TEST_F(MemoryAsyncContextTest, success_ReturnFalse_WhenBroadcastResultNotSuccess) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status(grpc::StatusCode::CANCELLED, "cancelled");
    worker0->response.mutable_mem_response()->set_success(true);

    auto result = makeCompletedBroadcastResult({worker0});

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryAsyncContextTest, success_ReturnFalse_WhenAnyResponseMissingMemResponse) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();  // default: no mem_response
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status::OK;
    auto result             = makeCompletedBroadcastResult({worker0});

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryAsyncContextTest, success_ReturnFalse_WhenAnyMemResponseFailed) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status::OK;
    worker0->response.mutable_mem_response()->set_success(true);
    auto worker1            = std::make_shared<MemoryWorkerCtxT>();
    worker1->client_context = std::make_shared<grpc::ClientContext>();
    worker1->status         = grpc::Status::OK;
    worker1->response.mutable_mem_response()->set_success(false);

    auto result = makeCompletedBroadcastResult({worker0, worker1});

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryAsyncContextTest, success_ReturnTrue_WhenAllResponsesSuccess) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status::OK;
    worker0->response.mutable_mem_response()->set_success(true);
    auto worker1            = std::make_shared<MemoryWorkerCtxT>();
    worker1->client_context = std::make_shared<grpc::ClientContext>();
    worker1->status         = grpc::Status::OK;
    worker1->response.mutable_mem_response()->set_success(true);

    auto result = makeCompletedBroadcastResult({worker0, worker1});

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_TRUE(ctx->success());
}

TEST_F(MemoryAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNullAndCallbackCalledOnce) {
    int  callback_cnt = 0;
    bool last_ok      = true;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(/*done_callback=*/cb);
    ctx->setBroadcastResult(nullptr);
    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_FALSE(last_ok);

    // Second call should be no-op.
    ctx->waitDone();
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(ctx->done());
}

TEST_F(MemoryAsyncContextTest, waitDone_ReturnsImmediately_WhenBroadcastResultNotSet_ThenCallbackOnce) {
    std::atomic<int>  callback_cnt{0};
    std::atomic<bool> last_ok{true};
    auto              cb = [&](bool ok) {
        callback_cnt.fetch_add(1);
        last_ok.store(ok);
    };

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(cb);
    EXPECT_FALSE(ctx->done());

    std::thread t([&]() { ctx->waitDone(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // 如果 broadcast_result_ 还没设置，waitDone() 不会阻塞，而是按失败处理并回调一次。
    EXPECT_TRUE(ctx->done());
    EXPECT_EQ(callback_cnt.load(), 1);
    EXPECT_FALSE(last_ok.load());

    ctx->setBroadcastResult(nullptr);
    t.join();
    EXPECT_TRUE(ctx->done());
    EXPECT_EQ(callback_cnt.load(), 1);
}

TEST_F(MemoryAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNonNullAndCallbackReceivesSuccess) {
    // Empty worker contexts => BroadcastResult::waitDone() returns immediately and sets all_request_success_ = true.
    auto result = std::make_shared<MemoryBroadcastResultT>(std::vector<std::shared_ptr<MemoryWorkerCtxT>>{});

    int  callback_cnt = 0;
    bool last_ok      = false;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(cb);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());  // default all_request_success_ is false before waitDone().

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(last_ok);

    // Second call should be no-op.
    ctx->waitDone();
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(ctx->done());
}

TEST_F(MemoryAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNonNullAndDoneCallbackNull) {
    auto result = std::make_shared<MemoryBroadcastResultT>(std::vector<std::shared_ptr<MemoryWorkerCtxT>>{});
    auto ctx    = std::make_shared<rtp_llm::MemoryAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
}

TEST_F(MemoryAsyncContextTest, waitDone_IsIdempotent_CallbackOnlyOnce) {
    int  callback_cnt = 0;
    bool last_ok      = false;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    // Use empty worker contexts: BroadcastResult::waitDone() completes immediately and marks success.
    auto result = std::make_shared<MemoryBroadcastResultT>(std::vector<std::shared_ptr<MemoryWorkerCtxT>>{});

    auto ctx = std::make_shared<rtp_llm::MemoryAsyncContext>(cb);
    ctx->setBroadcastResult(result);
    ctx->waitDone();
    ctx->waitDone();

    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(last_ok);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}