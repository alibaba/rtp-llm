#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include <grpcpp/alarm.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace rtp_llm::test {
namespace {
using Responses    = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
using Worker       = Responses::WorkerRpcContext;
using CopyResponse = MemoryOperationResponsePB;

std::shared_ptr<Worker> makeWorker(CopyResponse::ErrorCode error) {
    auto worker            = std::make_shared<Worker>();
    worker->client_context = std::make_shared<grpc::ClientContext>();
    worker->status         = grpc::Status::OK;
    worker->response.mutable_mem_response()->set_success(error == CopyResponse::NONE);
    worker->response.mutable_mem_response()->set_error_code(error);
    return worker;
}

std::shared_ptr<Responses> completedResponses(const std::vector<std::shared_ptr<Worker>>& workers) {
    auto                                      result = std::make_shared<Responses>(workers);
    std::vector<std::unique_ptr<grpc::Alarm>> alarms;
    for (size_t rank = 0; rank < workers.size(); ++rank) {
        alarms.push_back(std::make_unique<grpc::Alarm>());
        alarms.back()->Set(&workers[rank]->completion_queue,
                           std::chrono::system_clock::now(),
                           reinterpret_cast<void*>(static_cast<intptr_t>(rank)));
    }
    result->waitDone();
    return result;
}
}  // namespace

TEST(MemoryAsyncContextCrcTest, DoneCallbackReceivesConnectorErrorOnce) {
    for (auto expected : {CopyResponse::NONE,
                          CopyResponse::COPY_FAILED,
                          CopyResponse::INVALID_REQUEST,
                          CopyResponse::IO_FAILED,
                          CopyResponse::CRC_COMPUTE_FAILED,
                          CopyResponse::CRC_MISMATCH}) {
        SCOPED_TRACE(CopyResponse::ErrorCode_Name(expected));
        int                callbacks = 0;
        MemoryAsyncContext context([&](CopyResponse::ErrorCode error) {
            EXPECT_EQ(error, expected);
            ++callbacks;
        });
        context.setBroadcastResult(completedResponses({makeWorker(expected), makeWorker(expected)}));
        EXPECT_FALSE(context.errorInfo().hasError());
        context.waitDone();
        context.waitDone();
        EXPECT_TRUE(context.done());
        EXPECT_EQ(context.success(), expected == CopyResponse::NONE);
        EXPECT_EQ(context.errorInfo().hasError(), expected != CopyResponse::NONE);
        EXPECT_EQ(callbacks, 1);
    }
}

TEST(MemoryAsyncContextCrcTest, MissingOrUnclassifiedResponseIsCopyFailure) {
    for (int scenario = 0; scenario < 3; ++scenario) {
        SCOPED_TRACE(scenario);
        auto worker = makeWorker(CopyResponse::NONE);
        if (scenario == 0) {
            worker->response.clear_mem_response();
        } else if (scenario == 1) {
            worker->response.mutable_mem_response()->set_success(false);
        } else {
            worker->response.mutable_mem_response()->set_error_code(static_cast<CopyResponse::ErrorCode>(999));
        }
        MemoryAsyncContext context([](CopyResponse::ErrorCode error) {
            EXPECT_EQ(error, CopyResponse::COPY_FAILED);
        });
        context.setBroadcastResult(completedResponses({worker, makeWorker(CopyResponse::NONE)}));
        context.waitDone();
        EXPECT_EQ(context.errorInfo().code(), ErrorCode::KV_CACHE_REUSE_ERROR);
    }
}

TEST(MemoryAsyncContextCrcTest, ConfirmedMismatchSurvivesOtherRankRpcFailure) {
    for (bool mismatch_on_failed_rpc : {false, true}) {
        SCOPED_TRACE(mismatch_on_failed_rpc);
        auto first    = makeWorker(mismatch_on_failed_rpc ? CopyResponse::CRC_MISMATCH : CopyResponse::COPY_FAILED);
        first->status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "RPC failed");
        auto second   = makeWorker(mismatch_on_failed_rpc ? CopyResponse::NONE : CopyResponse::CRC_MISMATCH);
        MemoryAsyncContext context([&](CopyResponse::ErrorCode error) {
            EXPECT_EQ(error, mismatch_on_failed_rpc ? CopyResponse::RPC_FAILED : CopyResponse::CRC_MISMATCH);
        });
        context.setBroadcastResult(completedResponses({first, second}));
        context.waitDone();
        EXPECT_FALSE(context.success());
    }
}

TEST(MemoryAsyncContextCrcTest, NullBroadcastReportsRpcFailure) {
    int                callbacks = 0;
    MemoryAsyncContext context([&](CopyResponse::ErrorCode error) {
        EXPECT_EQ(error, CopyResponse::RPC_FAILED);
        ++callbacks;
    });
    EXPECT_FALSE(context.errorInfo().hasError());
    context.setBroadcastResult(nullptr);
    context.waitDone();
    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::KV_CACHE_REUSE_ERROR);
    EXPECT_EQ(callbacks, 1);
}

TEST(MemoryAsyncContextCrcTest, AllRanksFinishBeforeSingleFailureCallbackAndReferenceRelease) {
    for (auto status : {grpc::StatusCode::OK, grpc::StatusCode::UNAVAILABLE, grpc::StatusCode::CANCELLED}) {
        SCOPED_TRACE(static_cast<int>(status));
        auto first                = makeWorker(CopyResponse::COPY_FAILED);
        auto second               = makeWorker(CopyResponse::COPY_FAILED);
        first->status             = grpc::Status(status, status == grpc::StatusCode::OK ? "" : "RPC failed");
        auto            responses = std::make_shared<Responses>(std::vector<std::shared_ptr<Worker>>{first, second});
        BlockPoolConfig config;
        config.block_num = 3;
        auto pool        = std::make_shared<BlockPool>(config, AllocationType::HOST);
        // Exercise real reference accounting without allocating CPU/GPU backing storage.
        pool->initFreeBlocks();
        const auto blocks = pool->malloc(2);
        ASSERT_EQ(blocks.size(), 2u);
        pool->connectorReference(blocks[1]);
        pool->requestFree(blocks[1]);
        std::vector<int>    events;
        std::atomic<int>    callbacks{0};
        MemoryAsyncContext* observed_context   = nullptr;
        auto                reference          = std::shared_ptr<int>(new int(0), [&](int* value) {
            EXPECT_FALSE(observed_context->done());
            pool->requestFree(blocks[0]);
            pool->connectorFree(blocks[1]);
            events.push_back(2);
            delete value;
        });
        std::weak_ptr<int>  retained_reference = reference;
        MemoryAsyncContext  context([&, reference](CopyResponse::ErrorCode error) {
            EXPECT_EQ(error, status == grpc::StatusCode::OK ? CopyResponse::COPY_FAILED : CopyResponse::RPC_FAILED);
            EXPECT_FALSE(observed_context->done());
            ++callbacks;
            events.push_back(1);
         });
        observed_context = &context;
        reference.reset();
        context.setBroadcastResult(responses);
        grpc::Alarm first_done;
        grpc::Alarm second_done;
        first_done.Set(&first->completion_queue, std::chrono::system_clock::now(), nullptr);
        std::thread waiter([&] { context.waitDone(); });
        std::thread other_waiter([&] { context.waitDone(); });
        const auto  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (responses->finished_count_.load() != 1 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(responses->finished_count_.load(), 1);
        EXPECT_FALSE(context.done());
        EXPECT_FALSE(context.errorInfo().hasError());
        EXPECT_EQ(callbacks.load(), 0);
        EXPECT_FALSE(retained_reference.expired());
        EXPECT_EQ(pool->requestRefBlocksNum(), 1u);
        EXPECT_EQ(pool->connectorRefBlocksNum(), 1u);
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        second_done.Set(&second->completion_queue, std::chrono::system_clock::now(), nullptr);
        waiter.join();
        other_waiter.join();
        context.waitDone();
        EXPECT_TRUE(context.done());
        EXPECT_FALSE(context.success());
        EXPECT_EQ(context.errorInfo().code(), ErrorCode::KV_CACHE_REUSE_ERROR);
        EXPECT_EQ(callbacks.load(), 1);
        EXPECT_EQ(events, (std::vector<int>{1, 2}));
        EXPECT_TRUE(retained_reference.expired());
        EXPECT_EQ(pool->requestRefBlocksNum(), 0u);
        EXPECT_EQ(pool->connectorRefBlocksNum(), 0u);
        EXPECT_EQ(pool->freeBlocksNum(), 2u);
    }
}
}  // namespace rtp_llm::test
