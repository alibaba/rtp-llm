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
using Responses = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
using Worker    = Responses::WorkerRpcContext;

std::shared_ptr<Worker> makeWorker(bool success) {
    auto worker            = std::make_shared<Worker>();
    worker->client_context = std::make_shared<grpc::ClientContext>();
    worker->status         = grpc::Status::OK;
    worker->response.mutable_mem_response()->set_success(success);
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

TEST(MemoryAsyncContextCrcTest, OnlyCompletedBusinessFailureRejectsReuse) {
    for (bool reject_reuse : {false, true}) {
        for (int scenario = 0; scenario < 4; ++scenario) {
            SCOPED_TRACE(::testing::Message() << "reject_reuse=" << reject_reuse << " scenario=" << scenario);
            auto first  = makeWorker(scenario != 3);
            auto second = makeWorker(scenario == 0);
            if (scenario == 2) {
                second->response.clear_mem_response();
            }
            int                callbacks = 0, invalidations = 0;
            MemoryAsyncContext context(
                [&](bool success) {
                    EXPECT_EQ(success, scenario == 0);
                    ++callbacks;
                },
                reject_reuse,
                [&] { ++invalidations; });
            context.setBroadcastResult(completedResponses({first, second}));
            context.waitDone();
            context.waitDone();
            const bool rejected = reject_reuse && scenario != 0;
            EXPECT_TRUE(context.done());
            EXPECT_EQ(context.success(), scenario == 0);
            EXPECT_EQ(context.errorInfo().code(), rejected ? ErrorCode::KV_CACHE_REUSE_ERROR : ErrorCode::NONE_ERROR);
            EXPECT_EQ(callbacks, 1);
            EXPECT_EQ(invalidations, rejected ? 1 : 0);
        }
    }
}

TEST(MemoryAsyncContextCrcTest, NullBroadcastKeepsOrdinaryFailure) {
    int                callbacks = 0, invalidations = 0;
    MemoryAsyncContext context(
        [&](bool success) {
            EXPECT_FALSE(success);
            ++callbacks;
        },
        true,
        [&] { ++invalidations; });
    EXPECT_FALSE(context.errorInfo().hasError());
    context.setBroadcastResult(nullptr);
    context.waitDone();
    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_FALSE(context.errorInfo().hasError());
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(invalidations, 0);
}

TEST(MemoryAsyncContextCrcTest, AllRanksFinishBeforeSingleFailureCallbackAndReferenceRelease) {
    for (auto status : {grpc::StatusCode::OK, grpc::StatusCode::UNAVAILABLE, grpc::StatusCode::CANCELLED}) {
        SCOPED_TRACE(static_cast<int>(status));
        auto first                = makeWorker(false);
        auto second               = makeWorker(false);
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
        std::atomic<int>    callbacks{0}, invalidations{0};
        MemoryAsyncContext* observed_context   = nullptr;
        auto                reference          = std::shared_ptr<int>(new int(0), [&](int* value) {
            EXPECT_FALSE(observed_context->done());
            pool->requestFree(blocks[0]);
            pool->connectorFree(blocks[1]);
            events.push_back(3);
            delete value;
        });
        std::weak_ptr<int>  retained_reference = reference;
        MemoryAsyncContext  context(
             [&, reference](bool success) {
                EXPECT_FALSE(success);
                EXPECT_FALSE(observed_context->done());
                ++callbacks;
                events.push_back(2);
             },
             true,
             [&, reference] {
                EXPECT_FALSE(observed_context->done());
                ++invalidations;
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
        EXPECT_EQ(invalidations.load(), 0);
        EXPECT_FALSE(retained_reference.expired());
        EXPECT_EQ(pool->requestRefBlocksNum(), 1u);
        EXPECT_EQ(pool->connectorRefBlocksNum(), 1u);
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        second_done.Set(&second->completion_queue, std::chrono::system_clock::now(), nullptr);
        waiter.join();
        other_waiter.join();
        context.waitDone();
        const bool rejected = status == grpc::StatusCode::OK;
        EXPECT_TRUE(context.done());
        EXPECT_FALSE(context.success());
        EXPECT_EQ(context.errorInfo().code(), rejected ? ErrorCode::KV_CACHE_REUSE_ERROR : ErrorCode::NONE_ERROR);
        EXPECT_EQ(callbacks.load(), 1);
        EXPECT_EQ(invalidations.load(), rejected ? 1 : 0);
        EXPECT_EQ(events, rejected ? (std::vector<int>{1, 2, 3}) : (std::vector<int>{2, 3}));
        EXPECT_TRUE(retained_reference.expired());
        EXPECT_EQ(pool->requestRefBlocksNum(), 0u);
        EXPECT_EQ(pool->connectorRefBlocksNum(), 0u);
        EXPECT_EQ(pool->freeBlocksNum(), 2u);
    }
}
}  // namespace rtp_llm::test
