#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {
namespace {
using State = TransferTaskRegistry::State;

TransferTask task(size_t count = 1) {
    return TransferTask(std::vector<TransferDescriptor>(count, TransferDescriptor::hostToDisk(0, 1, 1)),
                        std::chrono::seconds(30));
}

TransferResult result(bool safe, TransferStatus status = TransferStatus::OK) {
    TransferResult value(status);
    value.buffer_safe = safe;
    return value;
}

State state(const BlockTransferDispatcher& dispatcher, uint64_t trace) {
    const auto& registry = dispatcher.taskRegistry();
    return registry.query(registry.epoch(), std::vector<uint64_t>{trace}).front();
}

enum class Observation { NORMAL, THROW_BEFORE_SUBSCRIBE, THROW_AFTER_SUBSCRIBE, THROW_ON_EXECUTE, NULL_CONTEXT };

class ProducerObservation final: public TransferAsyncContext {
public:
    ProducerObservation(std::shared_ptr<TransferBatchAsyncContext> producer, Observation mode):
        producer_(std::move(producer)), mode_(mode) {}
    void waitDone() override { producer_->waitDone(); }
    void onDone(DoneCallback callback) override {
        if (mode_ == Observation::THROW_BEFORE_SUBSCRIBE)
            throw std::runtime_error("observer subscription failed");
        producer_->onDone(std::move(callback));
        if (mode_ == Observation::THROW_AFTER_SUBSCRIBE)
            throw std::runtime_error("observer failed after subscription");
    }
    bool done() const override { return producer_->done(); }
    bool success() const override { return producer_->success(); }
    TransferResult result() const override { return producer_->result(); }
private:
    std::shared_ptr<TransferBatchAsyncContext> producer_;
    Observation mode_;
};

class ControlledProducer final: public PerRankBlockTransferEngine {
public:
    explicit ControlledProducer(Observation mode = Observation::NORMAL):
        PerRankBlockTransferEngine({}, false, {}, 128, 1, 1, 8, nullptr, 0, 4, true), mode(mode) {}
    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask batch) override {
        ++submissions;
        traces.push_back(batch.traceId());
        auto producer = std::make_shared<TransferBatchAsyncContext>(
            batch.descriptors().size(), nullptr, batch.completionProof());
        producers.push_back(producer);
        if (after_start)
            after_start(batch, producer);
        if (mode == Observation::THROW_ON_EXECUTE)
            throw std::runtime_error("execute threw after starting I/O");
        if (mode == Observation::NULL_CONTEXT)
            return nullptr;
        return std::make_shared<ProducerObservation>(producer, mode);
    }
    Observation mode;
    std::atomic<size_t> submissions{0};
    std::vector<uint64_t> traces;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> producers;
    std::function<void(const TransferTask&, const std::shared_ptr<TransferBatchAsyncContext>&)> after_start;
};

TEST(WorkerTransferRegistryTest, RegistersOneTraceAndWaitsForEveryPhysicalBatch) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    auto execution = task(3);
    const auto trace = execution.traceId();
    auto context = dispatcher.executePerRank(std::move(execution));
    ASSERT_EQ(engine->submissions, 3u);
    EXPECT_EQ(engine->traces, std::vector<uint64_t>(3, trace));
    EXPECT_EQ(dispatcher.taskRegistry().size(), 1u);
    EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    engine->producers[1]->complete(result(true, TransferStatus::DISK_IO_ERROR));
    engine->producers[0]->complete(result(true));
    EXPECT_FALSE(context->done());
    EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    context->onDone([&](TransferResult final) {
        EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
        EXPECT_FALSE(final.ok());
        EXPECT_TRUE(final.buffer_safe);
    });
    engine->producers[2]->complete(result(true));
    EXPECT_TRUE(context->done());
    EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
}

TEST(WorkerTransferRegistryTest, SynchronousBatchesCannotFinishBeforeSubmissionCloses) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    auto execution = task(3);
    const auto trace = execution.traceId();
    engine->after_start = [&](const TransferTask&, const std::shared_ptr<TransferBatchAsyncContext>& producer) {
        EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
        producer->complete(result(true));
        EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    };
    auto context = dispatcher.executePerRank(std::move(execution));
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
}

TEST(WorkerTransferRegistryTest, LateProofSurvivesFailedBusinessObservation) {
    for (auto mode : {Observation::THROW_BEFORE_SUBSCRIBE, Observation::THROW_AFTER_SUBSCRIBE,
                      Observation::THROW_ON_EXECUTE, Observation::NULL_CONTEXT}) {
        SCOPED_TRACE(static_cast<int>(mode));
        auto engine = std::make_shared<ControlledProducer>(mode);
        BlockTransferDispatcher dispatcher(engine);
        auto execution = task();
        const auto trace = execution.traceId();
        auto context = dispatcher.executePerRank(std::move(execution));
        ASSERT_TRUE(context->done());
        EXPECT_FALSE(context->success());
        const auto original_error = context->result().status.code();
        EXPECT_FALSE(context->result().buffer_safe);
        EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
        size_t callbacks = 0;
        context->onDone([&](TransferResult) { ++callbacks; });
        engine->producers.front()->complete(result(true));
        EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
        EXPECT_FALSE(context->result().buffer_safe);
        EXPECT_EQ(context->result().status.code(), original_error);
        EXPECT_EQ(callbacks, 1u);
    }
}

TEST(WorkerTransferRegistryTest, MissingProofStaysRunningUntilExplicitLateConfirmation) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    auto execution = task(2);
    const auto trace = execution.traceId();
    auto context = dispatcher.executePerRank(std::move(execution));
    engine->producers[0]->complete(result(false, TransferStatus::DEVICE_IO_ERROR));
    engine->producers[1]->complete(result(true));
    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->result().buffer_safe);
    EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    dispatcher.shutdown();
    EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    engine->producers[0]->complete(result(true));
    EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
    EXPECT_FALSE(context->result().buffer_safe);
}

TEST(WorkerTransferRegistryTest, DuplicateEvenWhenExpiredFailsSafelyWithoutChangingOriginalProof) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    auto execution = task();
    const auto trace = execution.traceId();
    auto original = dispatcher.executePerRank(execution);
    execution.deadline_ = TransferTask::Clock::now() - std::chrono::seconds(1);
    auto duplicate = dispatcher.executePerRank(execution);
    EXPECT_TRUE(duplicate->done());
    EXPECT_FALSE(duplicate->success());
    EXPECT_EQ(duplicate->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(duplicate->result().buffer_safe);
    EXPECT_EQ(duplicate->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(duplicate->result().block_errors, (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED}));
    EXPECT_EQ(engine->submissions, 1u);
    EXPECT_FALSE(original->done());
    EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    engine->producers.front()->complete(result(true));
    EXPECT_TRUE(original->success());
    duplicate = dispatcher.executePerRank(execution);
    EXPECT_FALSE(duplicate->success());
    EXPECT_EQ(duplicate->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(duplicate->result().buffer_safe);
    EXPECT_EQ(duplicate->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(duplicate->result().block_errors, (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED}));
    EXPECT_EQ(engine->submissions, 1u);
    EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
    EXPECT_EQ(dispatcher.taskRegistry().size(), 1u);
}

TEST(WorkerTransferRegistryTest, ConcurrentDuplicateSubmissionsHaveOneExecutionOwner) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    const auto execution = task();
    std::vector<std::shared_ptr<TransferAsyncContext>> contexts(16);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < contexts.size(); ++i)
        threads.emplace_back([&, i] { contexts[i] = dispatcher.executePerRank(execution); });
    for (auto& thread : threads)
        thread.join();
    EXPECT_EQ(engine->submissions, 1u);
    EXPECT_EQ(dispatcher.taskRegistry().size(), 1u);
    EXPECT_EQ(state(dispatcher, execution.traceId()), State::RUNNING);
    engine->producers.front()->complete(result(true));
    size_t successes = 0;
    for (const auto& context : contexts) {
        ASSERT_TRUE(context->done());
        successes += context->success();
        if (!context->success()) {
            EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_TRUE(context->result().buffer_safe);
            EXPECT_EQ(context->result().copy_error, CacheCopyError::NOT_EXECUTED);
        }
    }
    EXPECT_EQ(successes, 1u);
    EXPECT_EQ(state(dispatcher, execution.traceId()), State::BUFFER_SAFE);
}

TEST(WorkerTransferRegistryTest, AdmissionRejectionsRetainSafeRecordsAndWarningsDoNotLimitAdmission) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    dispatcher.next_registry_warning_ = 2;
    auto empty = task(0);
    const auto empty_trace = empty.traceId();
    EXPECT_FALSE(dispatcher.executePerRank(empty)->success());
    EXPECT_EQ(state(dispatcher, empty_trace), State::BUFFER_SAFE);
    for (size_t i = 0; i < 4; ++i) {
        auto expired = task();
        expired.deadline_ = TransferTask::Clock::now() - std::chrono::seconds(1);
        const auto trace = expired.traceId();
        const auto context = dispatcher.executePerRank(std::move(expired));
        EXPECT_FALSE(context->success());
        EXPECT_TRUE(context->result().buffer_safe);
        EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
    }
    EXPECT_EQ(engine->submissions, 0u);
    EXPECT_EQ(dispatcher.taskRegistry().size(), 5u);
    EXPECT_EQ(dispatcher.next_registry_warning_, 8u);
    EXPECT_EQ(state(dispatcher, empty_trace), State::BUFFER_SAFE);
}

TEST(WorkerTransferRegistryTest, LocalCoordinatorUsesActualChildTrace) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    auto parent = task();
    const auto parent_trace = parent.traceId();
    size_t calls = 0;
    dispatcher.runTransfer(std::move(parent), [&](TransferResult value) {
        ++calls;
        EXPECT_TRUE(value.ok());
    });
    ASSERT_EQ(engine->traces.size(), 1u);
    const auto child_trace = engine->traces.front();
    EXPECT_NE(child_trace, parent_trace);
    EXPECT_EQ(state(dispatcher, parent_trace), State::UNKNOWN);
    EXPECT_EQ(state(dispatcher, child_trace), State::RUNNING);
    engine->producers.front()->complete(result(true));
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(state(dispatcher, child_trace), State::BUFFER_SAFE);
}

TEST(WorkerTransferRegistryTest, SameTraceOnDifferentWorkersHasIndependentEvidence) {
    auto first = std::make_shared<ControlledProducer>();
    auto second = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher worker0(first), worker1(second);
    const auto execution = task();
    auto context0 = worker0.executePerRank(execution);
    auto context1 = worker1.executePerRank(execution);
    first->producers.front()->complete(result(true));
    EXPECT_EQ(state(worker0, execution.traceId()), State::BUFFER_SAFE);
    EXPECT_EQ(state(worker1, execution.traceId()), State::RUNNING);
    second->producers.front()->complete(result(true));
    EXPECT_EQ(state(worker1, execution.traceId()), State::BUFFER_SAFE);
    EXPECT_TRUE(context0->success());
    EXPECT_TRUE(context1->success());
}

}  // namespace
}  // namespace rtp_llm
