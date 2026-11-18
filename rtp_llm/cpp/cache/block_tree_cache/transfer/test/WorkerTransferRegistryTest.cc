#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <memory>
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

class ControlledProducer final: public PerRankBlockTransferEngine {
public:
    ControlledProducer(): PerRankBlockTransferEngine({}, false, {}, 128, 1, 1, 8, nullptr, 0, 4, true) {}
    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask batch) override {
        auto context = std::make_shared<TransferBatchAsyncContext>(batch.descriptors().size(), nullptr, batch.completionProof());
        proofs.push_back(batch.completionProof());
        contexts.push_back(context);
        on_submit(batch.traceId());
        return context;
    }
    std::vector<std::shared_ptr<TransferBufferSafety>> proofs;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts;
    std::function<void(uint64_t)> on_submit;
};

TEST(WorkerTransferRegistryTest, OneTraceBecomesSafeOnlyAfterSubmissionAndEveryBatchProof) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine);
    auto execution = task(3);
    const auto trace = execution.traceId();
    engine->on_submit = [&](uint64_t id) {
        EXPECT_EQ(id, trace);
        if (engine->contexts.size() == 1) {
            engine->contexts.front()->complete(result(true));
        }
        EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    };
    auto late = task();
    ASSERT_EQ(dispatcher.queryLocalCompletion(late.traceId()), State::REJECTED);
    ASSERT_EQ(dispatcher.forgetLocalCompletion(late.traceId()), TransferTaskRegistry::ForgetResult::DONE);
    auto rejected = dispatcher.executePerRank(std::move(late));
    EXPECT_EQ(rejected->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_TRUE(engine->contexts.empty());
    auto original = dispatcher.executePerRank(execution);
    ASSERT_EQ(engine->contexts.size(), 3u);
    execution.deadline_ = TransferTask::Clock::now() - std::chrono::seconds(1);
    auto check_duplicate = [&] {
        auto duplicate = dispatcher.executePerRank(execution);
        EXPECT_EQ(duplicate->errorInfo().code(), ErrorCode::INVALID_PARAMS);
        EXPECT_EQ(duplicate->result().copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_EQ(engine->contexts.size(), 3u);
    };
    check_duplicate();
    engine->contexts[1]->complete(result(true));
    engine->contexts[2]->complete(result(false, TransferStatus::DISK_IO_ERROR));
    ASSERT_TRUE(original->done());
    EXPECT_FALSE(original->success());
    engine->proofs[0]->report(true);
    EXPECT_EQ(state(dispatcher, trace), State::RUNNING);
    engine->proofs[2]->report(true);
    check_duplicate();
    EXPECT_EQ(state(dispatcher, trace), State::BUFFER_SAFE);
}
}
}
