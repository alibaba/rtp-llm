#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CompletionRpcTestServer.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

#include <array>
#include <future>
#include <stdexcept>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
using block_tree_cache_test::CompletionRpcTestServer;

class FrozenTestPool final: public IBlockPool {
public:
    explicit FrozenTestPool(std::shared_ptr<const BlockPoolConfigBase> config): IBlockPool(std::move(config)) {
        markInitialized();
    }
    size_t blockSizeBytes() const override {
        return 0;
    }
};

std::shared_ptr<FrozenTestPool> makePool(std::string name) {
    auto config                            = std::make_shared<BlockPoolConfigBase>();
    config->pool_name                      = std::move(name);
    config->pool_type                      = BlockPoolType::HOST;
    config->physical_block_count           = 16;
    return std::make_shared<FrozenTestPool>(std::move(config));
}

BlockIdxType heldBlock(const std::shared_ptr<FrozenTestPool>& pool) {
    const auto block = pool->malloc();
    EXPECT_TRUE(block.has_value());
    if (!block)
        return NULL_BLOCK_IDX;
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    return *block;
}

TEST(BlockTreeReclaimerTest, OnlyTheLastPendingWorkerReleasesTheTrace) {
    BlockTreeReclaimer owner;
    auto pool = makePool("pending-workers");
    const auto block = heldBlock(pool);
    const TransferCompletionKey key{uint64_t{41}};
    ASSERT_TRUE(owner.registerTask(key, {true, 3}, {{pool, block}}));
    owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(key, {1, 2, 1});
    EXPECT_FALSE(owner.markWorkerSafe(key, 0));
    EXPECT_FALSE(owner.markWorkerSafe(key, 2));
    EXPECT_FALSE(owner.markWorkerSafe(key, 2));
    EXPECT_TRUE(pool->isFrozen(block));
    EXPECT_TRUE(owner.forget_queue_.empty());
    EXPECT_TRUE(owner.markWorkerSafe(key, 1));
    EXPECT_FALSE(owner.markWorkerSafe(key, 1));
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_EQ(owner.forget_queue_.size(), 1u);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(BlockTreeReclaimerTest, UnknownWriteSourcesSurviveReadRecoveryAndForcedExpiry) {
    BlockTreeReclaimer owner;
    auto pool = makePool("write-source");
    const auto block = heldBlock(pool);
    const TransferCompletionKey read_trace{"test-read"};
    const TransferCompletionKey write_trace{"test-write"};
    ASSERT_TRUE(owner.registerTask(read_trace, {false, 1}, {{pool, block}}));
    owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(read_trace, {0});
    ASSERT_TRUE(owner.freezeWriteSources(write_trace, {{pool, block}, {pool, block}}));
    EXPECT_FALSE(owner.freezeWriteSources(write_trace, {{pool, block}}));
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->treeRefCount(block), 2u);

    EXPECT_TRUE(owner.markWorkerSafe(read_trace, 0));
    EXPECT_FALSE(owner.markWorkerSafe(write_trace, 0));
    EXPECT_FALSE(owner.forceRelease(write_trace));
    owner.frozen_.at(write_trace.trace_id).frozen_at -= std::chrono::minutes(2);
    owner.releaseExpired();
    owner.queryFrozen();
    ASSERT_TRUE(owner.frozen_.count(write_trace.trace_id));
    EXPECT_TRUE(pool->isFrozen(block));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_EQ(owner.forget_queue_.size(), 1u);
}

TEST(BlockTreeReclaimerTest, ReleasesEachOwnedTargetOnceBeforeSubmittingForget) {
    BlockTreeReclaimer owner;
    auto primary = makePool("primary");
    auto secondary = makePool("secondary");
    const auto shared = heldBlock(primary);
    const auto unique = heldBlock(primary);
    const auto other = heldBlock(secondary);
    ASSERT_EQ(shared, other);
    size_t notifications = 0;
    primary->setCapacityChangeCallback([&] {
        ++notifications;
        EXPECT_EQ(owner.forget_queue_.size(), 0u);
        EXPECT_FALSE(primary->isAllocated(unique));
        EXPECT_TRUE(primary->isFrozen(shared));
        throw std::runtime_error("capacity notification failed");
    });
    secondary->setCapacityChangeCallback([&] {
        ++notifications;
        EXPECT_EQ(owner.forget_queue_.size(), 0u);
        EXPECT_FALSE(secondary->isAllocated(other));
    });
    ASSERT_TRUE(owner.registerTask(36, {false, 1}, {{primary, shared}, {primary, shared}, {primary, unique}, {secondary, other}}));
    owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(36, {0});
    primary->setCapacityChangeCallback({});
    ASSERT_TRUE(owner.registerTask(37, {false, 1}, {{primary, shared}}));
    owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(37, {0});
    primary->decTreeRef({unique, shared}, BlockTreeRefType::CACHE);
    secondary->decTreeRef(other, BlockTreeRefType::CACHE);
    EXPECT_TRUE(owner.markWorkerSafe(36, 0));
    EXPECT_FALSE(owner.markWorkerSafe(36, 0));
    EXPECT_EQ(notifications, 2u);
    EXPECT_EQ(owner.forget_queue_.size(), 1u);
    EXPECT_EQ(primary->treeRefCount(shared), 1u);
    EXPECT_TRUE(owner.markWorkerSafe(37, 0));
    EXPECT_FALSE(primary->isAllocated(shared));
}

void answerQuery(size_t rank, const FunctionRequestPB& request, FunctionResponsePB& response,
            TransferCompletionStatePB state = TRANSFER_COMPLETION_BUFFER_SAFE) {
    auto* query = response.mutable_completion_query();
    query->set_rank(rank);
    if (!request.has_completion_query())
        return;
    for (const auto& key : request.completion_query().keys()) {
        auto* entry = query->add_entries();
        *entry->mutable_key() = key;
        entry->set_state(state);
    }
}

enum class Reply { RUNNING, UNKNOWN, WRONG_RANK, WRONG_KEY, INVALID_STATE, TRANSPORT_ERROR, DUPLICATE_RANK, SAFE, FENCED };

TEST(BlockTreeReclaimerTest, RecoveryRequiresEvidenceForTheOriginalTraceAndEveryParticipant) {
    std::atomic<Reply> reply{Reply::RUNNING};
    std::atomic<size_t> batched_calls{0};
    auto handler = [&](size_t rank, const auto& request, auto& response) {
        answerQuery(rank, request, response);
        const auto& keys = request.completion_query().keys();
        if (keys.size() == 2) {
            ++batched_calls;
            if (rank == 0)
                response.mutable_completion_query()->mutable_entries()->RemoveLast();
        } else if (keys.size() == 1 && keys.Get(0).trace_id() == "93") {
            EXPECT_EQ(rank, 1u);
        } else if (keys.size() == 1 && keys.Get(0).trace_id() == "read-94") {
            EXPECT_EQ(rank, 0u);
        }
        if (rank == 0)
            return grpc::Status::OK;
        auto* row = response.mutable_completion_query()->mutable_entries(0);
        switch (reply.load()) {
            case Reply::RUNNING: row->set_state(TRANSFER_COMPLETION_RUNNING); break;
            case Reply::UNKNOWN: row->set_state(TRANSFER_COMPLETION_UNKNOWN); break;
            case Reply::WRONG_RANK: response.mutable_completion_query()->set_rank(0); break;
            case Reply::WRONG_KEY: row->mutable_key()->set_trace_id("wrong-trace"); break;
            case Reply::INVALID_STATE: row->set_state(static_cast<TransferCompletionStatePB>(99)); break;
            case Reply::TRANSPORT_ERROR: return grpc::Status(grpc::StatusCode::UNAVAILABLE, "query reply lost");
            case Reply::DUPLICATE_RANK:
                response.mutable_completion_query()->add_entries()->CopyFrom(response.completion_query().entries(0));
                break;
            case Reply::SAFE: break;
            case Reply::FENCED:
                row->set_state(TRANSFER_COMPLETION_UNKNOWN);
                row->set_admission_closed(true);
                break;
        }
        return grpc::Status::OK;
    };
    CompletionRpcTestServer first(0, handler), second(1, handler);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    const std::vector<std::string> addresses{first.address(), second.address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    BlockTreeReclaimer owner;
    owner.configure(0, 0, 2000, 0, manager);
    for (const TransferCompletionKey& key : {TransferCompletionKey{uint64_t{91}}, TransferCompletionKey{std::string("read-92")}}) {
        ASSERT_TRUE(owner.registerTask(key, {true, addresses.size()}, {}));
        owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(key, {0, 1});
        reply = Reply::RUNNING;
        owner.queryFrozen();
        EXPECT_EQ(owner.frozen_.at(key.trace_id).pending, (std::vector<size_t>{1}));
        for (auto invalid : {Reply::UNKNOWN, Reply::WRONG_RANK, Reply::WRONG_KEY, Reply::INVALID_STATE,
                             Reply::TRANSPORT_ERROR, Reply::DUPLICATE_RANK}) {
            reply = invalid;
            owner.queryFrozen();
            EXPECT_EQ(owner.frozen_.count(key.trace_id), 1u);
        }
        reply = key.trace_id == "read-92" ? Reply::FENCED : Reply::SAFE;
        owner.queryFrozen();
        EXPECT_TRUE(owner.frozen_.empty());
    }
    for (const TransferCompletionKey& key : {TransferCompletionKey{uint64_t{93}}, TransferCompletionKey{std::string("read-94")}}) {
        ASSERT_TRUE(owner.registerTask(key, {true, addresses.size()}, {}));
        owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(key, {0, 1});
    }
    reply = Reply::RUNNING;
    owner.queryFrozen();
    EXPECT_EQ(batched_calls.load(), 2u);
    EXPECT_EQ(owner.frozen_.at("93").pending, (std::vector<size_t>{1}));
    EXPECT_EQ(owner.frozen_.at("read-94").pending, (std::vector<size_t>{0}));
    reply = Reply::SAFE;
    owner.queryFrozen();
    EXPECT_TRUE(owner.frozen_.empty());
    EXPECT_EQ(batched_calls.load(), 2u);
}

TEST(BlockTreeReclaimerTest, AcceptsLegacyTraceAndRejectsConflictingFields) {
    BlockTreeReclaimer owner;
    for (bool numeric : {false, true}) {
        const std::string trace = numeric ? "73" : "read-74";
        ASSERT_TRUE(owner.registerOnWorkerTaskWithTraceID(trace));
        owner.ioFinishedOnWorkerWithTraceID(trace);
        FunctionRequestPB request;
        auto* key = request.mutable_completion_query()->add_keys();
        if (numeric)
            key->set_transfer_trace_id(73);
        else
            key->set_kvcm_trace_id(trace);
        FunctionResponsePB response;
        for (auto id : {std::string{}, trace, std::string("wrong")}) {
            key->set_trace_id(id);
            ASSERT_TRUE(BlockTreeReclaimer::executeFunction(&owner, 0, request, response));
            EXPECT_EQ(response.completion_query().entries(0).state(),
                      id == "wrong" ? TRANSFER_COMPLETION_UNKNOWN : TRANSFER_COMPLETION_BUFFER_SAFE);
        }
    }
}

TEST(BlockTreeReclaimerTest, RemoteForgetRetriesOnlyUnconfirmedTraces) {
    std::array<BlockTreeReclaimer, 2> workers;
    std::atomic<bool> wrong_key{false}, lose_reply{false};
    auto handler = [&](size_t rank, const auto& request, auto& response) {
        BlockTreeReclaimer::executeFunction(&workers[rank], rank, request, response);
        if (wrong_key && rank == 1)
            response.mutable_completion_forget()->mutable_entries(0)->mutable_key()->set_trace_id("wrong");
        if (lose_reply && rank == 1)
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "reply lost");
        return grpc::Status::OK;
    };
    CompletionRpcTestServer first(0, handler), second(1, handler);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{first.address(), second.address()});
    ASSERT_TRUE(manager->init());
    BlockTreeReclaimer owner;
    owner.configure(0, 0, 2000, 0, manager);
    for (const auto& trace : {std::string("71"), std::string("72")}) {
        for (auto& worker : workers)
            ASSERT_TRUE(worker.registerOnWorkerTaskWithTraceID(trace));
        ASSERT_TRUE(owner.registerTask(trace, {true, workers.size()}, {}));
        owner.unregisterTask(trace);
    }
    workers[0].ioFinishedOnWorkerWithTraceID("71");
    workers[1].ioFinishedOnWorkerWithTraceID("71");
    workers[0].ioFinishedOnWorkerWithTraceID("72");
    owner.drainForget();
    ASSERT_EQ(owner.forget_queue_.size(), 1u);
    EXPECT_EQ(owner.forget_queue_.front().trace.trace_id, "72");
    workers[1].ioFinishedOnWorkerWithTraceID("72");
    wrong_key = true;
    owner.drainForget();
    EXPECT_EQ(owner.forget_queue_.size(), 1u);
    wrong_key = false;
    lose_reply = true;
    owner.drainForget();
    EXPECT_EQ(owner.forget_queue_.size(), 1u);
    lose_reply = false;
    owner.drainForget();
    EXPECT_TRUE(owner.forget_queue_.empty());
}

TEST(BlockTreeReclaimerTest, StopDuringQueryDiscardsReleaseAndWaitsForTheCallback) {
    BlockTreeReclaimer owner;
    std::promise<void> entered, release;
    auto wait = release.get_future().share();
    ASSERT_TRUE(owner.registerOnWorkerTaskWithTraceID("51", [&](auto) {
        entered.set_value();
        wait.wait_for(std::chrono::seconds(5));
        return true;
    }));
    ASSERT_TRUE(owner.registerTask(51, {false, 1}, {}));
    owner.freezeTargetBlocksWithTraceIDAndStillIOWorkerID(51, {0});
    owner.configure(0, 10, 2000, 0, nullptr);
    ASSERT_TRUE(owner.start());
    ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto stopped = std::async(std::launch::async, [&] { owner.stop(); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!owner.stopped_ && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_TRUE(owner.stopped_);
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    release.set_value();
    stopped.get();
    EXPECT_EQ(owner.frozen_.count("51"), 1u);
}

using Registry     = BlockTreeReclaimer;
using State        = Registry::State;
using ForgetResult = Registry::ForgetResult;

TEST(BlockTreeReclaimerTest, OptionalCallbacksCanPublishCompletionAndRetryForget) {
    Registry worker;
    int queries = 0;
    int forgets = 0;
    ASSERT_TRUE(worker.registerOnWorkerTaskWithTraceID("read", [&](auto) {
        ++queries;
        return true;
    }, [&](bool) {
        if (++forgets == 1)
            throw std::runtime_error("retry forget");
    }));
    ASSERT_TRUE(worker.registerOnWorkerTaskWithTraceID("local"));
    EXPECT_FALSE(worker.registerOnWorkerTaskWithTraceID("read"));
    EXPECT_EQ(worker.forgetTaskInfoOnWorker({"read"}).front(), ForgetResult::RUNNING);
    EXPECT_EQ(worker.queryIfIoFinishedInOneWorkerWithTraceIDs({"local", "read"}),
              (std::vector<State>{State::RUNNING, State::BUFFER_SAFE}));
    EXPECT_EQ(queries, 1);
    EXPECT_EQ(worker.forgetTaskInfoOnWorker({"read", "read", "read"}),
              (std::vector<ForgetResult>{ForgetResult::FAILED, ForgetResult::DONE, ForgetResult::DONE}));
    EXPECT_EQ(forgets, 2);
}

TEST(BlockTreeReclaimerTest, QueryWithoutProofPreservesConcurrentIoCompletion) {
    for (bool query_throws : {false, true}) {
        SCOPED_TRACE(query_throws);
        Registry worker;
        std::promise<void> query_started, io_finished;
        auto finished = io_finished.get_future();
        ASSERT_TRUE(worker.registerOnWorkerTaskWithTraceID("read", [&](auto) {
            query_started.set_value();
            EXPECT_EQ(finished.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            if (query_throws)
                throw std::runtime_error("query failed");
            return false;
        }));
        auto result = std::async(std::launch::async, [&] {
            return worker.queryIfIoFinishedInOneWorkerWithTraceIDs({"read"}).front();
        });
        EXPECT_EQ(query_started.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
        EXPECT_TRUE(worker.ioFinishedOnWorkerWithTraceID("read"));
        io_finished.set_value();
        EXPECT_EQ(result.get(), State::BUFFER_SAFE);
    }
}

TEST(BlockTreeReclaimerTest, ForcedForgetRemovesRunningRecordAfterActiveQuery) {
    Registry worker;
    bool discarded = false;
    ASSERT_TRUE(worker.registerOnWorkerTaskWithTraceID("read", [&](auto) {
        EXPECT_EQ(worker.forgetTaskInfoOnWorker({"read"}, true).front(), ForgetResult::RUNNING);
        return false;
    }, [&](bool forced) { discarded = forced; }));
    EXPECT_EQ(worker.forgetTaskInfoOnWorker({"read"}).front(), ForgetResult::RUNNING);
    EXPECT_EQ(worker.queryIfIoFinishedInOneWorkerWithTraceIDs({"read"}).front(), State::RUNNING);
    EXPECT_FALSE(discarded);
    EXPECT_EQ(worker.forgetTaskInfoOnWorker({"read"}, true).front(), ForgetResult::DONE);
    EXPECT_TRUE(discarded);
    EXPECT_FALSE(worker.ioFinishedOnWorkerWithTraceID("read"));
    for (bool forced : {false, true}) {
        EXPECT_EQ(worker.forgetTaskInfoOnWorker({"missing"}, forced).front(), ForgetResult::DONE);
        EXPECT_FALSE(worker.registerOnWorkerTaskWithTraceID("missing"));
    }
}

TEST(BlockTreeReclaimerTest, MissingAdmissionFenceSurvivesForgetUntilLateArrivalOrExpiry) {
    auto now = Registry::Clock::time_point{};
    Registry registry;
    registry.now_ = [&] { return now; };
    const std::vector<std::string> ids{"missing", "late"};
    EXPECT_EQ(registry.queryIfIoFinishedInOneWorkerWithTraceIDs(ids), (std::vector<State>{State::REJECTED, State::REJECTED}));
    now += std::chrono::minutes(30);
    registry.queryIfIoFinishedInOneWorkerWithTraceIDs(ids);
    EXPECT_EQ(registry.forgetTaskInfoOnWorker({"late"}).front(), ForgetResult::DONE);
    EXPECT_FALSE(registry.registerOnWorkerTaskWithTraceID("late"));
    now += std::chrono::minutes(89);
    EXPECT_EQ(registry.localState("missing"), State::REJECTED);
    EXPECT_EQ(registry.localState("late"), State::UNKNOWN);
    EXPECT_EQ(registry.forgetTaskInfoOnWorker({"missing"}).front(), ForgetResult::DONE);
    registry.cleanupExpiredMissing();
    EXPECT_EQ(registry.localState("missing"), State::REJECTED);
    now += std::chrono::minutes(1);
    registry.cleanupExpiredMissing();
    EXPECT_EQ(registry.localState("missing"), State::UNKNOWN);
    for (uint64_t trace_id = 1; trace_id <= 32; ++trace_id) {
        SCOPED_TRACE(trace_id);
        std::promise<void> start;
        const auto ready = start.get_future().share();
        auto admission = std::async(std::launch::async, [&] {
            ready.wait();
            return registry.registerOnWorkerTaskWithTraceID(std::to_string(trace_id));
        });
        auto query = std::async(std::launch::async, [&] {
            ready.wait();
            return registry.queryIfIoFinishedInOneWorkerWithTraceIDs({std::to_string(trace_id)}).front();
        });
        start.set_value();
        const bool admitted = admission.get();
        const auto observed = query.get();
        EXPECT_EQ(observed, admitted ? State::RUNNING : State::REJECTED);
        EXPECT_EQ(registry.localState(std::to_string(trace_id)), admitted ? State::RUNNING : State::UNKNOWN);
        if (admitted) {
            EXPECT_FALSE(registry.registerOnWorkerTaskWithTraceID(std::to_string(trace_id)));
        }
        EXPECT_EQ(registry.forgetTaskInfoOnWorker({std::to_string(trace_id)}).front(), admitted ? ForgetResult::RUNNING : ForgetResult::DONE);
    }
}

}
}
