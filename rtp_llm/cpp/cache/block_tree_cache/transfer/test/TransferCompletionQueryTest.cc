#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"

#include <atomic>

#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/cache/block_tree_cache/test/CompletionRpcTestServer.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

using State = TransferTaskRegistry::State;
using block_tree_cache_test::CompletionRpcTestServer;

void answer(size_t rank, const FunctionRequestPB& request, FunctionResponsePB& response,
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

enum class Reply { RUNNING, UNKNOWN, WRONG_RANK, WRONG_KEY, WRONG_REGISTRY, INVALID_STATE, TRANSPORT_ERROR, DUPLICATE_RANK, SAFE, FENCED };

TEST(TransferCompletionQueryTest, RecoveryRequiresEvidenceForTheOriginalTraceAndEveryParticipant) {
    std::atomic<Reply> reply{Reply::RUNNING};
    auto handler = [&](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        if (rank == 0)
            return grpc::Status::OK;
        auto* row = response.mutable_completion_query()->mutable_entries(0);
        switch (reply.load()) {
            case Reply::RUNNING: row->set_state(TRANSFER_COMPLETION_RUNNING); break;
            case Reply::UNKNOWN: row->set_state(TRANSFER_COMPLETION_UNKNOWN); break;
            case Reply::WRONG_RANK: response.mutable_completion_query()->set_rank(0); break;
            case Reply::WRONG_KEY:
                if (row->key().has_transfer_trace_id()) row->mutable_key()->set_transfer_trace_id(92);
                else row->mutable_key()->set_kvcm_trace_id("92");
                break;
            case Reply::WRONG_REGISTRY:
                if (row->key().has_transfer_trace_id()) row->mutable_key()->set_kvcm_trace_id("91");
                else row->mutable_key()->set_transfer_trace_id(91);
                break;
            case Reply::INVALID_STATE: row->set_state(static_cast<TransferCompletionStatePB>(99)); break;
            case Reply::TRANSPORT_ERROR: return grpc::Status(grpc::StatusCode::UNAVAILABLE, "query reply lost");
            case Reply::DUPLICATE_RANK:
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
    TransferCompletionQuery query(manager);
    TransferCompletionCleanup cleanup;
    for (const TransferCompletionKey& key : {TransferCompletionKey{uint64_t{91}}, TransferCompletionKey{std::string("91")}}) {
        FrozenTask::ExecutionInfo info;
        info.registry_type = key.registry_type;
        info.trace_id = key.trace_id;
        info.uses_rpc = true;
        info.rank_count = addresses.size();
        auto task = std::make_shared<FrozenTask>(FrozenTask::prepare(info),
            TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "lost read reply"), CacheCopyError::RPC_FAILED));
        task->freeze();
        FrozenTaskRecovery recovery(cleanup, [&] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
            [&](const auto& original, int timeout) {
                TransferCompletionIdentity identity;
                identity.trace_id = original.trace_id;
                identity.registry_type = original.registry_type;
                identity.remote = original.uses_rpc;
                identity.expected_rank_count = original.rank_count;
                auto evidence = query.query(identity, timeout);
                if (reply == Reply::DUPLICATE_RANK) {
                    EXPECT_TRUE(evidence.all_safe);
                    evidence.ranks.back().rank = evidence.ranks.front().rank;
                }
                return evidence;
            });
        reply = Reply::RUNNING;
        EXPECT_EQ(recovery.runOnce(2000).released, 0u);
        const auto running = task->recoverySnapshot().last_query;
        ASSERT_NE(running, nullptr);
        ASSERT_EQ(running->ranks.size(), 2u);
        EXPECT_EQ(running->ranks[1].state, State::RUNNING);
        for (const auto pending : {Reply::UNKNOWN, Reply::WRONG_RANK, Reply::WRONG_KEY, Reply::WRONG_REGISTRY,
                                   Reply::INVALID_STATE, Reply::TRANSPORT_ERROR, Reply::DUPLICATE_RANK}) {
            SCOPED_TRACE(static_cast<int>(pending));
            reply = pending;
            EXPECT_EQ(recovery.runOnce(2000).released, 0u);
            EXPECT_EQ(task->snapshot().state, FrozenTask::State::FROZEN);
        }
        reply = key.registry_type == RegistryType::KVCM_READ ? Reply::FENCED : Reply::SAFE;
        EXPECT_EQ(recovery.runOnce(2000).released, 1u);
        EXPECT_EQ(task->snapshot().state, FrozenTask::State::RELEASED);
        EXPECT_EQ(running->ranks[1].state, State::RUNNING);
    }
}

}
}
