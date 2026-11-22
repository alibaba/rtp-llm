#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionForget.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

#include <array>
#include <atomic>
#include <stdexcept>

#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/cache/block_tree_cache/test/CompletionRpcTestServer.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

using Outcome = TransferTaskRegistry::ForgetResult;
using Trace = TransferCompletionKey;
using block_tree_cache_test::CompletionRpcTestServer;

TransferCompletionForgetStatusPB wireStatus(Outcome outcome) {
    switch (outcome) {
        case Outcome::DONE: return TRANSFER_FORGET_DONE;
        case Outcome::RUNNING: return TRANSFER_FORGET_RUNNING;
        case Outcome::INVALID: return TRANSFER_FORGET_INVALID;
        default: return TRANSFER_FORGET_FAILED;
    }
}

void answer(size_t rank, const FunctionRequestPB& request, FunctionResponsePB& response,
            const std::function<Outcome(const TransferCompletionKeyPB&)>& forget = {}) {
    auto* reply = response.mutable_completion_forget();
    reply->set_rank(rank);
    if (!request.has_completion_forget())
        return;
    for (const auto& key : request.completion_forget().keys()) {
        auto* entry = reply->add_entries();
        *entry->mutable_key() = key;
        entry->set_status(wireStatus(forget ? forget(key) : Outcome::DONE));
    }
}

TEST(TransferCompletionForgetTest, RemoteBatchRequiresAllOriginalRanksButRunningDoesNotBlockOtherKeys) {
    auto workers = std::make_shared<std::array<TransferTaskRegistry, 2>>();
    const Trace finished{uint64_t{71}}, pending{std::string("72")};
    for (auto& worker : *workers) {
        ASSERT_TRUE(worker.begin(finished.trace_id));
        ASSERT_TRUE(worker.finish(finished.trace_id, true));
        ASSERT_TRUE(worker.begin(pending.trace_id));
    }
    ASSERT_TRUE((*workers)[0].finish(pending.trace_id, true));
    std::atomic<bool> wrong_key{false};
    auto handler = [&wrong_key, workers](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response, [&](const TransferCompletionKeyPB& key) {
            return (*workers)[rank].forget(key.trace_id());
        });
        if (rank == 1 && wrong_key.load())
            response.mutable_completion_forget()->mutable_entries(0)->mutable_key()->set_trace_id("71");
        return grpc::Status::OK;
    };
    CompletionRpcTestServer first(0, handler), second(1, handler);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    const std::vector<std::string> addresses{first.address(), second.address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    TransferCompletionForget forget(manager);
    const TransferCompletionTarget routing{true, addresses.size()};
    TransferCompletionCleanup cleanup;
    cleanup.enqueue(cleanup.prepare(routing, finished));
    cleanup.enqueue(cleanup.prepare(routing, pending));
    auto drain = [&](const std::vector<Trace>& expected, bool lose_reply = false) {
        return cleanup.drain([&](const auto& peers, const auto& batch, int timeout) {
            EXPECT_EQ(batch, expected);
            auto result = forget.forget(peers, batch, timeout);
            if (lose_reply)
                throw std::runtime_error("forget response lost");
            return result;
        }, 2000);
    };
    EXPECT_EQ(drain({finished, pending}).forgotten, 1u);
    EXPECT_EQ(cleanup.size(), 1u);
    EXPECT_EQ((*workers)[0].size(), 0u);
    EXPECT_EQ((*workers)[1].size(), 1u);
    ASSERT_TRUE((*workers)[1].finish(pending.trace_id, true));
    wrong_key = true;
    EXPECT_EQ(drain({pending}).pending, 1u);
    EXPECT_EQ((*workers)[1].size(), 0u);
    wrong_key = false;
    EXPECT_EQ(drain({pending}, true).pending, 1u);
    EXPECT_EQ(drain({pending}).forgotten, 1u);
    EXPECT_EQ(cleanup.size(), 0u);
}

}
}
