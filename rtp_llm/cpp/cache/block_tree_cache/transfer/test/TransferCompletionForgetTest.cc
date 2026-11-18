#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionForget.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

#include <algorithm>
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

TransferCompletionTarget target(const std::vector<std::string>& addresses, bool remote = true) {
    TransferCompletionTarget value;
    value.remote = remote;
    for (size_t rank = 0; rank < addresses.size(); ++rank)
        value.peers.push_back({rank, addresses[rank]});
    return value;
}

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
    struct Worker {
        TransferTaskRegistry ordinary;
        TransferTaskRegistry kvcm;
    };
    auto workers = std::make_shared<std::array<Worker, 2>>();
    for (size_t rank = 0; rank < 2; ++rank) {
        ASSERT_TRUE((*workers)[rank].ordinary.begin(uint64_t{71}));
        ASSERT_TRUE((*workers)[rank].ordinary.finish(uint64_t{71}, true));
        ASSERT_TRUE((*workers)[rank].kvcm.begin("71"));
    }
    ASSERT_TRUE((*workers)[0].kvcm.finish("71", true));
    std::atomic<bool> wrong_key{false};
    auto handler = [&wrong_key, workers](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response, [&](const TransferCompletionKeyPB& key) {
            return key.trace_case() == TransferCompletionKeyPB::kTransferTraceId ?
                       (*workers)[rank].ordinary.forget(key.transfer_trace_id()) :
                       (*workers)[rank].kvcm.forget(key.kvcm_trace_id());
        });
        if (rank == 1 && wrong_key.load())
            response.mutable_completion_forget()->mutable_entries(0)->mutable_key()->set_transfer_trace_id(71);
        return grpc::Status::OK;
    };
    CompletionRpcTestServer first(0, handler), second(1, handler);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    const std::vector<std::string> addresses{first.address(), second.address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    TransferCompletionForget forget(manager);
    auto routing = target(addresses);
    std::reverse(routing.peers.begin(), routing.peers.end());
    TransferCompletionCleanup cleanup;
    const Trace ordinary{uint64_t{71}}, kvcm{std::string("71")};
    cleanup.enqueue(cleanup.prepare(routing, ordinary));
    cleanup.enqueue(cleanup.prepare(routing, kvcm));
    auto drain = [&](const std::vector<Trace>& expected, bool lose_reply = false) {
        return cleanup.drain([&](const auto& peers, const auto& batch, int timeout) {
            EXPECT_EQ(batch, expected);
            auto result = forget.forget(peers, batch, timeout);
            if (lose_reply)
                throw std::runtime_error("forget response lost");
            return result;
        }, 2000);
    };
    EXPECT_EQ(drain({ordinary, kvcm}).forgotten, 1u);
    EXPECT_EQ(cleanup.size(), 1u);
    EXPECT_EQ((*workers)[0].ordinary.size(), 0u);
    EXPECT_EQ((*workers)[1].ordinary.size(), 0u);
    EXPECT_EQ((*workers)[1].kvcm.size(), 1u);
    ASSERT_TRUE((*workers)[1].kvcm.finish("71", true));
    wrong_key = true;
    EXPECT_EQ(drain({kvcm}).pending, 1u);
    EXPECT_EQ((*workers)[1].kvcm.size(), 0u);
    wrong_key = false;
    EXPECT_EQ(drain({kvcm}, true).pending, 1u);
    EXPECT_EQ(drain({kvcm}).forgotten, 1u);
    EXPECT_EQ(cleanup.size(), 0u);
}

}
}
