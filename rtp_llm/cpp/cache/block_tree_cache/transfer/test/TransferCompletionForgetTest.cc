#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionForget.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

using Outcome = TransferTaskRegistry::ForgetResult;
using Trace = TransferExecutionTrace;
using block_tree_cache_test::BoundedThread;
using namespace std::chrono_literals;
using Handler = std::function<grpc::Status(size_t, const FunctionRequestPB&, FunctionResponsePB&)>;

class ForgetService final: public RpcService::Service {
public:
    ForgetService(size_t rank, Handler handler): rank_(rank), handler_(std::move(handler)) {}
    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        ++calls;
        return handler_(rank_, *request, *response);
    }
    std::atomic<size_t> calls{0};
private:
    size_t rank_;
    Handler handler_;
};

class ForgetServer {
public:
    ForgetServer(size_t rank, Handler handler): service_(rank, std::move(handler)) {}
    ~ForgetServer() {
        if (server_) {
            server_->Shutdown(std::chrono::system_clock::now() + 2s);
            server_->Wait();
        }
    }
    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        return server_ != nullptr && port_ > 0;
    }
    std::string address() const { return "127.0.0.1:" + std::to_string(port_); }
    size_t calls() const { return service_.calls.load(); }
private:
    ForgetService service_;
    std::unique_ptr<grpc::Server> server_;
    int port_{0};
};

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

TEST(TransferCompletionForgetTest, LocalBatchPreservesNamespacesAndRetriesRunningAndMissingIndependently) {
    TransferTaskRegistry ordinary;
    TransferTaskRegistry kvcm;
    ASSERT_TRUE(ordinary.begin(uint64_t{71}));
    ASSERT_TRUE(ordinary.begin(uint64_t{72}));
    ASSERT_TRUE(ordinary.finish(uint64_t{72}, true));
    ASSERT_TRUE(kvcm.begin("71"));
    ASSERT_TRUE(kvcm.finish("71", true));
    size_t calls = 0;
    TransferCompletionForget forget(nullptr, [&](const Trace& trace) {
        ++calls;
        if (const auto* numeric = std::get_if<uint64_t>(&trace))
            return ordinary.forget(*numeric);
        return kvcm.forget(std::get<std::string>(trace));
    });
    const std::vector<Trace> traces{uint64_t{71}, std::string("71"), uint64_t{72}, uint64_t{0}, std::string{}};
    const auto first = forget.forget(target({"local"}, false), traces, 1000);
    ASSERT_EQ(first.entries.size(), traces.size());
    const std::vector<Outcome> expected{Outcome::RUNNING, Outcome::DONE, Outcome::DONE, Outcome::INVALID, Outcome::INVALID};
    for (size_t index = 0; index < traces.size(); ++index) {
        EXPECT_EQ(first.entries[index].trace_id, traces[index]);
        ASSERT_EQ(first.entries[index].ranks.size(), 1u);
        EXPECT_EQ(first.entries[index].ranks[0].result, expected[index]);
        EXPECT_EQ(first.entries[index].all_done, expected[index] == Outcome::DONE);
        EXPECT_FALSE(first.entries[index].ranks[0].cq_received);
    }
    EXPECT_EQ(calls, 3u);  // Invalid keys never reach the registry callback.
    EXPECT_EQ(ordinary.size(), 1u);
    EXPECT_EQ(kvcm.size(), 0u);
    ASSERT_TRUE(ordinary.finish(uint64_t{71}, true));
    const auto retried = forget.forget(target({"local"}, false), {uint64_t{71}, std::string("71"), uint64_t{72}}, 1000);
    EXPECT_TRUE(retried.error.empty());
    for (const auto& entry : retried.entries)
        EXPECT_TRUE(entry.all_done);
    EXPECT_EQ(ordinary.size(), 0u);
    EXPECT_EQ(first.entries[0].ranks[0].result, Outcome::RUNNING);
}

TEST(TransferCompletionForgetTest, LocalFailuresDoNotPreventLaterKeysFromCompleting) {
    size_t calls = 0;
    TransferCompletionForget forget(nullptr, [&](const Trace& trace) -> Outcome {
        ++calls;
        switch (std::get<uint64_t>(trace)) {
            case 1: throw std::runtime_error("local erase failed");
            case 2: return static_cast<Outcome>(99);
            case 3: return Outcome::FAILED;
            default: return Outcome::DONE;
        }
    });
    const auto result = forget.forget(target({"local"}, false), {uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{4}}, 1000);
    ASSERT_EQ(result.entries.size(), 4u);
    EXPECT_EQ(calls, 4u);
    for (size_t index = 0; index < 3; ++index) {
        EXPECT_FALSE(result.entries[index].all_done);
        EXPECT_EQ(result.entries[index].ranks[0].result, Outcome::FAILED);
        EXPECT_FALSE(result.entries[index].ranks[0].error.empty());
    }
    EXPECT_EQ(result.entries[0].ranks[0].error, "local erase failed");
    EXPECT_TRUE(result.entries[3].all_done);
    EXPECT_FALSE(result.error.empty());
}

TEST(TransferCompletionForgetTest, BatchLimitTimeoutAndLocalTargetValidationPrecedeDeletion) {
    size_t calls = 0;
    TransferCompletionForget forget(nullptr, [&](const Trace&) { ++calls; return Outcome::DONE; });
    const auto local = target({"local"}, false);
    for (const int timeout : {0, -1}) {
        const auto result = forget.forget(local, {uint64_t{1}}, timeout);
        EXPECT_FALSE(result.entries[0].all_done);
        EXPECT_FALSE(result.error.empty());
    }
    EXPECT_FALSE(forget.forget(local, {}, 1000).error.empty());
    const std::vector<Trace> too_many(kTransferCompletionForgetBatchSize + 1, uint64_t{1});
    EXPECT_FALSE(forget.forget(local, too_many, 1000).error.empty());
    EXPECT_FALSE(forget.forget(target({}, false), {uint64_t{1}}, 1000).error.empty());
    EXPECT_FALSE(forget.forget(target({"local", "other"}, false), {uint64_t{1}}, 1000).error.empty());
    EXPECT_FALSE(TransferCompletionForget(nullptr).forget(local, {uint64_t{1}}, 1000).entries[0].all_done);
    EXPECT_EQ(calls, 0u);
    const auto full = forget.forget(local, std::vector<Trace>(kTransferCompletionForgetBatchSize, uint64_t{1}), 1000);
    EXPECT_EQ(calls, kTransferCompletionForgetBatchSize);
    EXPECT_EQ(full.entries.size(), kTransferCompletionForgetBatchSize);
    for (const auto& entry : full.entries)
        EXPECT_TRUE(entry.all_done);
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
    auto handler = [workers](size_t rank, const auto& request, auto& response) {
        EXPECT_EQ(request.completion_forget().keys_size(), 2);
        answer(rank, request, response, [&](const TransferCompletionKeyPB& key) {
            return key.trace_case() == TransferCompletionKeyPB::kTransferTraceId ?
                       (*workers)[rank].ordinary.forget(key.transfer_trace_id()) :
                       (*workers)[rank].kvcm.forget(key.kvcm_trace_id());
        });
        return grpc::Status::OK;
    };
    ForgetServer first(0, handler), second(1, handler);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    const std::vector<std::string> addresses{first.address(), second.address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    TransferCompletionForget forget(manager);
    auto routing = target(addresses);
    std::reverse(routing.peers.begin(), routing.peers.end());
    const auto result = forget.forget(routing, {uint64_t{71}, std::string("71"), uint64_t{0}}, 2000);
    ASSERT_EQ(result.entries.size(), 3u);
    EXPECT_TRUE(result.entries[0].all_done);
    EXPECT_FALSE(result.entries[1].all_done);
    EXPECT_EQ(result.entries[1].ranks[0].result, Outcome::DONE);
    EXPECT_EQ(result.entries[1].ranks[1].result, Outcome::RUNNING);
    EXPECT_EQ(result.entries[2].ranks[0].result, Outcome::INVALID);
    for (size_t rank = 0; rank < 2; ++rank) {
        const auto& row = result.entries[0].ranks[rank];
        EXPECT_EQ(row.rank, rank);
        EXPECT_EQ(row.server_address, addresses[rank]);
        EXPECT_TRUE(row.cq_received);
        EXPECT_TRUE(row.cq_ok);
        EXPECT_EQ(row.grpc_code, std::optional<int>{grpc::StatusCode::OK});
        EXPECT_EQ((*workers)[rank].ordinary.size(), 0u);
    }
    ASSERT_TRUE((*workers)[1].kvcm.finish("71", true));
    const auto retried = forget.forget(routing, {uint64_t{71}, std::string("71")}, 2000);
    EXPECT_TRUE(retried.error.empty());
    EXPECT_TRUE(retried.entries[0].all_done);
    EXPECT_TRUE(retried.entries[1].all_done);
    EXPECT_EQ(first.calls(), 2u);
    EXPECT_EQ(second.calls(), 2u);
}

TEST(TransferCompletionForgetTest, LostAcknowledgmentRetriesTheAlreadyDeletedKeySuccessfully) {
    auto registry = std::make_shared<TransferTaskRegistry>();
    ASSERT_TRUE(registry->begin(uint64_t{81}));
    ASSERT_TRUE(registry->finish(uint64_t{81}, true));
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    ForgetServer server(0, [registry, calls](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response, [&](const auto& key) { return registry->forget(key.transfer_trace_id()); });
        if (calls->fetch_add(1) == 0)
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "forget acknowledgment lost");
        return grpc::Status::OK;
    });
    ASSERT_TRUE(server.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    ASSERT_TRUE(manager->init());
    TransferCompletionForget forget(manager);
    const auto routing = target({server.address()});
    const auto first = forget.forget(routing, {uint64_t{81}}, 2000);
    ASSERT_EQ(first.entries.size(), 1u);
    EXPECT_FALSE(first.entries[0].all_done);
    EXPECT_EQ(first.entries[0].ranks[0].result, Outcome::FAILED);
    EXPECT_EQ(first.entries[0].ranks[0].grpc_code, std::optional<int>{grpc::StatusCode::UNAVAILABLE});
    EXPECT_EQ(registry->size(), 0u);
    const auto second = forget.forget(routing, {uint64_t{81}}, 2000);
    EXPECT_TRUE(second.entries[0].all_done);
    EXPECT_EQ(registry->size(), 0u);
    EXPECT_EQ(calls->load(), 2u);
}

TEST(TransferCompletionForgetTest, InvalidParticipantSetsRejectBeforeRpcOrLocalFallback) {
    ForgetServer server(0, [](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        return grpc::Status::OK;
    });
    ASSERT_TRUE(server.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    ASSERT_TRUE(manager->init());
    size_t local_calls = 0;
    TransferCompletionForget forget(manager, [&](const Trace&) { ++local_calls; return Outcome::DONE; });
    std::vector<TransferCompletionTarget> invalid{
        target({}), target({server.address(), server.address()}), target({""}), target({server.address() + "-other"})};
    invalid.push_back(target({server.address()}));
    invalid.back().peers[0].rank = 1;
    invalid.push_back(target({server.address(), server.address()}));
    invalid.back().peers[1].rank = 0;
    for (const auto& routing : invalid) {
        const auto result = forget.forget(routing, {uint64_t{91}}, 1000);
        EXPECT_FALSE(result.entries[0].all_done);
        EXPECT_FALSE(result.error.empty());
    }
    const auto routing = target({server.address()});
    EXPECT_FALSE(TransferCompletionForget(nullptr).forget(routing, {uint64_t{91}}, 1000).entries[0].all_done);
    auto uninitialized = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    EXPECT_FALSE(TransferCompletionForget(uninitialized).forget(routing, {uint64_t{91}}, 1000).entries[0].all_done);
    EXPECT_FALSE(forget.forget(routing, {uint64_t{0}, std::string{}}, 1000).entries[0].all_done);
    EXPECT_EQ(server.calls(), 0u);
    EXPECT_EQ(local_calls, 0u);
}

TEST(TransferCompletionForgetTest, MalformedRepliesCannotAcknowledgeTheWrongRankKeyOrNamespace) {
    auto mode = std::make_shared<std::atomic<int>>(0);
    ForgetServer server(0, [mode](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        auto* reply = response.mutable_completion_forget();
        switch (mode->load()) {
            case 0: response.Clear(); break;
            case 1: reply->clear_worker(); break;
            case 2: reply->set_rank(rank + 1); break;
            case 3: reply->clear_entries(); break;
            case 4: {
                const auto extra = reply->entries(0);
                *reply->add_entries() = extra;
                break;
            }
            case 5: reply->mutable_entries(0)->mutable_key()->Clear(); break;
            case 6: reply->mutable_entries(0)->mutable_key()->set_transfer_trace_id(92); break;
            case 7: reply->mutable_entries(0)->mutable_key()->set_kvcm_trace_id("91"); break;
            case 8: reply->mutable_entries(0)->set_status(static_cast<TransferCompletionForgetStatusPB>(99)); break;
            case 9: reply->mutable_entries(0)->set_status(TRANSFER_FORGET_FAILED); break;
            case 10: reply->mutable_entries(0)->set_status(TRANSFER_FORGET_INVALID); break;
        }
        return grpc::Status::OK;
    });
    ASSERT_TRUE(server.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    ASSERT_TRUE(manager->init());
    TransferCompletionForget forget(manager);
    for (int malformed = 0; malformed < 11; ++malformed) {
        SCOPED_TRACE(malformed);
        mode->store(malformed);
        const auto result = forget.forget(target({server.address()}), {uint64_t{91}, std::string("unaffected")}, 2000);
        ASSERT_EQ(result.entries.size(), 2u);
        EXPECT_FALSE(result.entries[0].all_done);
        EXPECT_NE(result.entries[0].ranks[0].result, Outcome::DONE);
        EXPECT_FALSE(result.entries[0].ranks[0].error.empty());
        EXPECT_EQ(result.entries[1].all_done, malformed >= 5);
    }
}

TEST(TransferCompletionForgetTest, DeadlineKeepsUnacknowledgedRanksPendingAndRetryIsIdempotent) {
    struct SlowState {
        TransferTaskRegistry registry;
        std::atomic<size_t> calls{0};
        std::promise<void> entered;
        std::promise<void> finished;
    };
    auto slow = std::make_shared<SlowState>();
    ASSERT_TRUE(slow->registry.begin("slow-forget"));
    ASSERT_TRUE(slow->registry.finish("slow-forget", true));
    auto entered = slow->entered.get_future();
    auto finished = slow->finished.get_future();
    auto first = std::make_shared<ForgetServer>(0, [](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        return grpc::Status::OK;
    });
    auto second = std::make_shared<ForgetServer>(1, [slow](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response, [&](const auto& key) { return slow->registry.forget(key.kvcm_trace_id()); });
        if (slow->calls.fetch_add(1) == 0) {
            slow->entered.set_value();
            std::this_thread::sleep_for(1200ms);
            slow->finished.set_value();
        }
        return grpc::Status::OK;
    });
    ASSERT_TRUE(first->start());
    ASSERT_TRUE(second->start());
    const std::vector<std::string> addresses{first->address(), second->address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    const auto routing = target(addresses);
    BoundedThread<TransferCompletionForgetResult> running([first, second, manager, routing] {
        return TransferCompletionForget(manager).forget(routing, {std::string("slow-forget")}, 500);
    });
    ASSERT_EQ(entered.wait_for(2s), std::future_status::ready);
    ASSERT_EQ(running.waitFor(3s), std::future_status::ready);
    const auto result = running.get();
    ASSERT_EQ(result.entries.size(), 1u);
    EXPECT_FALSE(result.entries[0].all_done);
    ASSERT_EQ(result.entries[0].ranks.size(), 2u);
    EXPECT_EQ(result.entries[0].ranks[0].result, Outcome::DONE);
    EXPECT_EQ(result.entries[0].ranks[1].result, Outcome::FAILED);
    EXPECT_FALSE(result.error.empty());
    EXPECT_EQ(slow->registry.size(), 0u);
    ASSERT_EQ(finished.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(result.entries[0].ranks[1].result, Outcome::FAILED);
    const auto retried = TransferCompletionForget(manager).forget(routing, {std::string("slow-forget")}, 2000);
    EXPECT_TRUE(retried.entries[0].all_done);
    EXPECT_EQ(result.entries[0].ranks[1].result, Outcome::FAILED);
}

}  // namespace
}  // namespace rtp_llm
