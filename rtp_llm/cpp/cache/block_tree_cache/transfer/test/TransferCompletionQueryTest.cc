#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"

#include <algorithm>
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

using State = TransferTaskRegistry::State;
using block_tree_cache_test::BoundedThread;
using namespace std::chrono_literals;

using Handler = std::function<grpc::Status(size_t, const FunctionRequestPB&, FunctionResponsePB&)>;

class QueryService final: public RpcService::Service {
public:
    QueryService(size_t rank, Handler handler): rank_(rank), handler_(std::move(handler)) {}

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

class QueryServer {
public:
    QueryServer(size_t rank, Handler handler): service_(rank, std::move(handler)) {}
    ~QueryServer() {
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
    QueryService service_;
    std::unique_ptr<grpc::Server> server_;
    int port_{0};
};

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

FrozenTask::ExecutionInfo execution(TransferExecutionTrace trace,
                                    const std::vector<std::string>& addresses,
                                    bool remote = true) {
    FrozenTask::ExecutionInfo info;
    info.trace_id = std::move(trace);
    info.remote = remote;
    info.expected_rank_count = addresses.size();
    for (size_t rank = 0; rank < addresses.size(); ++rank)
        info.expected_peers.push_back({rank, addresses[rank]});
    return info;
}

TEST(TransferCompletionQueryTest, LocalReadPreservesNamespacesAndOnlyObservesExistingRecords) {
    TransferTaskRegistry ordinary;
    TransferTaskRegistry kvcm;
    ASSERT_TRUE(ordinary.begin(uint64_t{71}));
    ASSERT_TRUE(kvcm.begin(std::string("71")));
    ASSERT_TRUE(kvcm.finish(std::string("71"), true));
    size_t calls = 0;
    TransferCompletionQuery query(nullptr, [&](const TransferExecutionTrace& trace) {
        ++calls;
        if (const auto* numeric = std::get_if<uint64_t>(&trace))
            return ordinary.query(ordinary.epoch(), std::vector<uint64_t>{*numeric}).front();
        return kvcm.query(kvcm.epoch(), std::vector<std::string>{std::get<std::string>(trace)}).front();
    });
    auto numeric = execution(uint64_t{71}, {"local"}, false);
    auto text = execution(std::string("71"), {"local"}, false);
    auto first = query.query(numeric, 1000);
    EXPECT_FALSE(first.all_safe);
    ASSERT_EQ(first.ranks.size(), 1u);
    EXPECT_EQ(first.ranks[0].state, State::RUNNING);
    EXPECT_TRUE(first.error.empty());
    EXPECT_FALSE(first.ranks[0].cq_received);
    EXPECT_FALSE(first.ranks[0].grpc_code.has_value());
    EXPECT_EQ(first.trace_id, numeric.trace_id);
    auto second = query.query(text, 1000);
    EXPECT_TRUE(second.all_safe);
    EXPECT_EQ(second.trace_id, text.trace_id);
    EXPECT_EQ(second.ranks[0].state, State::BUFFER_SAFE);
    ASSERT_TRUE(ordinary.finish(uint64_t{71}, true));
    EXPECT_TRUE(query.query(numeric, 1000).all_safe);
    EXPECT_EQ(first.ranks[0].state, State::RUNNING);  // Prior snapshots never change.
    EXPECT_EQ(ordinary.size(), 1u);
    EXPECT_EQ(kvcm.size(), 1u);
    EXPECT_EQ(calls, 3u);
}

TEST(TransferCompletionQueryTest, InvalidLocalInputAndReaderFailuresCannotProveSafe) {
    size_t calls = 0;
    TransferCompletionQuery query(nullptr, [&](const auto&) {
        ++calls;
        return State::BUFFER_SAFE;
    });
    auto valid = execution(uint64_t{12}, {"local"}, false);
    for (int timeout : {0, -1}) {
        const auto result = query.query(valid, timeout);
        EXPECT_FALSE(result.all_safe);
        EXPECT_EQ(result.ranks[0].state, State::UNKNOWN);
        EXPECT_FALSE(result.error.empty());
    }
    for (const TransferExecutionTrace& trace : {TransferExecutionTrace{uint64_t{0}},
                                              TransferExecutionTrace{std::string{}}}) {
        auto invalid = valid;
        invalid.trace_id = trace;
        EXPECT_FALSE(query.query(invalid, 1000).all_safe);
    }
    auto missing = valid;
    missing.expected_rank_count = 0;
    missing.expected_peers.clear();
    EXPECT_FALSE(query.query(missing, 1000).all_safe);
    auto multiple = execution(uint64_t{12}, {"local", "other"}, false);
    EXPECT_FALSE(query.query(multiple, 1000).all_safe);
    EXPECT_EQ(calls, 0u);
    EXPECT_FALSE(TransferCompletionQuery(nullptr).query(valid, 1000).all_safe);

    TransferCompletionQuery throwing(nullptr, [](const auto&) -> State { throw std::runtime_error("reader failed"); });
    const auto failure = throwing.query(valid, 1000);
    EXPECT_FALSE(failure.all_safe);
    EXPECT_EQ(failure.ranks[0].state, State::UNKNOWN);
    EXPECT_EQ(failure.error, "reader failed");
    TransferCompletionQuery unknown(nullptr, [](const auto&) { return State::UNKNOWN; });
    const auto not_found = unknown.query(valid, 1000);
    EXPECT_FALSE(not_found.all_safe);
    EXPECT_TRUE(not_found.error.empty());
    TransferCompletionQuery invalid(nullptr, [](const auto&) { return static_cast<State>(99); });
    EXPECT_FALSE(invalid.query(valid, 1000).all_safe);
}

TEST(TransferCompletionQueryTest, RemoteAllSafeRequiresEveryOriginalRankForBothTraceNamespaces) {
    auto second_state = std::make_shared<std::atomic<int>>(TRANSFER_COMPLETION_BUFFER_SAFE);
    auto handler = [second_state](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response, rank == 0 ? TRANSFER_COMPLETION_BUFFER_SAFE :
            static_cast<TransferCompletionStatePB>(second_state->load()));
        return grpc::Status::OK;
    };
    QueryServer first(0, handler);
    QueryServer second(1, handler);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    const std::vector<std::string> addresses{first.address(), second.address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    TransferCompletionQuery query(manager);
    for (const TransferExecutionTrace& trace : {TransferExecutionTrace{uint64_t{77}},
                                              TransferExecutionTrace{std::string("77")}}) {
        for (const auto state : {TRANSFER_COMPLETION_BUFFER_SAFE, TRANSFER_COMPLETION_RUNNING,
                                TRANSFER_COMPLETION_UNKNOWN}) {
            second_state->store(state);
            auto info = execution(trace, addresses);
            // Validation uses rank identities rather than the vector's order.
            std::reverse(info.expected_peers.begin(), info.expected_peers.end());
            const auto result = query.query(info, 2000);
            EXPECT_EQ(result.trace_id, trace);
            EXPECT_EQ(result.all_safe, state == TRANSFER_COMPLETION_BUFFER_SAFE);
            EXPECT_TRUE(result.error.empty());
            ASSERT_EQ(result.ranks.size(), 2u);
            EXPECT_EQ(result.ranks[0].state, State::BUFFER_SAFE);
            EXPECT_EQ(result.ranks[1].state, state == TRANSFER_COMPLETION_BUFFER_SAFE ? State::BUFFER_SAFE :
                                           state == TRANSFER_COMPLETION_RUNNING ? State::RUNNING : State::UNKNOWN);
            for (size_t rank = 0; rank < 2; ++rank) {
                EXPECT_EQ(result.ranks[rank].rank, rank);
                EXPECT_EQ(result.ranks[rank].server_address, addresses[rank]);
                EXPECT_TRUE(result.ranks[rank].cq_received);
                EXPECT_TRUE(result.ranks[rank].cq_ok);
                EXPECT_EQ(result.ranks[rank].grpc_code, std::optional<int>{grpc::StatusCode::OK});
                EXPECT_TRUE(result.ranks[rank].error.empty());
            }
        }
    }
    EXPECT_EQ(first.calls(), 6u);
    EXPECT_EQ(second.calls(), 6u);
}

TEST(TransferCompletionQueryTest, ParticipantMismatchRejectsBeforeAnyRpcOrLocalFallback) {
    QueryServer server(0, [](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        return grpc::Status::OK;
    });
    ASSERT_TRUE(server.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    ASSERT_TRUE(manager->init());
    size_t local_calls = 0;
    TransferCompletionQuery query(manager, [&](const auto&) { ++local_calls; return State::BUFFER_SAFE; });
    const auto valid = execution(uint64_t{88}, {server.address()});
    std::vector<FrozenTask::ExecutionInfo> invalid;
    invalid.push_back(execution(uint64_t{88}, {}));
    invalid.push_back(execution(uint64_t{88}, {server.address(), server.address()}));
    invalid.push_back(valid);
    invalid.back().expected_rank_count = 2;
    invalid.push_back(valid);
    invalid.back().expected_peers.clear();
    invalid.push_back(valid);
    invalid.back().expected_peers[0].address += "-different";
    invalid.push_back(valid);
    invalid.back().expected_peers[0].address.clear();
    invalid.push_back(valid);
    invalid.back().expected_peers[0].rank = 1;
    auto duplicate = execution(uint64_t{88}, {server.address(), server.address()});
    duplicate.expected_peers[1].rank = 0;
    invalid.push_back(std::move(duplicate));
    for (const auto& info : invalid) {
        const auto result = query.query(info, 1000);
        EXPECT_FALSE(result.all_safe);
        EXPECT_FALSE(result.error.empty());
        for (const auto& row : result.ranks)
            EXPECT_EQ(row.state, State::UNKNOWN);
    }
    EXPECT_FALSE(TransferCompletionQuery(nullptr).query(valid, 1000).all_safe);
    auto not_started = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    EXPECT_FALSE(TransferCompletionQuery(not_started).query(valid, 1000).all_safe);
    EXPECT_EQ(server.calls(), 0u);
    EXPECT_EQ(local_calls, 0u);
}

TEST(TransferCompletionQueryTest, MalformedRankKeyAndStateResponsesNeverProveSafe) {
    auto mode = std::make_shared<std::atomic<int>>(0);
    QueryServer server(0, [mode](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        auto* query = response.mutable_completion_query();
        switch (mode->load()) {
            case 0: response.Clear(); break;
            case 1: query->clear_worker(); break;
            case 2: query->set_rank(rank + 1); break;
            case 3: query->clear_entries(); break;
            case 4: {
                const auto extra = query->entries(0);
                *query->add_entries() = extra;
                break;
            }
            case 5: query->mutable_entries(0)->mutable_key()->Clear(); break;
            case 6: {
                auto* key = query->mutable_entries(0)->mutable_key();
                if (key->trace_case() == TransferCompletionKeyPB::kTransferTraceId)
                    key->set_transfer_trace_id(key->transfer_trace_id() + 1);
                else
                    key->set_kvcm_trace_id(key->kvcm_trace_id() + "-other");
                break;
            }
            case 7: {
                auto* key = query->mutable_entries(0)->mutable_key();
                if (key->trace_case() == TransferCompletionKeyPB::kTransferTraceId)
                    key->set_kvcm_trace_id("91");
                else
                    key->set_transfer_trace_id(91);
                break;
            }
            case 8: query->mutable_entries(0)->set_state(static_cast<TransferCompletionStatePB>(99)); break;
        }
        return grpc::Status::OK;
    });
    ASSERT_TRUE(server.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    ASSERT_TRUE(manager->init());
    TransferCompletionQuery query(manager);
    for (const TransferExecutionTrace& trace : {TransferExecutionTrace{uint64_t{91}},
                                              TransferExecutionTrace{std::string("91")}}) {
        for (int malformed = 0; malformed < 9; ++malformed) {
            SCOPED_TRACE(malformed);
            mode->store(malformed);
            const auto result = query.query(execution(trace, {server.address()}), 2000);
            EXPECT_FALSE(result.all_safe);
            ASSERT_EQ(result.ranks.size(), 1u);
            EXPECT_EQ(result.ranks[0].state, State::UNKNOWN);
            EXPECT_FALSE(result.ranks[0].error.empty());
            EXPECT_EQ(result.ranks[0].grpc_code, std::optional<int>{grpc::StatusCode::OK});
        }
    }
}

TEST(TransferCompletionQueryTest, GrpcErrorCannotPromoteAnOtherwiseSafeResponse) {
    QueryServer server(0, [](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "worker temporarily unavailable");
    });
    ASSERT_TRUE(server.start());
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{server.address()});
    ASSERT_TRUE(manager->init());
    const auto result = TransferCompletionQuery(manager).query(execution(uint64_t{31}, {server.address()}), 2000);
    EXPECT_FALSE(result.all_safe);
    ASSERT_EQ(result.ranks.size(), 1u);
    EXPECT_EQ(result.ranks[0].state, State::UNKNOWN);
    EXPECT_EQ(result.ranks[0].grpc_code, std::optional<int>{grpc::StatusCode::UNAVAILABLE});
    EXPECT_EQ(result.ranks[0].error, "worker temporarily unavailable");
}

TEST(TransferCompletionQueryTest, DeadlineReturnsUnknownAndLateResponseCannotChangeTheSnapshot) {
    struct SlowState {
        std::atomic<size_t> calls{0};
        std::promise<void> entered;
        std::promise<void> finished;
    };
    auto slow = std::make_shared<SlowState>();
    auto entered = slow->entered.get_future();
    auto finished = slow->finished.get_future();
    auto first = std::make_shared<QueryServer>(0, [](size_t rank, const auto& request, auto& response) {
        answer(rank, request, response);
        return grpc::Status::OK;
    });
    auto second = std::make_shared<QueryServer>(1, [slow](size_t rank, const auto& request, auto& response) {
        if (slow->calls.fetch_add(1) == 0) {
            slow->entered.set_value();
            std::this_thread::sleep_for(1200ms);
            slow->finished.set_value();
        }
        answer(rank, request, response);
        return grpc::Status::OK;
    });
    ASSERT_TRUE(first->start());
    ASSERT_TRUE(second->start());
    const std::vector<std::string> addresses{first->address(), second->address()};
    auto manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(manager->init());
    const auto info = execution(std::string("slow-query"), addresses);
    // Shared captures keep RPC owners alive even if the outer deadline catches
    // a regression that accidentally waits forever in BroadcastResult.
    BoundedThread<TransferCompletionQueryResult> running([first, second, manager, info] {
        return TransferCompletionQuery(manager).query(info, 500);
    });
    ASSERT_EQ(entered.wait_for(2s), std::future_status::ready);
    ASSERT_EQ(running.waitFor(3s), std::future_status::ready);
    const auto result = running.get();
    EXPECT_FALSE(result.all_safe);
    ASSERT_EQ(result.ranks.size(), 2u);
    EXPECT_EQ(result.ranks[0].state, State::BUFFER_SAFE);
    EXPECT_EQ(result.ranks[1].state, State::UNKNOWN);
    EXPECT_FALSE(result.error.empty());
    ASSERT_EQ(finished.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(result.ranks[1].state, State::UNKNOWN);
    const auto next = TransferCompletionQuery(manager).query(info, 2000);
    EXPECT_TRUE(next.all_safe);
    EXPECT_EQ(result.ranks[1].state, State::UNKNOWN);
}

}  // namespace
}  // namespace rtp_llm
