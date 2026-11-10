#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <condition_variable>
#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::BoundedThread;
using RegistryState = TransferTaskRegistry::State;
using namespace std::chrono_literals;

class QuerySdkGate {
public:
    bool enterAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        return cv_.wait_for(lock, 10s, [&] { return released_; });
    }
    bool waitEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, 5s, [&] { return entered_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_{false};
    bool released_{false};
};

struct QueryWorker {
    std::shared_ptr<BackendEnvironment> environment;
    std::shared_ptr<MockClientWrapper> client;
    std::shared_ptr<KVCMStorageBackend> backend;
    std::shared_ptr<KVCacheManager> manager;

    bool init(size_t rank = 0, size_t ranks = 1, std::shared_ptr<BroadcastManager> broadcast = nullptr) {
        environment = std::make_shared<BackendEnvironment>(makeBackendEnvironment("kvcm_query_worker"));
        client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto parallel = singleRankConfig();
        parallel.tp_rank = rank;
        parallel.tp_size = ranks;
        parallel.world_rank = rank;
        auto handle = makeBackend(*environment, parallel, client, std::move(broadcast));
        backend = std::shared_ptr<KVCMStorageBackend>(std::move(handle.backend));
        if (!initSingleRank(*backend, *environment))
            return false;
        manager = std::make_shared<KVCacheManager>(environment->cache_config, false, nullptr,
                                                   KVCacheConfig{}, parallel);
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{environment->device_pool}, nullptr, nullptr);
        full->initialize(0, environment->cache_config.topologyPtr(), {"default"});
        manager->block_tree_cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({full});
        if (!manager->block_tree_cache_)
            return false;
        manager->block_tree_cache_->storage_backend_ = backend;
        return true;
    }
};

class QueryWorkerService final: public RpcService::Service {
public:
    explicit QueryWorkerService(std::shared_ptr<QueryWorker> worker): worker_(std::move(worker)) {}

    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        const bool read = request->has_remote_request() && request->remote_request().op() == REMOTE_OPERATION_READ;
        block_tree_cache_detail::ScopeRollback complete([this, read] {
            if (read) {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completed_reads_;
                cv_.notify_all();
            }
        });
        return worker_->manager->executeFunction(*request, *response) ? grpc::Status::OK :
                   grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "request was not handled");
    }

    bool waitReadCompleted() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, 5s, [&] { return completed_reads_ != 0; });
    }

private:
    std::shared_ptr<QueryWorker> worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t completed_reads_{0};
};

class QueryWorkerServer {
public:
    QueryWorkerServer(std::shared_ptr<QueryWorker> worker, std::shared_ptr<QuerySdkGate> gate = nullptr):
        service_(std::make_shared<QueryWorkerService>(std::move(worker))), gate_(std::move(gate)) {}

    ~QueryWorkerServer() {
        // This also releases blocked SDK work after a fatal assertion.
        if (gate_)
            gate_->release();
        if (server_) {
            BoundedThread<void> shutdown([server = server_, service = service_] {
                (void)service;
                server->Shutdown(std::chrono::system_clock::now() + 2s);
                server->Wait();
            });
            const auto ready = shutdown.waitFor(5s);
            EXPECT_EQ(ready, std::future_status::ready);
            if (ready == std::future_status::ready)
                shutdown.get();
        }
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        return server_ && port_ > 0;
    }
    std::string address() const { return "127.0.0.1:" + std::to_string(port_); }
    bool waitReadCompleted() { return service_->waitReadCompleted(); }

private:
    std::shared_ptr<QueryWorkerService> service_;
    std::shared_ptr<QuerySdkGate> gate_;
    std::shared_ptr<grpc::Server> server_;
    int port_{0};
};

FunctionRequestPB payloadRequest(const BackendEnvironment& environment, const std::string& trace) {
    FunctionRequestPB request;
    auto* payload = request.mutable_remote_request();
    payload->set_op(REMOTE_OPERATION_READ);
    payload->set_trace_id(trace);
    payload->set_timeout_ms(30000);
    payload->add_group_tags("default");
    payload->add_block_ids(environment.block_id);
    payload->add_uris("pace://query-payload");
    return request;
}

TEST(KVCMCompletionQueryTest, QueryNamespacesStaySeparateAndObserveReadCompletionWithoutSdkCalls) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto worker = std::make_shared<QueryWorker>();
    ASSERT_TRUE(worker->init());
    auto gate = std::make_shared<QuerySdkGate>();
    EXPECT_CALL(*worker->client, loadKvCachesForTag(_, _, _, _)).WillOnce(Invoke(
        [gate](const auto&, const auto&, auto&, const auto&) {
            const bool released = gate->enterAndWait();
            EXPECT_TRUE(released);
            return released;
        }));
    EXPECT_CALL(*worker->client, saveKvCachesForTag(_, _, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"pace://saved"})));
    EXPECT_CALL(*worker->client, match(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*worker->client, getWriteLocation(_, _, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*worker->client, finishWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*worker->client, removeCache(_, _, _, _, _)).Times(0);
    auto& ordinary = *worker->manager->block_tree_cache_->transfer_dispatcher_->task_registry_;
    ASSERT_TRUE(ordinary.begin(uint64_t{71}));
    ASSERT_TRUE(ordinary.finish(uint64_t{71}, true));

    QueryWorkerServer server(worker, gate);
    ASSERT_TRUE(server.start());
    auto channel = grpc::CreateChannel(server.address(), grpc::InsecureChannelCredentials());
    ASSERT_TRUE(channel->WaitForConnected(std::chrono::system_clock::now() + 5s));
    const auto read = payloadRequest(*worker->environment, "71");
    BoundedThread<grpc::Status> running([channel, read] {
        auto stub = RpcService::NewStub(channel);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 8s);
        FunctionResponsePB response;
        return stub->ExecuteFunction(&context, read, &response);
    });
    ASSERT_TRUE(gate->waitEntered());

    auto write = payloadRequest(*worker->environment, "write-71");
    write.mutable_remote_request()->set_op(REMOTE_OPERATION_WRITE);
    FunctionResponsePB write_response;
    ASSERT_TRUE(worker->manager->executeFunction(write, write_response));
    ASSERT_TRUE(write_response.remote_response().payload_buffer_safe());

    FunctionRequestPB request;
    auto* query = request.mutable_completion_query();
    query->add_keys()->set_kvcm_trace_id("71");
    query->add_keys()->set_transfer_trace_id(71);
    query->add_keys()->set_kvcm_trace_id("write-71");
    query->add_keys()->set_kvcm_trace_id("missing-metadata-id");
    FrozenTask::ExecutionInfo local_execution;
    local_execution.trace_id = std::string("71");
    local_execution.expected_rank_count = 1;
    local_execution.expected_peers = {{0, "local"}};
    for (const bool completed : {false, true}) {
        if (completed) {
            gate->release();
            ASSERT_TRUE(server.waitReadCompleted());
            ASSERT_EQ(running.waitFor(5s), std::future_status::ready);
            EXPECT_TRUE(running.get().ok());
        }
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 2s);
        FunctionResponsePB response;
        auto stub = RpcService::NewStub(channel);
        ASSERT_TRUE(stub->ExecuteFunction(&context, request, &response).ok());
        ASSERT_TRUE(response.has_completion_query());
        const auto& observed = response.completion_query();
        ASSERT_EQ(observed.entries_size(), 4);
        EXPECT_EQ(observed.rank(), 0u);
        EXPECT_EQ(observed.entries(0).key().kvcm_trace_id(), "71");
        EXPECT_EQ(observed.entries(0).state(), completed ? TRANSFER_COMPLETION_BUFFER_SAFE :
                                                         TRANSFER_COMPLETION_RUNNING);
        EXPECT_EQ(observed.entries(1).key().transfer_trace_id(), 71u);
        EXPECT_EQ(observed.entries(1).state(), TRANSFER_COMPLETION_BUFFER_SAFE);
        EXPECT_EQ(observed.entries(2).state(), TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_EQ(observed.entries(3).state(), TRANSFER_COMPLETION_UNKNOWN);
        const auto local = worker->manager->queryTransferCompletion(local_execution, 2000);
        ASSERT_EQ(local.ranks.size(), 1u);
        EXPECT_EQ(local.ranks.front().state, completed ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
        EXPECT_EQ(local.all_safe, completed);
        EXPECT_EQ(worker->backend->taskRegistry().size(), 1u);
        EXPECT_EQ(ordinary.size(), 1u);
    }
}

TEST(KVCMCompletionQueryTest, OriginalRpcParticipantsBecomeSafeWithoutThawingFrozenRead) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto worker0 = std::make_shared<QueryWorker>();
    auto worker1 = std::make_shared<QueryWorker>();
    auto gate = std::make_shared<QuerySdkGate>();
    QueryWorkerServer server0(worker0, gate);
    QueryWorkerServer server1(worker1);
    ASSERT_TRUE(server0.start());
    ASSERT_TRUE(server1.start());
    const std::vector<std::string> addresses{server0.address(), server1.address()};
    auto broadcast = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(broadcast->init());
    // Rank zero must have the full group before KVCM initialization. No
    // payload/query is submitted until both worker managers are installed.
    ASSERT_TRUE(worker0->init(0, 2, broadcast));
    ASSERT_TRUE(worker1->init(1, 2));
    EXPECT_CALL(*worker0->client, loadKvCachesForTag(_, _, _, _)).WillOnce(Invoke(
        [gate](const auto&, const auto&, auto&, const auto&) {
            const bool released = gate->enterAndWait();
            EXPECT_TRUE(released);
            return released;
        }));
    EXPECT_CALL(*worker1->client, loadKvCachesForTag(_, _, _, _)).WillOnce(Return(true));

    auto environment = makeBackendEnvironment("kvcm_query_coordinator");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
    auto parallel = singleRankConfig();
    parallel.tp_size = 2;
    auto backend = makeBackend(environment, parallel, client, broadcast);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://rank-0"},
                                                 {"tp1_Fdefault", "pace://rank-1"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto matched = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(matched.success);
    auto request = makeStorageRequest(environment);
    request.transfer_task.emplace(std::vector<TransferDescriptor>{}, 2s);
    auto done = std::make_shared<std::promise<ErrorInfo>>();
    auto completed = done->get_future();
    backend->read(std::move(request), std::move(matched.match_meta),
                  [done](ErrorInfo error) { done->set_value(std::move(error)); });
    ASSERT_TRUE(gate->waitEntered());
    ASSERT_EQ(completed.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(completed.get().code(), ErrorCode::DEADLINE_EXCEEDED);
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    const auto frozen = backend->frozenTasks();
    ASSERT_EQ(frozen.size(), 1u);
    const auto record = frozen.front();
    ASSERT_EQ(record->execution().expected_peers.size(), 2u);
    EXPECT_EQ(std::get<std::string>(record->execution().trace_id), "block_tree_read_1");
    const auto original = record->debugString();
    const auto references = environment.device_pool->refCount(environment.block_id);
    TransferCompletionQuery query(broadcast);
    for (const bool completed : {false, true}) {
        if (completed) {
            gate->release();
            ASSERT_TRUE(server0.waitReadCompleted());
        }
        const auto observed = query.query(record->execution(), 2000);
        ASSERT_EQ(observed.ranks.size(), 2u);
        EXPECT_TRUE(observed.error.empty()) << observed.error;
        EXPECT_EQ(observed.all_safe, completed);
        EXPECT_EQ(observed.ranks[0].state, completed ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
        EXPECT_EQ(observed.ranks[1].state, RegistryState::BUFFER_SAFE);
        for (size_t rank = 0; rank < 2; ++rank) {
            EXPECT_EQ(observed.ranks[rank].rank, rank);
            EXPECT_EQ(observed.ranks[rank].server_address, addresses[rank]);
            EXPECT_EQ(observed.ranks[rank].grpc_code, std::optional<int>{grpc::StatusCode::OK});
        }
        EXPECT_EQ(record->snapshot().state, FrozenTask::State::FROZEN);
        EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), references);
        EXPECT_EQ(record->debugString(), original);
        EXPECT_EQ(backend->frozenTasks().size(), 1u);
    }
}

}  // namespace
}  // namespace rtp_llm
