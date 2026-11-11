#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <condition_variable>
#include <grpcpp/grpcpp.h>
#include <thread>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

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

std::shared_ptr<KVCacheManager> makeQueryManager(const BackendEnvironment& environment,
                                                const std::shared_ptr<KVCMStorageBackend>& backend,
                                                const ParallelismConfig& parallel,
                                                std::shared_ptr<BroadcastManager> broadcast = nullptr,
                                                KVCacheConfig config = {}) {
    auto manager = std::make_shared<KVCacheManager>(environment.cache_config, false, nullptr,
                                                   config, parallel);
    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
    full->initialize(0, environment.cache_config.topologyPtr(), {"default"});
    manager->block_tree_cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({full});
    if (!manager->block_tree_cache_)
        return nullptr;
    manager->block_tree_cache_->storage_backend_ = backend;
    manager->transfer_broadcast_manager_ = std::move(broadcast);
    return manager;
}

template<typename Predicate>
bool waitUntil(Predicate&& predicate, std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(5ms);
    }
    return true;
}

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
        manager = makeQueryManager(*environment, backend, parallel);
        return manager != nullptr;
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

template<typename Code>
std::vector<Code> safeReadCompletionCodes() {
    std::vector<Code> codes{Code::ER_OK};
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        codes.push_back(Code::ER_SDK_ERROR_BUFFER_SAFE);
    }
    return codes;
}

TEST(KVCMCompletionQueryTest, LocalReadSettlementQueuesCleanupForSuccessAndSafeFailure) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool asynchronous : {false, true}) {
        for (const auto code : safeReadCompletionCodes<kv_cache_manager::ClientErrorCode>()) {
            SCOPED_TRACE(::testing::PrintToString(std::make_pair(asynchronous, code)));
            auto worker = std::make_shared<QueryWorker>();
            ASSERT_TRUE(worker->init());
            auto cleanup = std::make_shared<TransferCompletionCleanup>();
            worker->backend->setCompletionCleanup(cleanup);
            worker->client->load_status = code;
            EXPECT_CALL(*worker->client, loadKvCachesForTag(_, _, _, _)).Times(0);
            const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://cleanup-local"}}};
            EXPECT_CALL(*worker->client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
            const auto& environment = *worker->environment;
            const auto references = environment.device_pool->refCount(environment.block_id);
            auto matched = match(*worker->backend, makeStorageRequest(environment));
            ASSERT_TRUE(matched.success);
            EXPECT_EQ(cleanup->size(), 0u);
            EXPECT_EQ(worker->backend->taskRegistry().size(), 0u);
            auto request = makeStorageRequest(environment);
            const bool success = code == kv_cache_manager::ClientErrorCode::ER_OK;
            if (asynchronous) {
                EXPECT_EQ(readResult(*worker->backend, std::move(request), std::move(matched.match_meta)).ok(), success);
                ASSERT_TRUE(waitForBackendOperationsForTest(*worker->backend));
            } else {
                request.transfer_task.emplace(std::vector<TransferDescriptor>{}, 30s);
                if (success) {
                    EXPECT_NO_THROW(worker->backend->readImpl(request, matched.match_meta));
                } else {
                    EXPECT_THROW(worker->backend->readImpl(request, matched.match_meta), StorageOperationBufferSafeFailure);
                }
            }
            EXPECT_TRUE(worker->manager->frozenTasks().empty());
            EXPECT_EQ(worker->backend->taskRegistry().queryLocal(std::vector<std::string>{"block_tree_read_1"}).front(),
                      RegistryState::BUFFER_SAFE);
            ASSERT_EQ(cleanup->size(), 1u);
            size_t forget_calls = 0;
            auto forget = [&](const auto& target, const auto& traces, int timeout_ms) {
                ++forget_calls;
                EXPECT_FALSE(target.remote);
                EXPECT_EQ(target.peers.size(), 1u);
                EXPECT_EQ(traces.size(), 1u);
                return worker->manager->forgetTransferCompletion(target, traces, timeout_ms);
            };
            EXPECT_EQ(cleanup->drain(forget, 2000).forgotten, 1u);
            EXPECT_EQ(cleanup->size(), 0u);
            EXPECT_EQ(worker->backend->taskRegistry().size(), 0u);
            EXPECT_EQ(cleanup->drain(forget, 2000).attempted, 0u);
            EXPECT_EQ(forget_calls, 1u);
            EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), references);
        }
    }
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

    for (const bool missing_trace : {false, true}) {
        auto rejected = read;
        if (missing_trace)
            rejected.mutable_remote_request()->clear_trace_id();
        auto stub = RpcService::NewStub(channel);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 2s);
        FunctionResponsePB response;
        ASSERT_TRUE(stub->ExecuteFunction(&context, rejected, &response).ok());
        ASSERT_TRUE(response.has_remote_response());
        EXPECT_EQ(response.remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
        EXPECT_TRUE(response.remote_response().payload_buffer_safe());
        EXPECT_FALSE(response.remote_response().sdk_failed());
        EXPECT_EQ(worker->backend->taskRegistry().queryLocal(std::vector<std::string>{"71"}).front(),
                  RegistryState::RUNNING);
        EXPECT_EQ(worker->backend->taskRegistry().size(), 1u);
        EXPECT_TRUE(worker->backend->frozenTasks().empty());
    }

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
        EXPECT_FALSE(observed.entries(0).admission_closed());
        EXPECT_FALSE(observed.entries(1).admission_closed());
        EXPECT_TRUE(observed.entries(2).admission_closed());
        EXPECT_TRUE(observed.entries(3).admission_closed());
        const auto local = worker->manager->queryTransferCompletion(local_execution, 2000);
        ASSERT_EQ(local.ranks.size(), 1u);
        EXPECT_EQ(local.ranks.front().state, completed ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
        EXPECT_EQ(local.all_safe, completed);
        EXPECT_EQ(worker->backend->taskRegistry().size(), 3u);
        EXPECT_EQ(ordinary.size(), 1u);
    }
}

TEST(KVCMCompletionQueryTest, RpcBatchForgetKeepsRunningReadsAndSeparatesTraceNamespaces) {
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
    EXPECT_CALL(*worker->client, saveKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*worker->client, match(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*worker->client, getWriteLocation(_, _, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*worker->client, finishWrite(_, _, _, _, _)).Times(0);
    auto& ordinary = *worker->manager->block_tree_cache_->transfer_dispatcher_->task_registry_;
    ASSERT_TRUE(ordinary.begin(uint64_t{71}));
    ASSERT_TRUE(ordinary.finish(uint64_t{71}, true));
    const auto references = worker->environment->device_pool->refCount(worker->environment->block_id);

    QueryWorkerServer server(worker, gate);
    ASSERT_TRUE(server.start());
    auto channel = grpc::CreateChannel(server.address(), grpc::InsecureChannelCredentials());
    ASSERT_TRUE(channel->WaitForConnected(std::chrono::system_clock::now() + 5s));
    const auto read = payloadRequest(*worker->environment, "71");
    BoundedThread<std::pair<grpc::Status, FunctionResponsePB>> running([channel, read] {
        auto stub = RpcService::NewStub(channel);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 8s);
        FunctionResponsePB response;
        auto status = stub->ExecuteFunction(&context, read, &response);
        return std::make_pair(std::move(status), std::move(response));
    });
    ASSERT_TRUE(gate->waitEntered());
    auto rpc = [&](const FunctionRequestPB& request, FunctionResponsePB& response) {
        auto stub = RpcService::NewStub(channel);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 2s);
        return stub->ExecuteFunction(&context, request, &response);
    };

    FunctionRequestPB oversized;
    oversized.mutable_completion_forget()->add_keys()->set_transfer_trace_id(71);
    for (size_t index = 0; index < kTransferCompletionForgetBatchSize; ++index) {
        oversized.mutable_completion_forget()->add_keys()->set_kvcm_trace_id("missing");
    }
    FunctionResponsePB oversized_response;
    EXPECT_EQ(rpc(oversized, oversized_response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(ordinary.queryLocal(std::vector<uint64_t>{71}).front(), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(worker->backend->forgetReadCompletion(""), TransferTaskRegistry::ForgetResult::INVALID);
    EXPECT_EQ(worker->backend->taskRegistry().queryLocal(std::vector<std::string>{"71"}).front(),
              RegistryState::RUNNING);

    FunctionRequestPB request;
    auto* forget = request.mutable_completion_forget();
    forget->add_keys()->set_kvcm_trace_id("71");
    forget->add_keys()->set_transfer_trace_id(71);
    forget->add_keys()->set_kvcm_trace_id("missing");
    forget->add_keys()->set_transfer_trace_id(999);
    forget->add_keys()->set_transfer_trace_id(0);
    forget->add_keys()->set_kvcm_trace_id("");
    forget->add_keys();
    for (const bool completed : {false, true}) {
        if (completed) {
            gate->release();
            ASSERT_EQ(running.waitFor(5s), std::future_status::ready);
            const auto result = running.get();
            ASSERT_TRUE(result.first.ok());
            EXPECT_EQ(result.second.remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_SUCCESS);
            EXPECT_TRUE(result.second.remote_response().payload_buffer_safe());
        }
        FunctionResponsePB response;
        ASSERT_TRUE(rpc(request, response).ok());
        ASSERT_TRUE(response.has_completion_forget());
        const auto& observed = response.completion_forget();
        ASSERT_EQ(observed.entries_size(), 7);
        EXPECT_EQ(observed.rank(), 0u);
        EXPECT_EQ(observed.entries(0).key().kvcm_trace_id(), "71");
        EXPECT_EQ(observed.entries(0).status(), completed ? TRANSFER_FORGET_DONE : TRANSFER_FORGET_RUNNING);
        EXPECT_EQ(observed.entries(1).key().transfer_trace_id(), 71u);
        EXPECT_EQ(observed.entries(1).status(), TRANSFER_FORGET_DONE);
        for (int index = 2; index < 4; ++index) {
            EXPECT_EQ(observed.entries(index).status(), TRANSFER_FORGET_DONE);
        }
        for (int index = 4; index < 7; ++index) {
            EXPECT_EQ(observed.entries(index).status(), TRANSFER_FORGET_INVALID);
        }

        FunctionRequestPB query;
        query.mutable_completion_query()->add_keys()->set_kvcm_trace_id("71");
        query.mutable_completion_query()->add_keys()->set_transfer_trace_id(71);
        FunctionResponsePB query_response;
        ASSERT_TRUE(rpc(query, query_response).ok());
        ASSERT_EQ(query_response.completion_query().entries_size(), 2);
        EXPECT_EQ(query_response.completion_query().entries(0).state(),
                  completed ? TRANSFER_COMPLETION_UNKNOWN : TRANSFER_COMPLETION_RUNNING);
        EXPECT_EQ(query_response.completion_query().entries(1).state(), TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_EQ(query_response.completion_query().entries(0).admission_closed(), completed);
        EXPECT_TRUE(query_response.completion_query().entries(1).admission_closed());
        EXPECT_EQ(worker->backend->taskRegistry().size(), 1u);
        EXPECT_EQ(ordinary.size(), 1u);
        EXPECT_TRUE(worker->backend->frozenTasks().empty());
        EXPECT_FALSE(worker->environment->device_pool->isFrozen(worker->environment->block_id));
        EXPECT_EQ(worker->environment->device_pool->refCount(worker->environment->block_id), references);
    }
}

TEST(KVCMCompletionQueryTest, QueryDoesNotThawAndRecoveryWaitsForAllOriginalRpcParticipants) {
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
        })).WillOnce(Return(true));
    EXPECT_CALL(*worker1->client, loadKvCachesForTag(_, _, _, _)).Times(2).WillRepeatedly(Return(true));

    auto environment = makeBackendEnvironment("kvcm_query_coordinator");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
    auto parallel = singleRankConfig();
    parallel.tp_size = 2;
    auto handle = makeBackend(environment, parallel, client, broadcast);
    auto backend = std::shared_ptr<KVCMStorageBackend>(std::move(handle.backend));
    ASSERT_TRUE(initSingleRank(*backend, environment));
    auto coordinator = makeQueryManager(environment, backend, parallel, broadcast);
    ASSERT_NE(coordinator, nullptr);
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    backend->setCompletionCleanup(cleanup);
    auto forget = [coordinator](const auto& target, const auto& traces, int timeout_ms) {
        return coordinator->forgetTransferCompletion(target, traces, timeout_ms);
    };
    const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://rank-0"},
                                                 {"tp1_Fdefault", "pace://rank-1"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).Times(2).WillRepeatedly(Return(std::make_pair(true, locations)));
    auto matched = match(*backend, makeStorageRequest(environment));
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
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend));
    const auto frozen = coordinator->frozenTasks();
    ASSERT_EQ(frozen.size(), 1u);
    const auto record = frozen.front();
    ASSERT_EQ(record->execution().expected_peers.size(), 2u);
    EXPECT_EQ(std::get<std::string>(record->execution().trace_id), "block_tree_read_1");
    const auto failure_message = record->failure().status.ToString();
    const auto failure_diagnostics = record->failure().execution_diagnostics;
    ASSERT_NE(failure_diagnostics, nullptr);
    const auto original_ranks = failure_diagnostics->ranks;
    const auto references = environment.device_pool->refCount(environment.block_id);
    auto forget_rank_zero = [&] {
        auto stub = RpcService::NewStub(grpc::CreateChannel(addresses.front(), grpc::InsecureChannelCredentials()));
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 2s);
        FunctionRequestPB forget;
        forget.mutable_completion_forget()->add_keys()->set_kvcm_trace_id(
            std::get<std::string>(record->execution().trace_id));
        FunctionResponsePB response;
        const auto status = stub->ExecuteFunction(&context, forget, &response);
        EXPECT_TRUE(status.ok());
        return response;
    };
    const auto running_forget = forget_rank_zero();
    ASSERT_EQ(running_forget.completion_forget().entries_size(), 1);
    EXPECT_EQ(running_forget.completion_forget().entries(0).status(), TRANSFER_FORGET_RUNNING);
    EXPECT_EQ(cleanup->size(), 0u);
    EXPECT_EQ(cleanup->drain(forget, 2000).attempted, 0u);
    EXPECT_EQ(record->snapshot().state, FrozenTask::State::FROZEN);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), references);
    FrozenTaskRecovery recovery([coordinator] { return coordinator->frozenTasks(); },
                                [coordinator](const auto& execution, int timeout_ms) {
                                    return coordinator->queryTransferCompletion(execution, timeout_ms);
                                });
    for (const bool completed : {false, true}) {
        if (completed) {
            gate->release();
            ASSERT_TRUE(server0.waitReadCompleted());
        }
        const auto before_query = record->debugString();
        const auto observed = coordinator->queryTransferCompletion(record->execution(), 2000);
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
        EXPECT_EQ(record->debugString(), before_query);
        EXPECT_EQ(coordinator->frozenTasks().size(), 1u);
        EXPECT_EQ(cleanup->size(), 0u);
        const auto recovered = recovery.runOnce(2000);
        EXPECT_EQ(recovered.queried, 1u);
        EXPECT_EQ(recovered.released, completed ? 1u : 0u);
        EXPECT_EQ(record->snapshot().state, completed ? FrozenTask::State::RELEASED : FrozenTask::State::FROZEN);
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), !completed);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), references - (completed ? 1u : 0u));
        EXPECT_EQ(cleanup->size(), completed ? 1u : 0u);
    }
    EXPECT_EQ(record->failure().status.ToString(), failure_message);
    EXPECT_FALSE(record->failure().buffer_safe);
    EXPECT_EQ(record->failure().execution_diagnostics, failure_diagnostics);
    ASSERT_EQ(failure_diagnostics->ranks.size(), original_ranks.size());
    for (size_t rank = 0; rank < original_ranks.size(); ++rank) {
        EXPECT_EQ(failure_diagnostics->ranks[rank].grpc_code, original_ranks[rank].grpc_code);
        EXPECT_EQ(failure_diagnostics->ranks[rank].buffer_safe, original_ranks[rank].buffer_safe);
        EXPECT_EQ(failure_diagnostics->ranks[rank].worker_transfer_status, original_ranks[rank].worker_transfer_status);
    }
    const auto replay = recovery.runOnce(2000);
    EXPECT_EQ(replay.queried, 0u);
    EXPECT_EQ(replay.released, 0u);
    EXPECT_TRUE(coordinator->frozenTasks().empty());
    for (const auto& worker : {worker0, worker1}) {
        EXPECT_EQ(worker->backend->taskRegistry().size(), 1u);
        const auto trace = std::get<std::string>(record->execution().trace_id);
        const auto duplicate = payloadRequest(*worker->environment, trace);
        FunctionResponsePB response;
        ASSERT_TRUE(worker->manager->executeFunction(duplicate, response));
        EXPECT_EQ(response.remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
        EXPECT_TRUE(response.remote_response().payload_buffer_safe());
        EXPECT_FALSE(response.remote_response().sdk_failed());
        EXPECT_EQ(worker->backend->taskRegistry().queryLocal(std::vector<std::string>{trace}).front(),
                  RegistryState::BUFFER_SAFE);
    }
    const auto released_debug = record->debugString();
    const auto first_forget = forget_rank_zero();
    ASSERT_EQ(first_forget.completion_forget().entries_size(), 1);
    EXPECT_EQ(first_forget.completion_forget().entries(0).status(), TRANSFER_FORGET_DONE);
    const auto partial = coordinator->queryTransferCompletion(record->execution(), 2000);
    ASSERT_EQ(partial.ranks.size(), 2u);
    EXPECT_EQ(partial.ranks[0].state, RegistryState::UNKNOWN);
    EXPECT_EQ(partial.ranks[1].state, RegistryState::BUFFER_SAFE);
    EXPECT_TRUE(partial.ranks[0].admission_closed);
    EXPECT_FALSE(partial.ranks[1].admission_closed);
    EXPECT_TRUE(partial.all_safe);
    // The preallocated token moved through FrozenTask. One rank already
    // deleted its completed record and the query installed an admission fence;
    // the queued aggregate forget preserves that fence and confirms all ranks.
    EXPECT_EQ(cleanup->drain(forget, 2000).forgotten, 1u);
    EXPECT_EQ(cleanup->size(), 0u);

    TransferCompletionTarget target;
    target.remote = true;
    for (size_t rank = 0; rank < addresses.size(); ++rank) {
        target.peers.push_back({rank, addresses[rank]});
    }
    for (int retry = 0; retry < 2; ++retry) {
        const auto forgotten = coordinator->forgetTransferCompletion(target, {record->execution().trace_id}, 2000);
        EXPECT_TRUE(forgotten.error.empty()) << forgotten.error;
        ASSERT_EQ(forgotten.entries.size(), 1u);
        EXPECT_TRUE(forgotten.entries.front().all_done);
        ASSERT_EQ(forgotten.entries.front().ranks.size(), 2u);
        for (const auto& rank : forgotten.entries.front().ranks) {
            EXPECT_EQ(rank.result, TransferTaskRegistry::ForgetResult::DONE);
        }
        const auto queried = coordinator->queryTransferCompletion(record->execution(), 2000);
        ASSERT_EQ(queried.ranks.size(), 2u);
        for (const auto& rank : queried.ranks) {
            EXPECT_EQ(rank.state, RegistryState::UNKNOWN);
            EXPECT_TRUE(rank.admission_closed);
        }
        EXPECT_TRUE(queried.all_safe);
        EXPECT_EQ(worker0->backend->taskRegistry().size(), 1u);
        EXPECT_EQ(worker1->backend->taskRegistry().size(), 1u);
        EXPECT_EQ(record->snapshot().state, FrozenTask::State::RELEASED);
        EXPECT_EQ(record->debugString(), released_debug);
        EXPECT_EQ(record->failure().status.ToString(), failure_message);
        EXPECT_EQ(record->failure().execution_diagnostics, failure_diagnostics);
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), references - 1);
    }
    // Aggregate forget acknowledges cleanup but must not reopen a fenced trace.
    for (const auto& worker : {worker0, worker1}) {
        const auto trace = std::get<std::string>(record->execution().trace_id);
        const auto late = payloadRequest(*worker->environment, trace);
        FunctionResponsePB response;
        ASSERT_TRUE(worker->manager->executeFunction(late, response));
        EXPECT_EQ(response.remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
        EXPECT_TRUE(response.remote_response().payload_buffer_safe());
        EXPECT_EQ(worker->backend->taskRegistry().queryLocal(std::vector<std::string>{trace}).front(),
                  RegistryState::REJECTED);
    }
    auto normal_match = match(*backend, makeStorageRequest(environment));
    ASSERT_TRUE(normal_match.success);
    EXPECT_TRUE(readResult(*backend, makeStorageRequest(environment), std::move(normal_match.match_meta)).ok());
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend));
    EXPECT_TRUE(coordinator->frozenTasks().empty());
    EXPECT_EQ(cleanup->size(), 1u);
    for (const auto& worker : {worker0, worker1}) {
        EXPECT_EQ(worker->backend->taskRegistry().queryLocal(std::vector<std::string>{"block_tree_read_2"}).front(),
                  RegistryState::BUFFER_SAFE);
    }
    EXPECT_EQ(cleanup->drain(forget, 2000).forgotten, 1u);
    EXPECT_EQ(cleanup->size(), 0u);
    EXPECT_EQ(worker0->backend->taskRegistry().size(), 1u);
    EXPECT_EQ(worker1->backend->taskRegistry().size(), 1u);
    EXPECT_EQ(record->debugString(), released_debug);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), references - 1);
}

TEST(KVCMCompletionQueryTest, LocalSdkFailuresRecoverOnlyProvenHoldsOnSharedTargets) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto worker = std::make_shared<QueryWorker>();
    ASSERT_TRUE(worker->init());
    worker->client->load_status = kv_cache_manager::ClientErrorCode::ER_SDKREAD_ERROR;
    EXPECT_CALL(*worker->client, loadKvCachesForTag(_, _, _, _)).Times(0);
    const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://local-sdk-error"}}};
    EXPECT_CALL(*worker->client, match(_, _, _, _, _, _))
        .Times(2).WillRepeatedly(Return(std::make_pair(true, locations)));
    const auto& environment = *worker->environment;
    const auto original_references = environment.device_pool->refCount(environment.block_id);
    for (size_t index = 0; index < 2; ++index) {
        auto matched = match(*worker->backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        EXPECT_FALSE(readResult(*worker->backend, makeStorageRequest(environment), std::move(matched.match_meta)).ok());
        ASSERT_TRUE(waitForBackendOperationsForTest(*worker->backend));
    }
    const auto records = worker->manager->frozenTasks();
    ASSERT_EQ(records.size(), 2u);
    // Independent frozen tree references share one device backing reference.
    ASSERT_EQ(environment.device_pool->refCount(environment.block_id), original_references + 1);
    ASSERT_EQ(environment.device_pool->treeRefCount(environment.block_id), 2u);
    std::vector<std::string> messages;
    for (const auto& record : records) {
        EXPECT_FALSE(record->execution().remote);
        ASSERT_NE(record->failure().execution_diagnostics, nullptr);
        EXPECT_NE(record->failure().execution_diagnostics->failure_reason, "rpc_error");
        EXPECT_NE(record->failure().execution_diagnostics->failure_reason, "rpc_timeout");
        messages.push_back(record->failure().status.ToString());
    }
    FrozenTaskRecovery recovery([manager = worker->manager] { return manager->frozenTasks(); },
                                [manager = worker->manager](const auto& execution, int timeout_ms) {
                                    return manager->queryTransferCompletion(execution, timeout_ms);
                                });
    EXPECT_EQ(recovery.runOnce(2000).released, 0u);
    for (size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        const auto& trace = std::get<std::string>(record->execution().trace_id);
        // Test injection of a subsequent explicit safety proof. This does not
        // claim that the SDK currently exposes a late-completion callback.
        auto& registry = const_cast<TransferTaskRegistry&>(worker->backend->taskRegistry());
        ASSERT_TRUE(registry.finish(trace, true));
        const auto observed = worker->manager->queryTransferCompletion(record->execution(), 2000);
        EXPECT_TRUE(observed.all_safe);
        EXPECT_EQ(record->snapshot().state, FrozenTask::State::FROZEN);
        EXPECT_EQ(recovery.runOnce(2000).released, 1u);
        EXPECT_EQ(record->snapshot().state, FrozenTask::State::RELEASED);
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), index == 0);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_references + (index == 0 ? 1u : 0u));
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u - index);
        EXPECT_EQ(record->failure().status.ToString(), messages[index]);
        EXPECT_FALSE(record->failure().buffer_safe);
        EXPECT_EQ(registry.queryLocal(std::vector<std::string>{trace}).front(), RegistryState::BUFFER_SAFE);
        EXPECT_EQ(worker->manager->frozenTasks().size(), records.size() - index - 1);
    }
    EXPECT_TRUE(worker->manager->frozenTasks().empty());
    EXPECT_EQ(worker->backend->taskRegistry().size(), 2u);
    const auto replay = recovery.runOnce(2000);
    EXPECT_EQ(replay.queried, 0u);
    EXPECT_EQ(replay.released, 0u);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_references);
}

TEST(KVCMCompletionQueryTest, BackgroundRecoveryWaitsForProofAndStopKeepsLateProofFrozen) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool stop_before_safe : {false, true}) {
        SCOPED_TRACE(stop_before_safe);
        auto worker0 = std::make_shared<QueryWorker>();
        auto worker1 = std::make_shared<QueryWorker>();
        auto gate = std::make_shared<QuerySdkGate>();
        QueryWorkerServer server0(worker0, gate);
        QueryWorkerServer server1(worker1);
        ASSERT_TRUE(server0.start());
        ASSERT_TRUE(server1.start());
        auto broadcast = std::make_shared<BroadcastManager>(
            std::vector<std::string>{server0.address(), server1.address()});
        ASSERT_TRUE(broadcast->init());
        ASSERT_TRUE(worker0->init(0, 2, broadcast));
        ASSERT_TRUE(worker1->init(1, 2));
        EXPECT_CALL(*worker0->client, loadKvCachesForTag(_, _, _, _)).WillOnce(Invoke(
            [gate](const auto&, const auto&, auto&, const auto&) {
                const bool released = gate->enterAndWait();
                EXPECT_TRUE(released);
                return released;
            }));
        EXPECT_CALL(*worker1->client, loadKvCachesForTag(_, _, _, _)).WillOnce(Return(true));

        auto environment = makeBackendEnvironment("kvcm_background_recovery");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
        auto parallel = singleRankConfig();
        parallel.tp_size = 2;
        auto handle = makeBackend(environment, parallel, client, broadcast);
        auto backend = std::shared_ptr<KVCMStorageBackend>(std::move(handle.backend));
        ASSERT_TRUE(initSingleRank(*backend, environment));
        KVCacheConfig recovery_config;
        recovery_config.block_tree_frozen_recovery_interval_ms = 20;
        recovery_config.block_tree_frozen_recovery_query_timeout_ms = 1000;
        auto coordinator = makeQueryManager(environment, backend, parallel, broadcast, recovery_config);
        ASSERT_NE(coordinator, nullptr);
        coordinator->completion_cleanup_ = std::make_shared<TransferCompletionCleanup>();
        backend->setCompletionCleanup(coordinator->completion_cleanup_);
        ASSERT_TRUE(coordinator->startFrozenTaskRecovery());
        block_tree_cache_detail::ScopeRollback stop([coordinator] { coordinator->stopFrozenTaskRecovery(); });

        const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://background-0"},
                                                     {"tp1_Fdefault", "pace://background-1"}}};
        EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
        auto matched = match(*backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        const auto original_references = environment.device_pool->refCount(environment.block_id);
        auto request = makeStorageRequest(environment);
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, 2s);
        auto settlements = std::make_shared<std::atomic<size_t>>(0);
        auto done = std::make_shared<std::promise<ErrorInfo>>();
        auto completed = done->get_future();
        backend->read(std::move(request), std::move(matched.match_meta), [done, settlements](ErrorInfo error) {
            if (settlements->fetch_add(1) == 0)
                done->set_value(std::move(error));
        });
        ASSERT_TRUE(gate->waitEntered());
        ASSERT_EQ(completed.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(completed.get().code(), ErrorCode::DEADLINE_EXCEEDED);
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend));
        const auto records = coordinator->frozenTasks();
        ASSERT_EQ(records.size(), 1u);
        const auto record = records.front();
        const auto failure_message = record->failure().status.ToString();
        const auto diagnostics = record->failure().execution_diagnostics;
        ASSERT_NE(diagnostics, nullptr);
        const auto failed_ranks = diagnostics->ranks;
        ASSERT_TRUE(waitUntil([record] { return record->recoverySnapshot().query_count >= 2; }));
        const auto pending = record->recoverySnapshot();
        ASSERT_NE(pending.last_query, nullptr);
        ASSERT_EQ(pending.last_query->ranks.size(), 2u);
        EXPECT_EQ(pending.last_query->ranks[0].state, RegistryState::RUNNING);
        EXPECT_EQ(pending.last_query->ranks[1].state, RegistryState::BUFFER_SAFE);
        EXPECT_EQ(record->snapshot().state, FrozenTask::State::FROZEN);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_references + 1);
        EXPECT_EQ(coordinator->completion_cleanup_->size(), 0u);

        if (stop_before_safe)
            coordinator->stopFrozenTaskRecovery();
        const auto queries_before_safe = record->recoverySnapshot().query_count;
        gate->release();
        ASSERT_TRUE(server0.waitReadCompleted());
        if (stop_before_safe) {
            EXPECT_TRUE(coordinator->queryTransferCompletion(record->execution(), 1000).all_safe);
            EXPECT_FALSE(waitUntil([record] { return record->snapshot().state != FrozenTask::State::FROZEN; }, 100ms));
            EXPECT_EQ(record->recoverySnapshot().query_count, queries_before_safe);
            EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_references + 1);
        } else {
            ASSERT_TRUE(waitUntil([record] { return record->snapshot().state == FrozenTask::State::RELEASED; }));
            ASSERT_TRUE(waitUntil([worker0, worker1, coordinator] {
                return worker0->backend->taskRegistry().size() == 0
                       && worker1->backend->taskRegistry().size() == 0
                       && coordinator->completion_cleanup_->size() == 0;
            }));
            coordinator->stopFrozenTaskRecovery();
            EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_references);
        }
        EXPECT_EQ(settlements->load(), 1u);
        EXPECT_EQ(coordinator->frozenTasks().size(), stop_before_safe ? 1u : 0u);
        EXPECT_EQ(record->failure().status.ToString(), failure_message);
        EXPECT_FALSE(record->failure().buffer_safe);
        EXPECT_EQ(record->failure().execution_diagnostics, diagnostics);
        ASSERT_EQ(diagnostics->ranks.size(), failed_ranks.size());
        for (size_t rank = 0; rank < failed_ranks.size(); ++rank) {
            EXPECT_EQ(diagnostics->ranks[rank].grpc_code, failed_ranks[rank].grpc_code);
            EXPECT_EQ(diagnostics->ranks[rank].buffer_safe, failed_ranks[rank].buffer_safe);
        }
        for (const auto& worker : {worker0, worker1}) {
            EXPECT_EQ(worker->backend->taskRegistry().size(), stop_before_safe ? 1u : 0u);
            EXPECT_EQ(worker->backend->taskRegistry().queryLocal(
                          std::vector<std::string>{std::get<std::string>(record->execution().trace_id)}).front(),
                      stop_before_safe ? RegistryState::BUFFER_SAFE : RegistryState::UNKNOWN);
        }
    }
}

}  // namespace
}  // namespace rtp_llm
