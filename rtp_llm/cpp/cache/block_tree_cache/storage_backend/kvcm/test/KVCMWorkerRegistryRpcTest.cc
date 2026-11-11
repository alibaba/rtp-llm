#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <condition_variable>
#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::BoundedThread;
using RegistryState = TransferTaskRegistry::State;
using namespace std::chrono_literals;

class RegistrySdkGate {
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
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    entered_{false};
    bool                    released_{false};
};

class RegistryWorkerRpcService final: public RpcService::Service {
public:
    RegistryWorkerRpcService(std::shared_ptr<BackendEnvironment> environment,
                             std::shared_ptr<BackendHandle> backend):
        environment_(std::move(environment)), backend_(std::move(backend)) {}

    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        block_tree_cache_detail::ScopeRollback completed([this] { completed_.set_value(); });
        if (!request->has_remote_request()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "missing remote payload");
        }
        (void)backend_->backend->execute(request->remote_request(), *response->mutable_remote_response());
        return grpc::Status::OK;
    }

    std::future_status waitCompleted(std::chrono::milliseconds timeout) {
        return completion_.wait_for(timeout);
    }

private:
    // Keep the backing pools alive if bounded cleanup has to outlive a failed test.
    std::shared_ptr<BackendEnvironment> environment_;
    std::shared_ptr<BackendHandle>      backend_;
    std::promise<void>                 completed_;
    std::future<void>                  completion_{completed_.get_future()};
};

TEST(KVCMWorkerRegistryRpcTest, RecoveryFenceRejectsLateReadBeforeSdkAdmission) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_registry_read_fence");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    request.set_trace_id("read-after-recovery-fence");
    request.set_timeout_ms(30000);
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    request.add_uris("pace://registry-read-fence");
    EXPECT_EQ(backend.backend->queryReadCompletion(request.trace_id()), RegistryState::REJECTED);
    RemoteOperationResponsePB response;
    (void)backend.backend->execute(request, response);
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_TRUE(response.payload_buffer_safe());
    EXPECT_EQ(backend.backend->queryReadCompletion(request.trace_id()), RegistryState::REJECTED);
    EXPECT_EQ(backend.backend->forgetReadCompletion(request.trace_id()), TransferTaskRegistry::ForgetResult::DONE);
    (void)backend.backend->execute(request, response);
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_EQ(backend.backend->queryReadCompletion(request.trace_id()), RegistryState::REJECTED);
}

TEST(KVCMWorkerRegistryRpcTest, RpcDeadlineDoesNotCompleteWorkerEvidenceBeforeSdkReturns) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool sdk_safe : {false, true}) {
        SCOPED_TRACE(sdk_safe);
        auto environment = std::make_shared<BackendEnvironment>(makeBackendEnvironment("kvcm_registry_rpc_deadline"));
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = std::make_shared<BackendHandle>(makeBackend(*environment, singleRankConfig(), client));
        ASSERT_TRUE(initSingleRank(*backend->backend, *environment));
        auto gate = std::make_shared<RegistrySdkGate>();
        EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _))
            .WillOnce(Invoke([gate, sdk_safe](const auto&, const auto&, auto&, const auto&) {
                const bool released = gate->enterAndWait();
                EXPECT_TRUE(released) << "SDK gate was not released within the test deadline";
                return released && sdk_safe;
            }));

        auto service = std::make_shared<RegistryWorkerRpcService>(environment, backend);
        grpc::ServerBuilder builder;
        int port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(service.get());
        std::shared_ptr<grpc::Server> server = builder.BuildAndStart();
        // This guard also runs after an ASSERT. Its shared captures protect every
        // fixture referenced by gRPC while a failed cleanup finishes in the background.
        block_tree_cache_detail::ScopeRollback cleanup([gate, server, service] {
            gate->release();
            if (server) {
                BoundedThread<void> shutdown([server, service] {
                    (void)service;  // Retain the callback owner until Wait returns.
                    server->Shutdown(std::chrono::system_clock::now() + 2s);
                    server->Wait();
                });
                const auto ready = shutdown.waitFor(5s);
                EXPECT_EQ(ready, std::future_status::ready) << "gRPC server did not drain after releasing the SDK";
                if (ready == std::future_status::ready)
                    shutdown.get();
            }
        });
        ASSERT_NE(server, nullptr);
        ASSERT_GT(port, 0);
        auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());
        ASSERT_TRUE(channel->WaitForConnected(std::chrono::system_clock::now() + 5s));

        FunctionRequestPB request;
        auto* read = request.mutable_remote_request();
        read->set_op(REMOTE_OPERATION_READ);
        read->set_trace_id("read-past-rpc-deadline");
        read->set_timeout_ms(30000);
        read->add_group_tags("default");
        read->add_block_ids(environment->block_id);
        read->add_uris("pace://registry-deadline");
        const auto& registry = backend->backend->taskRegistry();
        const auto state = [&] {
            return registry.query(registry.epoch(), std::vector<std::string>{read->trace_id()}).front();
        };

        BoundedThread<grpc::Status> rpc([channel, request] {
            auto stub = RpcService::NewStub(channel);
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + 2s);
            FunctionResponsePB response;
            return stub->ExecuteFunction(&context, request, &response);
        });
        ASSERT_TRUE(gate->waitEntered());
        EXPECT_EQ(state(), RegistryState::RUNNING);
        ASSERT_EQ(rpc.waitFor(5s), std::future_status::ready);
        EXPECT_EQ(rpc.get().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
        EXPECT_EQ(service->waitCompleted(0ms), std::future_status::timeout);
        EXPECT_EQ(state(), RegistryState::RUNNING);

        gate->release();
        ASSERT_EQ(service->waitCompleted(5s), std::future_status::ready);
        EXPECT_EQ(state(), sdk_safe ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
        EXPECT_EQ(registry.size(), 1u);
    }
}

}  // namespace
}  // namespace rtp_llm
