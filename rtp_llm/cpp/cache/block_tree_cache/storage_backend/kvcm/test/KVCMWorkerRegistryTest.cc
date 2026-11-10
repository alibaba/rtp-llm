#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <stdexcept>
#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {
namespace {

using RegistryState = TransferTaskRegistry::State;
using SdkCode = kv_cache_manager::ClientErrorCode;

BackendHandle makeRegistryBackend(const BackendEnvironment& environment,
                                  const std::shared_ptr<MockClientWrapper>& client,
                                  ParallelismConfig parallel = singleRankConfig()) {
    KVCacheConfig config;
    config.kvcm_server_address = "unused-test-address";
    config.kvcm_instance_id_salt = "worker-registry";
    config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
    config.kvcm_read_backend_type = 0;
    config.kvcm_storage_thread_num = 1;
    config.kvcm_asyncwrapper_thread_num = 1;
    config.kvcm_asyncwrapper_queue_size = 1;
    parallel.role_type = RoleType::PREFILL;
    parallel.prefill_cp_config.prefill_cp_size = 1;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_test_model";
    return BackendHandle(std::make_unique<MockQuiescentKVCMBackend>(environment.cache_config,
        config, runtime, parallel, SpeculativeExecutionConfig{}, nullptr, true, nullptr, client));
}

RegistryState registryState(const KVCMStorageBackend& backend, const std::string& trace_id) {
    const auto& registry = backend.taskRegistry();
    return registry.query(registry.epoch(), std::vector<std::string>{trace_id}).front();
}

RemoteOperationRequestPB readRequest(const BackendEnvironment& environment, const std::string& trace_id) {
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    request.set_trace_id(trace_id);
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    request.add_uris("pace://registry");
    return request;
}

template<typename Code>
std::vector<Code> completionCodes() {
    std::vector<Code> codes{Code::ER_OK, Code::ER_SDKREAD_ERROR, Code::ER_SDK_TIMEOUT, Code::ER_INVALID_PARAMS};
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        codes.push_back(Code::ER_SDK_TIMEOUT_BUFFER_SAFE);
        codes.push_back(Code::ER_SDK_ERROR_BUFFER_SAFE);
    }
    return codes;
}

TEST(KVCMWorkerRegistryTest, RpcReadUsesExplicitCompletionProofAndRejectsReplays) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_payload : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_payload ? "1" : "0");
        for (const auto code : completionCodes<SdkCode>()) {
            SCOPED_TRACE(::testing::PrintToString(std::make_pair(protected_payload, code)));
            auto environment = makeBackendEnvironment("kvcm_registry_rpc");
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeRegistryBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            client->load_status = code;
            EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
            auto request = readRequest(environment, "rpc-read");
            RemoteOperationResponsePB response;
            response.set_payload_buffer_safe(true);
            EXPECT_EQ(backend->execute(request, response), code == SdkCode::ER_OK);
            const bool safe = kvcm::sdkCodeProvesBufferSafety(code);
            EXPECT_EQ(response.payload_buffer_safe(), safe);
            EXPECT_EQ(registryState(*backend.backend, request.trace_id()),
                      safe ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
            EXPECT_EQ(backend->taskRegistry().size(), 1u);

            // Reusing a successful response must not turn duplicate or missing
            // identities into fresh completion evidence, even after owner stop.
            response.set_payload_buffer_safe(true);
            response.add_actual_uris("stale");
            EXPECT_FALSE(backend->execute(request, response));
            EXPECT_FALSE(response.payload_buffer_safe());
            EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
            EXPECT_EQ(response.actual_uris_size(), 0);
            EXPECT_EQ(registryState(*backend.backend, request.trace_id()),
                      safe ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
            request.clear_trace_id();
            response.set_payload_buffer_safe(true);
            EXPECT_FALSE(backend->execute(request, response));
            EXPECT_FALSE(response.payload_buffer_safe());
            EXPECT_EQ(registryState(*backend.backend, ""), RegistryState::UNKNOWN);
            EXPECT_EQ(backend->taskRegistry().size(), 1u);
        }
    }
}

TEST(KVCMWorkerRegistryTest, ThrowingSdkCannotPublishSafeFromReusedResponse) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_payload : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_payload ? "1" : "0");
        auto environment = makeBackendEnvironment("kvcm_registry_exception");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeRegistryBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _))
            .WillOnce(Invoke([&](const auto&, const auto&, auto&, const auto&) -> bool {
                EXPECT_EQ(registryState(*backend.backend, "throwing"), RegistryState::RUNNING);
                throw std::runtime_error("injected SDK exception after entering I/O");
            }));
        auto request = readRequest(environment, "throwing");
        RemoteOperationResponsePB response;
        response.set_payload_buffer_safe(true);
        EXPECT_FALSE(backend->execute(request, response));
        EXPECT_FALSE(response.payload_buffer_safe());
        EXPECT_EQ(registryState(*backend.backend, "throwing"), RegistryState::RUNNING);
    }
}

TEST(KVCMWorkerRegistryTest, QueueRejectionAndExpiryAreSafeWhileRunningReplayIsNot) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_registry_queue");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto parallel = singleRankConfig();
    parallel.tp_size = 2;
    parallel.tp_rank = 1;
    auto backend = makeRegistryBackend(environment, client, parallel);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto released = release->get_future().share();
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _))
        .WillOnce(Invoke([&](const auto&, const auto&, auto&, const auto&) {
            EXPECT_EQ(registryState(*backend.backend, "running"), RegistryState::RUNNING);
            entered->set_value();
            EXPECT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            return true;
        }));
    auto running = readRequest(environment, "running");
    running.set_timeout_ms(30000);
    RemoteOperationResponsePB running_response;
    auto running_result = std::async(std::launch::async, [&] { return backend->execute(running, running_response); });
    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    RemoteOperationResponsePB replay;
    replay.set_payload_buffer_safe(true);
    EXPECT_FALSE(backend->execute(running, replay));
    EXPECT_FALSE(replay.payload_buffer_safe());
    EXPECT_EQ(registryState(*backend.backend, "running"), RegistryState::RUNNING);

    auto queued = readRequest(environment, "expired-in-queue");
    queued.set_timeout_ms(100);
    RemoteOperationResponsePB queued_response;
    auto queued_result = std::async(std::launch::async, [&] { return backend->execute(queued, queued_response); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (backend->queueSizes().load != 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(backend->queueSizes().load, 1u);
    EXPECT_EQ(registryState(*backend.backend, "expired-in-queue"), RegistryState::RUNNING);
    auto rejected = readRequest(environment, "queue-rejected");
    RemoteOperationResponsePB rejected_response;
    EXPECT_FALSE(backend->execute(rejected, rejected_response));
    EXPECT_TRUE(rejected_response.payload_buffer_safe());
    EXPECT_EQ(registryState(*backend.backend, "queue-rejected"), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(queued_result.wait_for(std::chrono::milliseconds(150)), std::future_status::timeout);
    release->set_value();
    EXPECT_TRUE(running_result.get());
    EXPECT_FALSE(queued_result.get());
    EXPECT_TRUE(running_response.payload_buffer_safe());
    EXPECT_TRUE(queued_response.payload_buffer_safe());
    EXPECT_EQ(queued_response.transfer_status(), REMOTE_TRANSFER_STATUS_TIMEOUT);
    EXPECT_EQ(registryState(*backend.backend, "running"), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(registryState(*backend.backend, "expired-in-queue"), RegistryState::BUFFER_SAFE);
}

TEST(KVCMWorkerRegistryTest, LocalAsyncQueuePreservesReadOwnersForRejectionAndExpiry) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_registry_local_queue");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeRegistryBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://registry"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).Times(3).WillRepeatedly(Return(std::make_pair(true, locations)));
    std::vector<std::shared_ptr<StorageBackendMatchMeta>> matches;
    for (size_t i = 0; i < 3; ++i) {
        auto observation = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(observation.success);
        ASSERT_NE(observation.match_meta, nullptr);
        matches.push_back(std::move(observation.match_meta));
    }
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto released = release->get_future().share();
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _))
        .WillOnce(Invoke([&](const auto&, const auto&, auto&, const auto&) {
            EXPECT_EQ(registryState(*backend.backend, "block_tree_read_1"), RegistryState::RUNNING);
            entered->set_value();
            EXPECT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            return true;
        }));
    auto start_read = [&](size_t index, std::chrono::milliseconds timeout) {
        auto request = makeStorageRequest(environment);
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, timeout);
        auto done = std::make_shared<std::promise<ErrorInfo>>();
        auto result = done->get_future();
        backend->read(std::move(request), matches[index], [done](ErrorInfo error) {
            done->set_value(std::move(error));
        });
        return result;
    };
    auto running = start_read(0, std::chrono::seconds(30));
    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto queued = start_read(1, std::chrono::milliseconds(500));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((backend->queueSizes().load != 1 || backend->taskRegistry().size() != 2)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(backend->queueSizes().load, 1u);
    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_2"), RegistryState::RUNNING);
    auto rejected = start_read(2, std::chrono::seconds(30));
    EXPECT_FALSE(await(rejected).ok());
    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_1"), RegistryState::RUNNING);
    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_2"), RegistryState::RUNNING);
    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_3"), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(queued.wait_for(std::chrono::milliseconds(550)), std::future_status::timeout);
    release->set_value();
    EXPECT_TRUE(await(running).ok());
    EXPECT_EQ(await(queued).code(), ErrorCode::DEADLINE_EXCEEDED);
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_1"), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_2"), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(backend->taskRegistry().size(), 3u);
    EXPECT_TRUE(backend->frozenTasks().empty());
}

TEST(KVCMWorkerRegistryTest, WarningThresholdNeverCapsReadAdmissionOrDeletesEvidence) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_registry_no_capacity_limit");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
    auto backend = makeRegistryBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    RemoteOperationResponsePB response;
    for (size_t i = 0; i < 4097; ++i) {
        request.set_trace_id("empty-read-" + std::to_string(i));
        ASSERT_TRUE(backend->execute(request, response));
        ASSERT_TRUE(response.payload_buffer_safe());
    }
    EXPECT_EQ(backend->taskRegistry().size(), 4097u);
    EXPECT_EQ(registryState(*backend.backend, "empty-read-0"), RegistryState::BUFFER_SAFE);
    EXPECT_EQ(registryState(*backend.backend, "empty-read-4096"), RegistryState::BUFFER_SAFE);
    request.set_trace_id("empty-read-0");
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_FALSE(response.payload_buffer_safe());
    EXPECT_EQ(backend->taskRegistry().size(), 4097u);
}

TEST(KVCMWorkerRegistryTest, AllTagsMustFinishBeforeTheReadCanBecomeSafe) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool last_tag_succeeds : {false, true}) {
        auto environment = makeMultiGroupBackendEnvironment("kvcm_registry_multitag", 2, 1);
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeRegistryBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        RemoteOperationRequestPB request;
        request.set_op(REMOTE_OPERATION_READ);
        request.set_trace_id("all-tags");
        for (const auto& tag : environment.cache_config.groupTags()) {
            request.add_group_tags(tag);
            request.add_block_ids(environment.block_id);
            request.add_uris("mooncake://" + tag);
        }
        size_t completed = 0;
        EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(3)
            .WillRepeatedly(Invoke([&](const auto&, const auto&, auto&, const auto&) {
                EXPECT_EQ(registryState(*backend.backend, "all-tags"), RegistryState::RUNNING);
                return ++completed != 3 || last_tag_succeeds;
            }));
        RemoteOperationResponsePB response;
        EXPECT_EQ(backend->execute(request, response), last_tag_succeeds);
        EXPECT_EQ(completed, 3u);
        EXPECT_EQ(registryState(*backend.backend, "all-tags"),
                  last_tag_succeeds ? RegistryState::BUFFER_SAFE : RegistryState::RUNNING);
    }
}

TEST(KVCMWorkerRegistryTest, LocalAsyncAndCommonReadsPublishProofBeforeBusinessCallback) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_payload : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_payload ? "1" : "0");
        for (const auto code : completionCodes<SdkCode>()) {
            SCOPED_TRACE(::testing::PrintToString(std::make_pair(protected_payload, code)));
            auto environment = makeBackendEnvironment("kvcm_registry_local_read");
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeRegistryBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://registry"}}};
            EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
            auto observation = match(*backend.backend, makeStorageRequest(environment));
            ASSERT_TRUE(observation.success);
            ASSERT_NE(observation.match_meta, nullptr);
            client->load_status = code;
            const auto expected = kvcm::sdkCodeProvesBufferSafety(code) ? RegistryState::BUFFER_SAFE :
                                                                       RegistryState::RUNNING;
            auto done = std::make_shared<std::promise<ErrorInfo>>();
            auto result = done->get_future();
            backend->read(makeStorageRequest(environment), std::move(observation.match_meta),
                [&, done](ErrorInfo error) {
                    EXPECT_EQ(registryState(*backend.backend, "block_tree_read_1"), expected);
                    done->set_value(std::move(error));
                });
            EXPECT_EQ(await(result).ok(), code == SdkCode::ER_OK);
            ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
            EXPECT_EQ(registryState(*backend.backend, "block_tree_read_1"), expected);
            EXPECT_EQ(backend->taskRegistry().size(), 1u);
        }
    }
}

TEST(KVCMWorkerRegistryTest, ShutdownKeepsReplayRejectionDistinctFromUnsubmittedRead) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_registry_shutdown");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeRegistryBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    client->load_status = SdkCode::ER_OK;
    auto request = readRequest(environment, "completed");
    RemoteOperationResponsePB response;
    ASSERT_TRUE(backend->execute(request, response));
    backend->shutdown();
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_FALSE(response.payload_buffer_safe());
    EXPECT_EQ(registryState(*backend.backend, "completed"), RegistryState::BUFFER_SAFE);
    request.set_trace_id("after-shutdown");
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_TRUE(response.payload_buffer_safe());
    EXPECT_EQ(registryState(*backend.backend, "after-shutdown"), RegistryState::BUFFER_SAFE);
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_FALSE(response.payload_buffer_safe());
    request.clear_trace_id();
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_FALSE(response.payload_buffer_safe());
    request.set_op(REMOTE_OPERATION_WRITE);
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_EQ(backend->taskRegistry().size(), 2u);
}

TEST(KVCMWorkerRegistryTest, WriteAndMetadataDoNotParticipateInReadAdmission) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_registry_write_metadata");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeRegistryBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    client->load_status = SdkCode::ER_OK;
    client->save_status = SdkCode::ER_OK;
    auto request = readRequest(environment, "read-identity");
    RemoteOperationResponsePB response;
    ASSERT_TRUE(backend->execute(request, response));
    request.set_op(REMOTE_OPERATION_WRITE);
    for (const auto& trace_id : {std::string("read-identity"), std::string{}}) {
        request.set_trace_id(trace_id);
        EXPECT_TRUE(backend->execute(request, response));
        EXPECT_TRUE(backend->execute(request, response));
        EXPECT_EQ(backend->taskRegistry().size(), 1u);
    }
    request.Clear();
    request.set_op(REMOTE_OPERATION_REMOVE_CACHE);
    request.mutable_metadata()->set_query_type(1);
    EXPECT_CALL(*client, removeCache(_, _, _, _, _)).Times(2).WillRepeatedly(Return(true));
    EXPECT_TRUE(backend->execute(request, response));
    EXPECT_TRUE(backend->execute(request, response));
    EXPECT_EQ(backend->taskRegistry().size(), 1u);
    EXPECT_EQ(registryState(*backend.backend, ""), RegistryState::UNKNOWN);
}

}  // namespace
}  // namespace rtp_llm
