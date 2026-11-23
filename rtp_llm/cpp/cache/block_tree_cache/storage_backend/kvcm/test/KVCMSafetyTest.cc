#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

ErrorInfo
readStatus(KVCMStorageBackend& backend, StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta) {
    auto promise = std::make_shared<std::promise<ErrorInfo>>();
    auto future  = promise->get_future();
    backend.read(std::move(request), std::move(match_meta), [promise](ErrorInfo error) {
        promise->set_value(std::move(error));
    });
    return await(future);
}

kv_cache_manager::Locations singleLocation() {
    return {{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
}

kv_cache_manager::Locations tp2Locations() {
    return {{kv_cache_manager::LocationSpecUnit{"tp1_Fdefault", "read_rank_1"},
             kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_rank_0"}}};
}

kv_cache_manager::WriteLocation singleWriteLocation() {
    kv_cache_manager::WriteLocation result;
    result.write_session_id = "safety_session";
    result.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    result.locations        = singleLocation();
    return result;
}

TEST(KVCMSafetyTest, LocalSdkResultPermanentlyProtectsUnknownCompletion) {
    for (const auto code :
         {kv_cache_manager::ER_OK, kv_cache_manager::ER_SDKREAD_ERROR, kv_cache_manager::ER_SDK_TIMEOUT}) {
        auto environment     = makeBackendEnvironment("kvcm_safety_local_status");
        auto wrapper         = std::make_shared<MockClientWrapper>();
        wrapper->load_status = code;
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, singleRankConfig(), wrapper);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, singleLocation())));
        auto matched = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        ASSERT_NE(matched.match_meta, nullptr);
        EXPECT_CALL(*wrapper, loadKvCachesForTagWithStatus("default", _, _, _)).WillOnce(Return(code));
        auto                               request = makeStorageRequest(environment);
        std::weak_ptr<const CacheKeysType> keys    = request.keys;
        const auto error = readStatus(*backend.backend, std::move(request), std::move(matched.match_meta));
        EXPECT_EQ(error.code(),
                  code == kv_cache_manager::ER_OK          ? ErrorCode::NONE_ERROR :
                  code == kv_cache_manager::ER_SDK_TIMEOUT ? ErrorCode::DEADLINE_EXCEEDED :
                                                             ErrorCode::EXECUTION_EXCEPTION);
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        const bool unknown = code != kv_cache_manager::ER_OK;
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), unknown);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), unknown ? 2u : 1u);
        backend->shutdown();
        EXPECT_TRUE(keys.expired());
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), unknown);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), unknown ? 2u : 1u);
    }
}

TEST(KVCMSafetyTest, TransferStatusUsesFeatureWireNumbersAndRoundTrips) {
    const auto* field = RemoteOperationResponsePB::descriptor()->FindFieldByName("transfer_status");
    ASSERT_NE(field, nullptr);
    EXPECT_EQ(field->number(), 3);
    for (const auto status : {REMOTE_TRANSFER_STATUS_UNSPECIFIED,
                              REMOTE_TRANSFER_STATUS_SUCCESS,
                              REMOTE_TRANSFER_STATUS_FAILED,
                              REMOTE_TRANSFER_STATUS_TIMEOUT}) {
        RemoteOperationResponsePB response;
        response.set_transfer_status(status);
        RemoteOperationResponsePB restored;
        ASSERT_TRUE(restored.ParseFromString(response.SerializeAsString()));
        EXPECT_EQ(restored.transfer_status(), status);
    }
    EXPECT_EQ(REMOTE_TRANSFER_STATUS_UNSPECIFIED, 0);
    EXPECT_EQ(REMOTE_TRANSFER_STATUS_SUCCESS, 1);
    EXPECT_EQ(REMOTE_TRANSFER_STATUS_FAILED, 2);
    EXPECT_EQ(REMOTE_TRANSFER_STATUS_TIMEOUT, 3);
}

TEST(KVCMSafetyTest, WorkerPublishesSdkStatusAndRejectsInvalidRequestBeforeIo) {
    auto environment = makeBackendEnvironment("kvcm_safety_worker_status");
    auto wrapper     = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    for (const auto operation : {REMOTE_OPERATION_READ, REMOTE_OPERATION_WRITE}) {
        for (const auto status :
             {REMOTE_TRANSFER_STATUS_SUCCESS, REMOTE_TRANSFER_STATUS_FAILED, REMOTE_TRANSFER_STATUS_TIMEOUT}) {
            RemoteOperationRequestPB request;
            request.set_op(operation);
            identifyReadRequest(request);
            request.add_group_tags("default");
            request.add_block_ids(environment.block_id);
            request.add_uris("uri");
            const auto code = status == REMOTE_TRANSFER_STATUS_SUCCESS ? kv_cache_manager::ER_OK :
                              status == REMOTE_TRANSFER_STATUS_TIMEOUT ? kv_cache_manager::ER_SDK_TIMEOUT :
                                                                         kv_cache_manager::ER_INVALID_PARAMS;
            if (operation == REMOTE_OPERATION_READ) {
                EXPECT_CALL(*wrapper, loadKvCachesForTagWithStatus("default", _, _, _)).WillOnce(Return(code));
            } else {
                EXPECT_CALL(*wrapper, saveKvCachesForTagWithStatus("default", _, _, _))
                    .WillOnce(Return(std::make_pair(code, kv_cache_manager::UriStrVec{})));
            }
            RemoteOperationResponsePB response;
            EXPECT_EQ(backend->execute(request, response), status == REMOTE_TRANSFER_STATUS_SUCCESS);
            EXPECT_EQ(response.transfer_status(), status);
        }
    }
    RemoteOperationRequestPB invalid;
    invalid.set_op(REMOTE_OPERATION_READ);
    identifyReadRequest(invalid);
    invalid.add_group_tags("not_registered");
    invalid.add_block_ids(environment.block_id);
    invalid.add_uris("uri");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(invalid, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    invalid.clear_group_tags();
    EXPECT_FALSE(backend->execute(invalid, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
}

struct SafetyBroadcastGroup {
    std::vector<std::shared_ptr<KVCMBroadcastState>>     states;
    std::vector<std::unique_ptr<KVCMBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager>                    manager;

    bool start() {
        std::vector<std::string> addresses;
        for (size_t rank = 0; rank < 2; ++rank) {
            auto state  = std::make_shared<KVCMBroadcastState>();
            auto server = std::make_unique<KVCMBroadcastRpcServer>(rank, state);
            if (!server->start()) {
                return false;
            }
            addresses.push_back(server->address());
            states.push_back(std::move(state));
            servers.push_back(std::move(server));
        }
        manager = std::make_shared<BroadcastManager>(addresses);
        return manager->init();
    }
};

ParallelismConfig safetyTp2() {
    auto config    = singleRankConfig();
    config.tp_size = 2;
    return config;
}

TEST(KVCMSafetyTest, TpApplicationTimeoutWinsOverFailedForEitherRankOrder) {
    for (size_t timeout_rank : {0u, 1u}) {
        SafetyBroadcastGroup group;
        ASSERT_TRUE(group.start());
        group.states[timeout_rank]->transfer_status     = REMOTE_TRANSFER_STATUS_TIMEOUT;
        group.states[timeout_rank]->payload_buffer_safe = false;
        group.states[1 - timeout_rank]->transfer_status = REMOTE_TRANSFER_STATUS_FAILED;
        auto environment                                = makeBackendEnvironment("kvcm_safety_tp_timeout");
        auto wrapper                                    = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, safetyTp2(), wrapper, group.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, tp2Locations())));
        auto matched = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        EXPECT_EQ(readStatus(*backend.backend, makeStorageRequest(environment), matched.match_meta).code(),
                  ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
    }
}

TEST(KVCMSafetyTest, TpFailedAndLegacyResponsesRequireExplicitCompletionProof) {
    for (const auto status :
         {REMOTE_TRANSFER_STATUS_FAILED, REMOTE_TRANSFER_STATUS_UNSPECIFIED, REMOTE_TRANSFER_STATUS_SUCCESS}) {
        SafetyBroadcastGroup group;
        ASSERT_TRUE(group.start());
        group.states[0]->transfer_status = status;
        auto environment                 = makeBackendEnvironment("kvcm_safety_tp_failed_legacy");
        auto wrapper                     = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, safetyTp2(), wrapper, group.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, tp2Locations())));
        auto matched = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        const auto error = readStatus(*backend.backend, makeStorageRequest(environment), matched.match_meta);
        EXPECT_EQ(error.ok(), status == REMOTE_TRANSFER_STATUS_SUCCESS);
        const bool unknown = status == REMOTE_TRANSFER_STATUS_UNSPECIFIED;
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), unknown);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), unknown ? 2u : 1u);
    }
}

TEST(KVCMSafetyTest, TpMissingInvalidAndRpcErrorResponsesFreezeBlocks) {
    for (int failure = 0; failure < 3; ++failure) {
        SafetyBroadcastGroup group;
        ASSERT_TRUE(group.start());
        if (failure == 0) {
            group.states[1]->omit_response = true;
        } else if (failure == 1) {
            group.states[1]->transfer_status = static_cast<RemoteTransferStatusPB>(99);
        } else {
            group.states[1]->grpc_code = grpc::StatusCode::INTERNAL;
        }
        auto environment = makeBackendEnvironment("kvcm_safety_tp_invalid");
        auto wrapper     = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, safetyTp2(), wrapper, group.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, tp2Locations())));
        auto matched = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        EXPECT_EQ(readStatus(*backend.backend, makeStorageRequest(environment), matched.match_meta).code(),
                  ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
    }
}

TEST(KVCMSafetyTest, BroadcastWaitClassifiesDeadlineAndRpcFailureBeforeFreezing) {
    for (const auto status : {grpc::StatusCode::DEADLINE_EXCEEDED, grpc::StatusCode::UNAVAILABLE}) {
        SCOPED_TRACE(static_cast<int>(status));
        const bool           timeout = status == grpc::StatusCode::DEADLINE_EXCEEDED;
        SafetyBroadcastGroup group;
        ASSERT_TRUE(group.start());
        group.states[1]->grpc_code = status;
        auto environment           = makeBackendEnvironment("kvcm_safety_wait_failure");
        auto wrapper               = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, safetyTp2(), wrapper, group.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, tp2Locations())));
        auto matched = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);

        try {
            auto request = makeStorageRequest(environment);
            request.transfer_task.emplace(std::vector<TransferDescriptor>{}, std::chrono::seconds(1));
            backend->readImpl(request, matched.match_meta);
            FAIL() << "expected broadcast failure";
        } catch (const StorageOperationTimeout& exception) {
            EXPECT_TRUE(timeout);
            EXPECT_EQ(exception.reason(), StorageQuarantineReason::RPC_TIMEOUT);
        } catch (const StorageOperationCompletionUnknown& exception) {
            EXPECT_FALSE(timeout);
            EXPECT_EQ(exception.reason(), StorageQuarantineReason::RPC_ERROR);
        }
        ASSERT_EQ(backend.reclaimer->frozen_.size(), 1u);
        ASSERT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
        const auto error = readStatus(*backend.backend, makeStorageRequest(environment), matched.match_meta);
        EXPECT_EQ(error.code(), timeout ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
        // Direct readImpl and asynchronous read are distinct executions and
        // retain separate holds even when they target the same physical block.
        EXPECT_EQ(backend.reclaimer->frozen_.size(), 2u);
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 2u);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 2u);
        EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    }
}

TEST(KVCMSafetyTest, RejectedBroadcastHasNoSubmittedWorkersAndDoesNotFreeze) {
    auto environment = makeBackendEnvironment("kvcm_safety_tp_not_submitted");
    auto wrapper     = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*wrapper, shutdown()).Times(1);
    auto manager = std::make_shared<BroadcastManager>(std::vector<std::string>{"unused", "unused"});
    auto backend = makeBackend(environment, safetyTp2(), wrapper, manager);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, tp2Locations())));
    auto matched = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(matched.success);
    EXPECT_EQ(readStatus(*backend.backend, makeStorageRequest(environment), matched.match_meta).code(),
              ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_TRUE(backend.reclaimer->frozen_.empty());
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 0u);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 1u);
}

TEST(KVCMSafetyTest, TimeoutKeepsReadDestinationFrozenAndReleasesAbortedWriteSource) {
    for (const bool is_write : {false, true}) {
        SafetyBroadcastGroup group;
        ASSERT_TRUE(group.start());
        auto               entered        = std::make_shared<std::promise<void>>();
        auto               entered_future = entered->get_future();
        std::promise<void> release;
        auto               released    = release.get_future().share();
        auto               environment = makeBackendEnvironment("kvcm_safety_late_worker");
        const auto         buffers     = environmentBuffers(environment, 0, "default", environment.block_id);
        ASSERT_FALSE(buffers.empty());
        void*          device_address = buffers.front().addr;
        const uint32_t initial_value  = 0x29;
        const uint32_t late_value     = 0x71;
        const uint32_t reused_value   = 0x9b;
        ASSERT_EQ(cudaMemcpy(device_address, &initial_value, sizeof(initial_value), cudaMemcpyHostToDevice),
                  cudaSuccess);
        group.states[1]->worker_action = [entered, released, device_address, is_write, late_value] {
            entered->set_value();
            EXPECT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
            if (!is_write) {
                EXPECT_EQ(cudaMemcpy(device_address, &late_value, sizeof(late_value), cudaMemcpyHostToDevice),
                          cudaSuccess);
            }
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        };
        auto wrapper = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, safetyTp2(), wrapper, group.manager, 100);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        std::future<ErrorInfo> read_future;
        if (is_write) {
            auto write_location      = singleWriteLocation();
            write_location.locations = tp2Locations();
            EXPECT_CALL(*wrapper, getWriteLocation(_, _, _, _, _, _, 0))
                .WillOnce(Return(std::make_pair(true, write_location)));
            EXPECT_CALL(*wrapper,
                        finishWrite(_, _, "safety_session",
                                    ::testing::VariantWith<kv_cache_manager::BlockMaskOffset>(::testing::Eq(0u)),
                                    ::testing::IsEmpty()))
                .WillOnce(Return(true));
            EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment))));
        } else {
            EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, tp2Locations())));
            auto matched = match(*backend.backend, makeStorageRequest(environment));
            ASSERT_TRUE(matched.success);
            auto promise = std::make_shared<std::promise<ErrorInfo>>();
            read_future  = promise->get_future();
            backend->read(
                makeStorageRequest(environment), matched.match_meta, [promise, &environment](ErrorInfo error) {
                    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
                    promise->set_value(std::move(error));
                });
        }
        EXPECT_EQ(entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        const auto late_requests = snapshotRequests(group.states[1]);
        EXPECT_EQ(late_requests.size(), 1u);
        if (late_requests.size() == 1) {
            EXPECT_EQ(late_requests.front().op(), is_write ? REMOTE_OPERATION_WRITE : REMOTE_OPERATION_READ);
            EXPECT_EQ(late_requests.front().group_tags_size(), 1);
            EXPECT_EQ(late_requests.front().block_ids_size(), 1);
            EXPECT_EQ(late_requests.front().uris_size(), 1);
            if (late_requests.front().block_ids_size() == 1) {
                EXPECT_EQ(late_requests.front().block_ids(0), environment.block_id);
            }
        }
        EXPECT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        if (!is_write) {
            EXPECT_EQ(await(read_future).code(), ErrorCode::DEADLINE_EXCEEDED);
        }
        // Drop the caller's hold while the remote worker is still running.
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), is_write ? 1u : 2u);
        environment.device_pool->decRef(environment.block_id);
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), !is_write);
        EXPECT_EQ(environment.device_pool->isAllocated(environment.block_id), !is_write);
        ScopedReferencedBlocks available(environment.device_pool, environment.device_pool->availableBlocksNum());
        EXPECT_EQ(std::count(available.get().begin(), available.get().end(), environment.block_id), is_write ? 1 : 0);
        if (is_write) {
            EXPECT_EQ(cudaMemcpy(device_address, &reused_value, sizeof(reused_value), cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
        EXPECT_FALSE(environment.device_pool->malloc().has_value());
        release.set_value();
        group.servers.clear();
        uint32_t final_value = 0;
        EXPECT_EQ(cudaMemcpy(&final_value, device_address, sizeof(final_value), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(final_value, is_write ? reused_value : late_value);
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), !is_write);
        // BackendEnvironment owns a normal reference for its destructor.
        environment.device_pool->incRef(environment.block_id);
    }
}

TEST(KVCMSafetyTest, UnknownWriteReleasesSourceWhenSessionAbortFails) {
    for (const bool compensation_throws : {false, true}) {
        auto environment = makeBackendEnvironment("kvcm_safety_write_cleanup");
        auto wrapper     = std::make_shared<MockClientWrapper>();
        // Mark the SDK call as submitted so its timeout has unknown completion.
        wrapper->save_status = kv_cache_manager::ER_SDK_TIMEOUT;
        EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*wrapper, shutdown()).Times(1);
        auto backend = makeBackend(environment, singleRankConfig(), wrapper);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_CALL(*wrapper, getWriteLocation(_, _, _, _, _, _, 0))
            .WillOnce(Return(std::make_pair(true, singleWriteLocation())));
        EXPECT_CALL(*wrapper, saveKvCachesForTagWithStatus("default", _, _, _))
            .WillOnce(Return(std::make_pair(kv_cache_manager::ER_SDK_TIMEOUT, kv_cache_manager::UriStrVec{})));
        EXPECT_CALL(*wrapper, finishWrite(_, _, "safety_session", _, _))
            .WillOnce(Invoke([compensation_throws](const auto&,
                                                   const auto&,
                                                   const auto&,
                                                   const kv_cache_manager::BlockMask& mask,
                                                   const auto&                        locations) {
                EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 0u);
                EXPECT_TRUE(locations.empty());
                if (compensation_throws) {
                    throw std::runtime_error("cleanup failure");
                }
                return false;
            }));
        EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment))));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 1u);
        backend->shutdown();
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 1u);
    }
}

TEST(KVCMSafetyTest, LocalSdkExceptionPermanentlyProtectsDestination) {
    auto environment = makeBackendEnvironment("kvcm_safety_local_exception");
    auto wrapper     = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    EXPECT_CALL(*wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, singleLocation())));
    auto matched = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(matched.success);
    EXPECT_CALL(*wrapper, loadKvCachesForTagWithStatus("default", _, _, _))
        .WillOnce(Invoke([](const auto&, const auto&, auto&, const auto&) -> kv_cache_manager::ClientErrorCode {
            throw std::runtime_error("direct SDK exception");
        }));
    EXPECT_EQ(readStatus(*backend.backend, makeStorageRequest(environment), matched.match_meta).code(),
              ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 2u);
    backend->shutdown();
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 2u);
}

TEST(KVCMSafetyTest, AttemptedNormalFinishWriteIsNeverCompensatedAfterThrow) {
    auto environment = makeBackendEnvironment("kvcm_safety_finish_throw");
    auto wrapper     = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    EXPECT_CALL(*wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, singleWriteLocation())));
    EXPECT_CALL(*wrapper, saveKvCachesForTagWithStatus("default", _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ER_OK, kv_cache_manager::UriStrVec{})));
    EXPECT_CALL(*wrapper, finishWrite(_, _, "safety_session", _, _))
        .WillOnce(Invoke([](const auto&, const auto&, const auto&, const auto&, const auto&) -> bool {
            throw std::runtime_error("normal FinishWrite failed");
        }));
    EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 1u);
}

}  // namespace
}  // namespace rtp_llm
