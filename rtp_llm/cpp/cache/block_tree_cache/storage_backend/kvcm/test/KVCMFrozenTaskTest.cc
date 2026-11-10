#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <set>
#include <stdexcept>
#include <tuple>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"

namespace rtp_llm {
namespace {

using FrozenState = FrozenTask::State;
using SdkCode = kv_cache_manager::ClientErrorCode;

std::shared_ptr<FrozenTask> onlyFrozenTask(const KVCMStorageBackend& backend) {
    const auto tasks = backend.frozenTasks();
    EXPECT_EQ(tasks.size(), 1u);
    return tasks.size() == 1 ? tasks.front() : nullptr;
}

std::shared_ptr<StorageBackendMatchMeta> matchOne(KVCMStorageBackend& backend,
                                               const BackendEnvironment& environment,
                                               MockClientWrapper& client) {
    const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://read-destination"}}};
    EXPECT_CALL(client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto matched = match(backend, makeStorageRequest(environment));
    EXPECT_TRUE(matched.success);
    EXPECT_NE(matched.match_meta, nullptr);
    return std::move(matched.match_meta);
}

template<typename Code>
std::vector<Code> safeCompletionCodes() {
    std::vector<Code> codes{Code::ER_OK};
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        codes.push_back(Code::ER_SDK_TIMEOUT_BUFFER_SAFE);
        codes.push_back(Code::ER_SDK_ERROR_BUFFER_SAFE);
    }
    return codes;
}

TEST(KVCMFrozenTaskTest, LocalUnknownIsRetainedBeforeCallbackAndKeepsOriginalTrace) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const int outcome : {0, 1, 2}) {
        SCOPED_TRACE(outcome);
        auto environment = makeBackendEnvironment("kvcm_frozen_local_unknown");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeBackend(environment, singleRankConfig(), client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        auto meta = matchOne(*backend.backend, environment, *client);
        ASSERT_NE(meta, nullptr);
        if (outcome == 2) {
            EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _))
                .WillOnce(Invoke([](const auto&, const auto&, auto&, const auto&) -> bool {
                    throw std::runtime_error("unknown completion after entering SDK");
                }));
        } else {
            client->load_status = outcome == 0 ? SdkCode::ER_SDKREAD_ERROR : SdkCode::ER_SDK_TIMEOUT;
        }
        const auto before = environment.device_pool->refCount(environment.block_id);
        auto done = std::make_shared<std::promise<ErrorInfo>>();
        auto completed = done->get_future();
        auto request = makeStorageRequest(environment);
        request.request_id = 912;
        backend->read(std::move(request), std::move(meta), [&, done](ErrorInfo error) {
            const auto record = onlyFrozenTask(*backend.backend);
            EXPECT_NE(record, nullptr);
            if (record) {
                EXPECT_EQ(record->snapshot().state, FrozenState::FROZEN);
                EXPECT_EQ(record->snapshot().target_block_count, 1u);
                EXPECT_EQ(std::get<std::string>(record->execution().trace_id), "block_tree_read_1");
            }
            // Business pins have already been settled by the callback. The
            // independently owned frozen reference still protects the target.
            EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before + 1);
            done->set_value(std::move(error));
            if (outcome == 2) {
                throw std::runtime_error("injected business callback failure after observing frozen ownership");
            }
        });
        EXPECT_FALSE(await(completed).ok());
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        const auto record = onlyFrozenTask(*backend.backend);
        ASSERT_NE(record, nullptr);
        const auto& execution = record->execution();
        EXPECT_EQ(execution.request_id, std::optional<int64_t>{912});
        EXPECT_FALSE(execution.remote);
        EXPECT_EQ(execution.expected_rank_count, 1u);
        ASSERT_EQ(execution.descriptors.size(), 1u);
        const auto& descriptor = execution.descriptors.front();
        EXPECT_EQ(descriptor.rank, std::optional<size_t>{0});
        EXPECT_EQ(descriptor.group_tag, "default");
        EXPECT_EQ(descriptor.source_uri, "pace://read-destination");
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        EXPECT_TRUE(descriptor.sources.empty());
        ASSERT_EQ(descriptor.targets.size(), 1u);
        EXPECT_EQ(descriptor.targets.front().pool.get(), environment.device_pool.get());
        EXPECT_EQ(descriptor.targets.front().block, environment.block_id);
        EXPECT_EQ(record->failure().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_FALSE(record->failure().buffer_safe);
        const auto diagnostic = record->failure().execution_diagnostics;
        ASSERT_NE(diagnostic, nullptr);
        EXPECT_EQ(std::get<std::string>(diagnostic->trace_id), "block_tree_read_1");
        EXPECT_FALSE(diagnostic->failure_reason.empty());
        ASSERT_EQ(diagnostic->ranks.size(), 1u);
        const auto& rank = diagnostic->ranks.front();
        EXPECT_EQ(rank.rank, 0u);
        EXPECT_FALSE(rank.remote);
        EXPECT_FALSE(rank.cq_received);
        EXPECT_FALSE(rank.grpc_code.has_value());
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
    }
}

TEST(KVCMFrozenTaskTest, ExplicitSafeCompletionDoesNotCreateFrozenOwnership) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const auto code : safeCompletionCodes<SdkCode>()) {
        SCOPED_TRACE(static_cast<int>(code));
        auto environment = makeBackendEnvironment("kvcm_frozen_safe_completion");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeBackend(environment, singleRankConfig(), client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        auto meta = matchOne(*backend.backend, environment, *client);
        ASSERT_NE(meta, nullptr);
        const auto before = environment.device_pool->refCount(environment.block_id);
        client->load_status = code;
        EXPECT_EQ(readResult(*backend.backend, makeStorageRequest(environment), std::move(meta)).ok(),
                  code == SdkCode::ER_OK);
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_TRUE(backend->frozenTasks().empty());
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before);
    }
}

TEST(KVCMFrozenTaskTest, RpcFailureRetainsOriginalParticipantsAndCompletedRankEvidence) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const auto code : {grpc::StatusCode::DEADLINE_EXCEEDED, grpc::StatusCode::UNAVAILABLE}) {
        SCOPED_TRACE(static_cast<int>(code));
        auto environment = makeBackendEnvironment("kvcm_frozen_rpc_failure");
        auto client = std::make_shared<MockClientWrapper>();
        auto healthy = std::make_shared<KVCMBroadcastState>();
        auto failed = std::make_shared<KVCMBroadcastState>();
        failed->fail = true;
        failed->failure_code = code;
        KVCMBroadcastRpcServer server0(0, healthy);
        KVCMBroadcastRpcServer server1(1, failed);
        ASSERT_TRUE(server0.start());
        ASSERT_TRUE(server1.start());
        const std::vector<std::string> addresses{server0.address(), server1.address()};
        auto broadcast = std::make_shared<BroadcastManager>(addresses);
        ASSERT_TRUE(broadcast->init());
        auto parallel = singleRankConfig();
        parallel.tp_size = 2;
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeBackend(environment, parallel, client, broadcast);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "rank-0-uri"},
                                                      {"tp1_Fdefault", "rank-1-uri"}}};
        EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
        auto matched = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(matched.success);
        ASSERT_NE(matched.match_meta, nullptr);
        auto done = std::make_shared<std::promise<ErrorInfo>>();
        auto completed = done->get_future();
        backend->read(makeStorageRequest(environment), std::move(matched.match_meta), [&, done](ErrorInfo error) {
            EXPECT_EQ(backend->frozenTasks().size(), 1u);
            EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
            done->set_value(std::move(error));
        });
        EXPECT_EQ(await(completed).code(), code == grpc::StatusCode::DEADLINE_EXCEEDED ?
                                             ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION);
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        const auto record = onlyFrozenTask(*backend.backend);
        ASSERT_NE(record, nullptr);
        const auto& execution = record->execution();
        EXPECT_TRUE(execution.remote);
        EXPECT_EQ(execution.expected_rank_count, 2u);
        ASSERT_EQ(execution.expected_peers.size(), 2u);
        ASSERT_EQ(execution.descriptors.size(), 2u);
        EXPECT_EQ(record->snapshot().target_block_count, 1u);
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
        const auto trace = std::get<std::string>(execution.trace_id);
        EXPECT_EQ(trace, "block_tree_read_1");
        const auto diagnostic = record->failure().execution_diagnostics;
        ASSERT_NE(diagnostic, nullptr);
        EXPECT_EQ(std::get<std::string>(diagnostic->trace_id), trace);
        EXPECT_EQ(diagnostic->failure_reason, code == grpc::StatusCode::DEADLINE_EXCEEDED ?
                                                  "rpc_timeout" : "rpc_error");
        ASSERT_EQ(diagnostic->ranks.size(), 2u);
        for (size_t rank = 0; rank < 2; ++rank) {
            const auto requests = snapshotRequests(rank == 0 ? healthy : failed);
            ASSERT_EQ(requests.size(), 1u);
            EXPECT_EQ(requests.front().trace_id(), trace);
            EXPECT_EQ(execution.expected_peers[rank].rank, rank);
            EXPECT_EQ(execution.expected_peers[rank].address, addresses[rank]);
            const auto& snapshot = diagnostic->ranks[rank];
            EXPECT_EQ(snapshot.rank, rank);
            EXPECT_EQ(snapshot.server_address, addresses[rank]);
            EXPECT_TRUE(snapshot.remote);
            EXPECT_EQ(snapshot.dispatch_state, TransferRpcDispatchState::SUBMITTED);
            EXPECT_TRUE(snapshot.cq_received);
            EXPECT_TRUE(snapshot.cq_ok);
            EXPECT_EQ(snapshot.grpc_code, std::optional<int>{rank == 0 ? grpc::StatusCode::OK : code});
            EXPECT_EQ(snapshot.response_received, rank == 0);
            EXPECT_EQ(snapshot.response_valid, rank == 0);
            EXPECT_EQ(snapshot.buffer_safe, rank == 0 ? std::optional<bool>{true} : std::nullopt);
            const auto& descriptor = execution.descriptors[rank];
            EXPECT_EQ(descriptor.rank, std::optional<size_t>{rank});
            EXPECT_EQ(descriptor.group_tag, "default");
            EXPECT_EQ(descriptor.source_uri, "rank-" + std::to_string(rank) + "-uri");
        }
        EXPECT_EQ(diagnostic->ranks[0].worker_transfer_status,
                  std::optional<int>{REMOTE_TRANSFER_STATUS_SUCCESS});
        // The fake worker does not send sdk_failed; absence stays explicit.
        EXPECT_FALSE(diagnostic->ranks[0].sdk_failed.has_value());
        EXPECT_FALSE(diagnostic->ranks[1].worker_transfer_status.has_value());
        EXPECT_FALSE(diagnostic->ranks[1].sdk_failed.has_value());
    }
}

TEST(KVCMFrozenTaskTest, TargetsExcludeLocalPrefixFilteredLinearStateAndUnsentTail) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeMultiGroupBackendEnvironment("kvcm_frozen_target_boundary", 1, 1);
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    ScopedReferencedBlocks allocation(environment, 4);
    const auto& blocks = allocation.get();
    const kv_cache_manager::Locations locations{
        {{"tp0_Ffull0", "full-102"}, {"tp0_Llinear0", "linear-102-filtered"}},
        {{"tp0_Ffull0", "full-103"}, {"tp0_Llinear0", "linear-103"}},
        {{"tp0_Ffull0", "full-104-unsent"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto request = makeGroupedStorageRequest(environment, {101, 102, 103, 104}, 1, blocks);
    auto matched = match(*backend.backend, request);
    ASSERT_TRUE(matched.success);
    ASSERT_EQ(matched.matched_blocks_num, 3u);
    ASSERT_NE(matched.match_meta, nullptr);
    request.keys = std::make_shared<const CacheKeysType>(CacheKeysType{101, 102, 103});
    request.handles.resize(3);
    // Keep the local-prefix handles present: they are business inputs but were
    // never emitted as remote destinations. The earlier LINEAR handle is also
    // present, although the group policy selected only the newest state.
    client->load_status = SdkCode::ER_SDK_TIMEOUT;
    EXPECT_EQ(readResult(*backend.backend, std::move(request), std::move(matched.match_meta)).code(),
              ErrorCode::DEADLINE_EXCEEDED);
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    const auto record = onlyFrozenTask(*backend.backend);
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->snapshot().target_block_count, 3u);
    std::set<std::tuple<std::string, BlockIdxType, std::string>> targets;
    for (const auto& descriptor : record->execution().descriptors) {
        EXPECT_EQ(descriptor.rank, std::optional<size_t>{0});
        ASSERT_EQ(descriptor.targets.size(), 1u);
        const auto& target = descriptor.targets.front();
        EXPECT_EQ(target.pool.get(), environment.pools_by_tag.at(descriptor.group_tag).get());
        targets.emplace(descriptor.group_tag, target.block, descriptor.source_uri);
    }
    EXPECT_EQ(targets, (std::set<std::tuple<std::string, BlockIdxType, std::string>>{
        {"full0", blocks[1], "full-102"}, {"full0", blocks[2], "full-103"},
        {"linear0", blocks[2], "linear-103"}}));
    for (const auto& tag : environment.cache_config.groupTags()) {
        const auto& pool = environment.pools_by_tag.at(tag);
        for (size_t i = 0; i < blocks.size(); ++i) {
            const bool frozen = i == 2 || (tag == "full0" && i == 1);
            EXPECT_EQ(pool->isFrozen(blocks[i]), frozen) << tag << " index=" << i;
            EXPECT_EQ(pool->treeRefCount(blocks[i]), frozen ? 1u : 0u);
        }
    }
}

TEST(KVCMFrozenTaskTest, DistinctReadTracesOwnIndependentHoldsOnTheSameTarget) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_frozen_shared_target");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    client->load_status = SdkCode::ER_SDKREAD_ERROR;
    const auto before = environment.device_pool->refCount(environment.block_id);
    for (size_t i = 0; i < 2; ++i) {
        auto meta = matchOne(*backend.backend, environment, *client);
        ASSERT_NE(meta, nullptr);
        EXPECT_FALSE(readResult(*backend.backend, makeStorageRequest(environment), std::move(meta)).ok());
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_EQ(backend->frozenTasks().size(), i + 1);
        // Device ownership aggregates tree references into one backing hold.
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before + 1);
        EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), i + 1);
    }
    const auto records = backend->frozenTasks();
    ASSERT_EQ(records.size(), 2u);
    EXPECT_NE(std::get<std::string>(records[0]->execution().trace_id),
              std::get<std::string>(records[1]->execution().trace_id));
    records[0]->unfreeze();
    EXPECT_EQ(records[0]->snapshot().state, FrozenState::RELEASED);
    EXPECT_EQ(records[1]->snapshot().state, FrozenState::FROZEN);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before + 1);
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    records[0]->unfreeze();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before + 1);
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
    records[1]->unfreeze();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before);
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 0u);
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(backend->frozenTasks().size(), 2u);
}

TEST(KVCMFrozenTaskTest, ShutdownAndDestructionNeverAutomaticallyReleaseFrozenTargets) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeBackendEnvironment("kvcm_frozen_shutdown");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    auto meta = matchOne(*backend.backend, environment, *client);
    ASSERT_NE(meta, nullptr);
    client->load_status = SdkCode::ER_SDKREAD_ERROR;
    const auto before = environment.device_pool->refCount(environment.block_id);
    EXPECT_FALSE(readResult(*backend.backend, makeStorageRequest(environment), std::move(meta)).ok());
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    auto records = backend->frozenTasks();
    ASSERT_EQ(records.size(), 1u);
    backend->shutdown();
    EXPECT_EQ(records.front()->snapshot().state, FrozenState::FROZEN);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before + 1);
    backend.backend.reset();
    EXPECT_EQ(records.front()->snapshot().state, FrozenState::FROZEN);
    records.clear();
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), before + 1);
}

TEST(KVCMFrozenTaskTest, WorkerExecutionRecordsProofWithoutOwningCoordinatorFrozenTargets) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    auto environment = makeBackendEnvironment("kvcm_frozen_worker_boundary");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    client->load_status = SdkCode::ER_SDK_TIMEOUT;
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    request.set_trace_id("worker-read");
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    request.add_uris("pace://worker");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(request, response));
    EXPECT_FALSE(response.payload_buffer_safe());
    EXPECT_TRUE(backend->frozenTasks().empty());
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    const auto& registry = backend->taskRegistry();
    EXPECT_EQ(registry.query(registry.epoch(), std::vector<std::string>{"worker-read"}).front(),
              TransferTaskRegistry::State::RUNNING);
}

}  // namespace
}  // namespace rtp_llm
