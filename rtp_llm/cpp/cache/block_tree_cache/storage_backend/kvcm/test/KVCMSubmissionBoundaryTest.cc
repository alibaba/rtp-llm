#include "rtp_llm/cpp/utils/DecodePublicationTest.h"
// Isolated white-box target: compile the CURRENT backend source, not a copied
// proof/settlement model. A test-only genrule includes its exact current bytes.
// Other targets still link the normal backend library.
// Preload its dependencies so the substitutions below affect only this backend
// and its test fixture, never the real BroadcastManager or CUDA declarations.
#include <cuda_runtime_api.h>
#include <condition_variable>
#include <thread>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cctype>
#include <exception>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <map>
#include <numeric>
#include <new>
#include <set>
#include <tuple>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <unistd.h>
#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "kvcm_client/HostCacheStateCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PayloadDispatchProof.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PublicationSettlement.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DecodePublicationAccounting.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/WorkerPayloadProof.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MockKVCMClient.h"

namespace rtp_llm {
class BoundaryBroadcastManager: public BroadcastManager {
public:
    using BroadcastManager::BroadcastManager;
    std::function<void(size_t)> before_rank;
    mutable std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> observed;

    template<class Request, class Response, class RpcCall>
    auto broadcast(const std::vector<Request>& requests, int timeout, const RpcCall& call,
                   std::optional<int64_t> id = {}, std::string direction = "unknown", bool fatal = true) const {
        size_t rank = 0;
        auto gate = [&](const auto& stub, const auto& context, const auto& request, auto* queue) {
            if (before_rank) before_rank(rank);
            ++rank;
            // Invoke the backend's ACTUAL preflight/marker lambda, including its
            // BroadcastRpcNotStarted throw. The gate never manufactures proof.
            return call(stub, context, request, queue);
        };
        auto result = BroadcastManager::broadcast<Request, Response>(requests, timeout, gate, id, direction, fatal);
        observed = result;
        return result;
    }
};
} // namespace rtp_llm

namespace {
thread_local std::function<void()> boundary_before_cuda_query;
cudaError_t boundaryCudaGetDevice(int* device) {
    if (boundary_before_cuda_query) boundary_before_cuda_query();
    return ::cudaGetDevice(device); // Keep the real CUDA call and device selection.
}
} // namespace

// Compile-time dependency gates exist only in this test translation unit. No
// production header, submission API, clock or runtime protocol gains a hook.
#define BroadcastManager BoundaryBroadcastManager
#define cudaGetDevice boundaryCudaGetDevice
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMSubmissionBoundaryImplementation.inc"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#undef cudaGetDevice
#undef BroadcastManager

namespace rtp_llm {
namespace {
using BoundaryClock = std::chrono::steady_clock;

class KVCMSubmissionBoundaryTest: public ::testing::Test {
protected:
    static constexpr auto kSetupWait = std::chrono::seconds(5);
    static constexpr auto kOperationTimeout = 2 * kSetupWait;
    static constexpr auto kSettlementWait = kOperationTimeout + kSetupWait;

    autil::EnvGuard quarantine_abort{"RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0"};
    autil::EnvGuard async{"RTP_LLM_STREAM_ASYNC", "0"};
    autil::EnvGuard pending_bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1"};
    void SetUp() override { initLogger(); }

    struct Fixture {
        BackendEnvironment env = makeBackendEnvironment("submission_boundary");
        std::shared_ptr<MockClientWrapper> client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<MockQuiescentKVCMBackend> backend;
        std::atomic<int> starts{0}, saves{0}, closes{0};
        std::atomic<bool> bad_close{false};
        size_t refs = env.device_pool->refCount(env.block_id);

        Fixture(bool decode, std::shared_ptr<BoundaryBroadcastManager> broadcast = {},
                size_t workers = 1, size_t queue_size = 8) {
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).WillRepeatedly(Invoke(
                [this](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                    ++starts;
                    kv_cache_manager::WriteLocation location;
                    location.write_session_id = "boundary-session";
                    location.block_mask = kv_cache_manager::BlockMaskOffset{0};
                    for (size_t key = 0; key < keys.size(); ++key) {
                        kv_cache_manager::Location row;
                        for (const auto& item : backend->impl_->group_policy_->spec_info_map())
                            row.push_back({item.first, "pace://boundary:" + std::to_string(item.second.tp_rank)});
                        location.locations.push_back(std::move(row));
                    }
                    return std::make_pair(true, location);
                }));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).WillRepeatedly(Invoke(
                [this](const auto&, const auto& uris, const auto& buffers, const auto&) {
                    ++saves;
                    for (const auto& buffer : buffers)
                        for (const auto& iov : buffer.iovs)
                            EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                    return std::make_pair(true, uris); // Completed source; destination still needs all peers.
                }));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).WillRepeatedly(Invoke(
                [this](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                    ++closes;
                    const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&mask);
                    bad_close = !locations.empty() || !offset || *offset != 0;
                    return true; // ACK only an unused-session close.
                }));
            KVCacheConfig config;
            config.kvcm_server_address = "unused-test-address";
            config.kvcm_instance_id_salt = decode ? "projected-boundary" : "broadcast-boundary";
            config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
            config.kvcm_asyncwrapper_thread_num = workers;
            config.kvcm_asyncwrapper_queue_size = queue_size;
            auto parallel = singleRankConfig();
            parallel.role_type = decode ? RoleType::DECODE : RoleType::PREFILL;
            parallel.tp_size = decode ? 1 : 2;
            parallel.prefill_cp_config.prefill_cp_size = 2;
            parallel.prefill_cp_config.kv_cache_sharded = false;
            RuntimeConfig runtime;
            runtime.model_name = "boundary_test";
            backend = std::make_unique<MockQuiescentKVCMBackend>(env.cache_config, config, runtime, parallel,
                SpeculativeExecutionConfig{}, std::move(broadcast), true, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Fixture() { if (backend) { backend->proveMockQuiescence(); backend->shutdown(); } }

        StorageRequest request(bool decode) {
            if (!decode) return makeStorageRequest(env);
            BatchKVCacheResource resource;
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            resource.setBatchBlocks(0, "default", {env.block_id});
            resource.setBatchCacheKeys(0, {101});
            return backend->impl_->planCompletedDecode(resource, env.cache_config.seq_size_per_block);
        }
        bool submit(StorageRequest request) {
            return backend->write(backend->prepareWrite(std::move(request)));
        }
        void expectSettled(bool retained, bool source_unknown) {
            {
                std::unique_lock<std::mutex> lock(backend->lifecycle_mutex_);
                ASSERT_TRUE(backend->lifecycle_cv_.wait_for(
                    lock, kSettlementWait, [this] { return backend->in_flight_ == 0; }));
            }
            EXPECT_EQ(starts.load(), 1);
            EXPECT_EQ(closes.load(), retained ? 0 : 1);
            EXPECT_FALSE(bad_close.load());
            const bool protected_owner = source_unknown && backend->protectUnknownCompletion();
            // Ordinary P writes only read local sources and keep no FROZEN target. On
            // retains the live source owner/pins until quiescence; freezing happens during shutdown.
            EXPECT_FALSE(env.device_pool->isFrozen(env.block_id));
            if (!protected_owner) {
                EXPECT_EQ(env.device_pool->refCount(env.block_id), refs);
            } else {
                EXPECT_GT(env.device_pool->refCount(env.block_id), refs);
            }
            std::lock_guard<std::mutex> lock(backend->quarantine_mutex_);
            EXPECT_EQ(static_cast<bool>(backend->protected_quarantine_), protected_owner);
            EXPECT_EQ(backend->quarantined_task_count_, source_unknown ? 1u : 0u);
        }
    };

    void broadcastCase(bool partial) {
        for (const char* enabled : {"0", "1"}) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled);
            ::rtp_llm::test::withDecodePublicationStartup([&] {
            SCOPED_TRACE(::testing::Message() << "enabled=" << enabled << " partial=" << partial);
            auto first_state = std::make_shared<KVCMBroadcastState>();
            auto second_state = std::make_shared<KVCMBroadcastState>();
            auto arrived = std::make_shared<std::promise<void>>();
            auto arrival = arrived->get_future().share();
            first_state->on_request = [arrived] { arrived->set_value(); };
            KVCMBroadcastRpcServer first(0, first_state), second(1, second_state);
            ASSERT_TRUE(first.start()); ASSERT_TRUE(second.start());
            auto broadcast = std::make_shared<BoundaryBroadcastManager>(
                std::vector<std::string>{first.address(), second.address()});
            ASSERT_TRUE(broadcast->init()); // This is NOT the uninitialized-manager test.
            Fixture s(false, broadcast);
            ASSERT_TRUE(s.backend->initialized_);
            auto request = s.request(false);
            request.transfer_task.emplace(std::vector<TransferDescriptor>{}, kOperationTimeout);
            const auto deadline = request.transfer_task->deadline();
            auto gates = std::make_shared<std::atomic<int>>(0);
            broadcast->before_rank = [partial, deadline, arrival, gates](size_t rank) {
                ++*gates;
                if (rank != (partial ? 1u : 0u)) return;
                if (partial) {
                EXPECT_EQ(arrival.wait_for(kSetupWait), std::future_status::ready);
            }
                std::this_thread::sleep_until(deadline + std::chrono::milliseconds(1));
            };
            ASSERT_TRUE(s.submit(std::move(request)));
            ASSERT_NO_FATAL_FAILURE(s.expectSettled(false, partial));
            ASSERT_NE(broadcast->observed, nullptr);
            EXPECT_TRUE(broadcast->observed->done());
            EXPECT_FALSE(broadcast->observed->success());
            if (partial) {
                EXPECT_TRUE(broadcast->observed->rpcSucceeded(0));
            }
            EXPECT_FALSE(broadcast->observed->rpcSucceeded(1));
            EXPECT_EQ(gates->load(), partial ? 2 : 1);
            EXPECT_EQ(snapshotRequests(first_state).size(), partial ? 1u : 0u);
            EXPECT_TRUE(snapshotRequests(second_state).empty());
            EXPECT_EQ(s.saves.load(), 0); // Real test RPC servers, no local SDK path.
            });
        }
    }

    struct PeerObservation {
        std::promise<void> ready;
        std::shared_future<void> done = ready.get_future().share();
        std::atomic<size_t> calls{0};
        bool touched{true}, sdk_failed{false};
        std::exception_ptr error;
        std::vector<FunctionResponsePB> responses;
        std::shared_ptr<kvcm::PayloadDispatchProof> proof = std::make_shared<kvcm::PayloadDispatchProof>();
    };

    static std::shared_ptr<PeerObservation> dispatchPeers(Fixture& s, bool expired = false) {
        // Exercise the actual local D dispatch and SDK, independently of metadata.
        // Full Start/Finish integration is covered by projectedCase and cases below.
        auto request = s.request(true);
        request.transfer_task.emplace(std::vector<TransferDescriptor>{},
                                      expired ? std::chrono::milliseconds(0) : kOperationTimeout);
        std::vector<FunctionRequestPB> peers(2);
        for (size_t peer = 0; peer < peers.size(); ++peer) {
            auto* remote = peers[peer].mutable_remote_request();
            remote->set_op(REMOTE_OPERATION_WRITE);
            remote->add_group_tags("default");
            remote->add_block_ids(s.env.block_id);
            remote->add_uris("pace://peer:" + std::to_string(peer));
        }
        auto observed = std::make_shared<PeerObservation>();
        auto prepare = [observed](auto collect) {
            return [observed, collect = std::move(collect)]() mutable {
                ++observed->calls;
                try { observed->responses = collect(observed->touched, observed->sdk_failed); }
                catch (...) { observed->error = std::current_exception(); }
                observed->ready.set_value();
            };
        };
        bool touched = false;
        s.backend->impl_->dispatchLocalPeers(peers, request, touched, prepare, observed->proof);
        EXPECT_TRUE(touched); // Carrier accepted/pending; settlement provides final proof.
        return observed;
    }

    void projectedCase(bool mixed) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        Fixture s(true);
        ASSERT_TRUE(s.backend->initialized_);
        auto request = s.request(true);
        ASSERT_FALSE(request.empty()); ASSERT_TRUE(request.decode_completion);
        auto deadline = std::make_shared<BoundaryClock::time_point>();
        auto checks = std::make_shared<std::atomic<int>>(0);
        auto entered = std::make_shared<std::promise<void>>();
        auto entered_future = entered->get_future();
        auto release = std::make_shared<std::promise<void>>();
        auto released = release->get_future().share();
        // Pin the sole REAL payload worker, not the business/completion lane.
        // Its next CUDA query is runLocalPeer AFTER cancel/expiry checks, BEFORE SDK.
        ASSERT_TRUE(s.backend->impl_->local_payload_pool_->submit(BlockTreeTaskClass::BACKGROUND,
            [entered, released, checks, deadline, mixed] {
                boundary_before_cuda_query = [checks, deadline, mixed] {
                    if (++*checks == (mixed ? 2 : 1))
                        std::this_thread::sleep_until(*deadline + std::chrono::milliseconds(1));
                };
                entered->set_value();
                EXPECT_EQ(released.wait_for(kSettlementWait), std::future_status::ready);
            }));
        block_tree_cache_detail::ScopeRollback unblock([release] { release->set_value(); });
        ASSERT_EQ(entered_future.wait_for(kSetupWait), std::future_status::ready);
        // Start the operation clock only after the worker gate is installed.
        // Promise release synchronizes this value with the worker's first query.
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, kOperationTimeout);
        *deadline = request.transfer_task->deadline();
        ASSERT_TRUE(s.submit(std::move(request)));
        // One worker + FIFO BACKGROUND barrier proves projected dispatch has
        // enqueued both peers while the payload worker is still held.
        auto barrier = std::make_shared<std::promise<void>>();
        auto queued = barrier->get_future();
        ASSERT_TRUE(s.backend->executor_->submit([barrier] { barrier->set_value(); }, BlockTreeTaskClass::BACKGROUND));
        ASSERT_EQ(queued.wait_for(kSetupWait), std::future_status::ready);
        EXPECT_EQ(s.backend->impl_->local_payload_pool_->queueSizes().background, 2u);
        unblock.run();
        // SDK success of peer0 proves source BUFFER_SAFE in the mixed case; it
        // must NOT be misused as permission to abort an incomplete destination.
        ASSERT_NO_FATAL_FAILURE(s.expectSettled(mixed, false));
        EXPECT_EQ(checks->load(), mixed ? 2 : 1);
        EXPECT_EQ(s.saves.load(), mixed ? 1 : 0);
        EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 0u);
        });
    }
};

TEST_F(KVCMSubmissionBoundaryTest, CleanupRetriesOnlyLocallyProvenNotSentWithinOriginalDeadline) {
    for (const bool enabled : {false, true}) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        using Result = kvcm::ClientWrapper::FinishWriteResult;
        struct Case {
            const char* name;
            std::vector<Result> replies;
            bool cleanup = true;
            bool expired = false;
            bool no_task = false;
            bool throws = false;
            size_t calls = 1;
            bool acknowledged = false;
            bool stopping = false;
        };
        const std::vector<Case> cases{
            {"local refusal then ACK", {Result::NOT_SENT, Result::NOT_SENT, Result::ACKNOWLEDGED},
             true, false, false, false, 3, true},
            {"bounded local refusals", {Result::NOT_SENT}, true, false, false, false, 3},
            {"entered after local refusal", {Result::NOT_SENT, Result::UNKNOWN}, true, false, false, false, 2},
            {"entered without ACK", {Result::UNKNOWN}},
            {"throw after entry", {}, true, false, false, true},
            {"commit never retries", {Result::NOT_SENT}, false},
            {"expired cleanup", {Result::NOT_SENT}, true, true},
            {"no deadline proof", {Result::NOT_SENT}, true, false, true},
            {"shutdown stops retry", {Result::NOT_SENT}, true, false, false, false, 1, false, true},
        };
        for (const bool decode : {false, true}) {
            if (decode && !enabled) continue;
            for (const auto& test : cases) {
                SCOPED_TRACE(::testing::Message() << "enabled=" << enabled
                             << " decode=" << decode << " case=" << test.name);
                auto broadcast = decode ? nullptr : std::make_shared<BoundaryBroadcastManager>(
                    std::vector<std::string>{"unused-cleanup-peer0", "unused-cleanup-peer1"});
                Fixture fixture(decode, std::move(broadcast));
                ASSERT_TRUE(fixture.backend->initialized_);
                auto request = fixture.request(decode);
                if (test.no_task) request.transfer_task.reset();
                else request.transfer_task.emplace(std::vector<TransferDescriptor>{},
                    test.expired ? std::chrono::milliseconds(-1) : kOperationTimeout);
                KVCMStorageBackend::Impl::WritePublication write;
                write.start_acknowledged = true;
                write.location.write_session_id = "cleanup-boundary-session";
                kv_cache_manager::Locations locations;
                if (!test.cleanup) locations.resize(1);
                size_t calls = 0;
                fixture.client->finish_write_result = [&](const auto& mask, const auto& locations) -> Result {
                    ++calls;
                    const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&mask);
                    EXPECT_NE(offset, nullptr);
                    if (offset) {
                        EXPECT_EQ(*offset, test.cleanup ? 0u : 1u);
                    }
                    EXPECT_EQ(locations.size(), test.cleanup ? 0u : 1u);
                    if (test.throws) throw std::runtime_error("FinishWrite ACK may be lost");
                    if (test.stopping) fixture.client->kvcm::ClientWrapper::shutdown();
                    return test.replies[std::min(calls - 1, test.replies.size() - 1)];
                };
                const auto finish = [&] {
                    return fixture.backend->impl_->finishSession(request, write,
                        kv_cache_manager::BlockMaskOffset{test.cleanup ? 0u : 1u}, locations, test.cleanup);
                };
                if (test.throws) {
                    EXPECT_THROW(finish(), std::runtime_error);
                } else {
                    EXPECT_EQ(finish(), test.acknowledged);
                }
                EXPECT_EQ(calls, test.calls);
                EXPECT_TRUE(write.finish_attempted);
                // No second cleanup after ACK, NOT_SENT exhaustion, UNKNOWN or
                // an exception; normal commit failures cannot become aborts.
                fixture.backend->impl_->closeUnusedWrite(request, write);
                EXPECT_EQ(calls, test.calls);
                EXPECT_TRUE(write.dispatch_proof.neverStarted());
                EXPECT_EQ(fixture.starts.load(), 0);
                EXPECT_EQ(fixture.saves.load(), 0);
                EXPECT_EQ(fixture.env.device_pool->refCount(fixture.env.block_id), fixture.refs);
                EXPECT_FALSE(fixture.env.device_pool->isFrozen(fixture.env.block_id));
            }
        }
    });
    }
}

TEST_F(KVCMSubmissionBoundaryTest, OnlyDecodeCreatesTransportRetentionOwner) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        for (const bool decode : {false, true}) {
            // P TP2 requires its original broadcast manager, but this test
            // submits no RPC or payload; only actual backend init is observed.
            auto broadcast = decode ? nullptr : std::make_shared<BoundaryBroadcastManager>(
                std::vector<std::string>{"unused-owner-peer0", "unused-owner-peer1"});
            Fixture s(decode, std::move(broadcast));
            ASSERT_TRUE(s.backend->initialized_);
            EXPECT_EQ(static_cast<bool>(s.backend->completionResourceOwner()), decode);
            if (decode) {
                ASSERT_TRUE(s.backend->impl_->completion_resources_);
                EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 0u);
            } else {
                EXPECT_FALSE(s.backend->impl_->completion_resources_);
            }
            EXPECT_EQ(s.starts.load(), 0);
            EXPECT_EQ(s.saves.load(), 0);
            EXPECT_EQ(s.closes.load(), 0);
        }
    });
}

TEST_F(KVCMSubmissionBoundaryTest, InitializedBroadcastWithZeroRankHandoffsClosesUnusedSession) {
    broadcastCase(false);
}
TEST_F(KVCMSubmissionBoundaryTest, PartialRankHandoffCannotBeDowngradedByNextRankExpiry) {
    broadcastCase(true);
}
TEST_F(KVCMSubmissionBoundaryTest, ProjectedAllPeersStopBeforeSdkClosesUnusedSession) {
    projectedCase(false);
}
TEST_F(KVCMSubmissionBoundaryTest, ProjectedOneSdkHandoffAndOnePreflightExpiryRetainsSession) {
    projectedCase(true);
}

TEST_F(KVCMSubmissionBoundaryTest, TargetProtectionPreservesExceptionalSdkUnknownType) {
    const auto failure = std::make_exception_ptr(StorageOperationPayloadExceptionUnknown(
        "exceptional SDK completion", StorageQuarantineReason::RPC_ERROR));
    for (const auto protection : {StorageLocalTargetProtection::LEGACY,
                                  StorageLocalTargetProtection::TARGETS_HELD,
                                  StorageLocalTargetProtection::NO_LOCAL_TARGETS}) {
        try {
            std::rethrow_exception(KVCMStorageBackend::Impl::withTargetProtection(failure, protection));
        } catch (const StorageOperationPayloadExceptionUnknown& error) {
            EXPECT_STREQ(error.what(), "exceptional SDK completion");
            EXPECT_EQ(error.reason(), StorageQuarantineReason::RPC_ERROR);
            EXPECT_EQ(error.localTargetProtection(), protection);
        } catch (...) {
            FAIL() << "target ownership must not erase exceptional SDK UNKNOWN";
        }
    }
}

TEST_F(KVCMSubmissionBoundaryTest, LocalPeerLatchWaitsForDriverAfterEveryInlineTerminal) {
    // Actual production latch: peers can finish inline, but cannot notify until
    // the driver drops +1. No copy of the old window/proof model is tested here.
    using Impl = KVCMStorageBackend::Impl;
    TransferTask task({}, kOperationTimeout);
    auto write = std::make_shared<Impl::LocalPeerWrite>(task, 2);
    write->responses.resize(2);
    for (auto& item : write->responses) {
        item.mutable_remote_response()->set_payload_buffer_safe(true);
        item.mutable_remote_response()->set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
    }
    write->proof = std::make_shared<kvcm::PayloadDispatchProof>();
    write->proof->pending();
    size_t calls = 0;
    const std::weak_ptr<Impl::LocalPeerWrite> weak = write;
    write->business = [write, &calls] { ++calls; }; // Same self-cycle as production collect.
    Impl::finishLocalPeer(write, 0, true);
    EXPECT_EQ(calls, 0u);
    EXPECT_FALSE(write->proof->neverStarted());
    Impl::finishLocalPeer(write, 1, true);
    EXPECT_EQ(calls, 0u);
    EXPECT_EQ(write->remaining.load(), 1u);
    Impl::finishLocalPeers(write);
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(write->proof->neverStarted());
    write.reset();
    EXPECT_TRUE(weak.expired()); // Finalization broke the actual callback/context cycle.
}

TEST_F(KVCMSubmissionBoundaryTest, LocalPeersRejectedOrInlineExpiredSettleOnceWithoutSdk) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        for (const bool expired : {false, true}) {
            Fixture s(true);
            if (!expired) s.backend->impl_->local_payload_pool_->stopAdmission();
            auto observed = dispatchPeers(s, expired);
            ASSERT_EQ(observed->done.wait_for(kSetupWait), std::future_status::ready);
            EXPECT_EQ(observed->calls.load(), 1u);
            EXPECT_TRUE(observed->error);
            EXPECT_FALSE(observed->touched);
            EXPECT_TRUE(observed->proof->neverStarted());
            EXPECT_EQ(s.saves.load(), 0);
            EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 0u);
        }
    });
}

TEST_F(KVCMSubmissionBoundaryTest, PartialLocalQueueRejectionStillWaitsForAcceptedPeer) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        Fixture s(true, {}, 1, 1); // Running job occupies no queue slot; one peer can queue.
        auto entered = std::make_shared<std::promise<void>>();
        auto started = entered->get_future();
        auto release = std::make_shared<std::promise<void>>();
        auto released = release->get_future().share();
        ASSERT_TRUE(s.backend->impl_->local_payload_pool_->submit(BlockTreeTaskClass::BACKGROUND,
            [entered, released] { entered->set_value(); released.wait(); }));
        block_tree_cache_detail::ScopeRollback unblock([release] { release->set_value(); });
        ASSERT_EQ(started.wait_for(kSetupWait), std::future_status::ready);
        auto observed = dispatchPeers(s);
        EXPECT_EQ(s.backend->impl_->local_payload_pool_->queueSizes().background, 1u);
        EXPECT_EQ(observed->done.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
        EXPECT_EQ(observed->calls.load(), 0u);
        EXPECT_FALSE(observed->proof->neverStarted()); // Pending accepted peer, not zero-submission.
        unblock.run();
        ASSERT_EQ(observed->done.wait_for(kSetupWait), std::future_status::ready);
        EXPECT_EQ(observed->calls.load(), 1u);
        EXPECT_TRUE(observed->error);
        EXPECT_FALSE(observed->touched);
        EXPECT_TRUE(observed->proof->neverStarted());
        EXPECT_EQ(s.saves.load(), 0); // Queued accepted peer observes cancellation before SDK.
    });
}

TEST_F(KVCMSubmissionBoundaryTest, ParallelPeersHoldSourcesAndPublishOnceAfterBothSdkCalls) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        Fixture s(true, {}, 2);
        auto mutex = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();
        auto entered = std::make_shared<size_t>(0);
        auto release = std::make_shared<bool>(false);
        EXPECT_CALL(*s.client, saveKvCachesForTag(_, _, _, _)).Times(2).WillRepeatedly(Invoke(
            [&, mutex, cv, entered, release](const auto&, const auto& uris, const auto&, const auto&) {
                ++s.saves;
                std::unique_lock<std::mutex> lock(*mutex);
                ++*entered;
                cv->notify_all();
                EXPECT_TRUE(cv->wait_for(lock, kSettlementWait, [&] { return *release; }));
                return std::make_pair(true, uris);
            }));
        EXPECT_CALL(*s.client, finishWrite(_, _, _, _, _)).Times(1).WillOnce(Invoke(
            [&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                ++s.closes;
                const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&mask);
                s.bad_close = !locations.empty() || !offset || *offset != 1; // Unchanged URIs need no overrides.
                return true;
            }));
        block_tree_cache_detail::ScopeRollback unblock([mutex, cv, release] {
            { std::lock_guard<std::mutex> lock(*mutex); *release = true; }
            cv->notify_all();
        });
        auto request = s.request(true);
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, kOperationTimeout);
        ASSERT_TRUE(s.submit(std::move(request)));
        {
            std::unique_lock<std::mutex> lock(*mutex);
            ASSERT_TRUE(cv->wait_for(lock, kSetupWait, [&] { return *entered == 2; }));
        }
        EXPECT_EQ(s.saves.load(), 2); // Two actual SDK mocks entered before either returns.
        EXPECT_EQ(s.closes.load(), 0);
        EXPECT_GT(s.env.device_pool->refCount(s.env.block_id), s.refs);
        unblock.run();
        ASSERT_NO_FATAL_FAILURE(s.expectSettled(false, false));
        EXPECT_EQ(s.saves.load(), 2);
        EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 0u);
    });
}

TEST_F(KVCMSubmissionBoundaryTest, ShutdownDrainsAcceptedLocalPeersBeforeClientShutdown) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        auto fixture = std::make_shared<Fixture>(true, std::shared_ptr<BoundaryBroadcastManager>{}, 2);
        auto& s = *fixture;
        auto entered = std::make_shared<std::promise<void>>();
        auto started = entered->get_future();
        auto release = std::make_shared<std::promise<void>>();
        auto released = release->get_future().share();
        EXPECT_CALL(*s.client, saveKvCachesForTag(_, _, _, _)).Times(2).WillRepeatedly(Invoke(
            [saves = &s.saves, entered, released](const auto&, const auto& uris, const auto&, const auto&) {
                if (++*saves == 2) entered->set_value();
                EXPECT_EQ(released.wait_for(kSettlementWait), std::future_status::ready);
                return std::make_pair(true, uris);
            }));
        block_tree_cache_detail::ScopeRollback unblock([release] { release->set_value(); });
        auto observed = dispatchPeers(s);
        ASSERT_EQ(started.wait_for(kSetupWait), std::future_status::ready);
        // Accepted operations remain live when new payload admission stops.
        s.backend->impl_->local_payload_pool_->stopAdmission();
        EXPECT_FALSE(s.backend->impl_->submitLocalPayload(BlockTreeTaskClass::BACKGROUND,
            [] {}, BoundaryClock::now() + kOperationTimeout, [] {}));
        block_tree_cache_test::BoundedThread<void> stopping([fixture] { fixture->backend->shutdown(); });
        // On timeout the joiner detaches while retaining fixture, never a stack reference.
        // Unblock before the thread handle is destroyed on an early ASSERT.
        block_tree_cache_detail::ScopeRollback release_before_join([&] { unblock.run(); });
        EXPECT_EQ(stopping.waitFor(std::chrono::milliseconds(20)), std::future_status::timeout);
        EXPECT_EQ(observed->calls.load(), 0u);
        unblock.run();
        ASSERT_EQ(observed->done.wait_for(kSetupWait), std::future_status::ready);
        EXPECT_EQ(observed->calls.load(), 1u);
        EXPECT_FALSE(observed->error);
        EXPECT_FALSE(observed->touched);
        EXPECT_EQ(observed->responses.size(), 2u);
        EXPECT_FALSE(observed->proof->neverStarted());
        ASSERT_EQ(stopping.waitFor(kSetupWait), std::future_status::ready);
        stopping.get();
    });
}

TEST_F(KVCMSubmissionBoundaryTest, LocalPeerSafeFailureCancelsTailButCannotAbortStartedSession) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        Fixture s(true); // One worker makes the canceled tail deterministic.
        EXPECT_CALL(*s.client, saveKvCachesForTag(_, _, _, _)).Times(1).WillOnce(Invoke(
            [&](const auto&, const auto&, const auto&, const auto&) {
                ++s.saves;
                // SDK returned buffer-safe success, but an invalid URI cardinality.
                return std::make_pair(true, kv_cache_manager::UriStrVec{"pace://bad:0", "pace://bad:1"});
            }));
        ASSERT_TRUE(s.submit(s.request(true)));
        ASSERT_NO_FATAL_FAILURE(s.expectSettled(true, false));
        EXPECT_EQ(s.saves.load(), 1);
        EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 0u);
    });
}

TEST_F(KVCMSubmissionBoundaryTest, LocalPeerUnknownRetainsOperationAndSourceWithoutFinishOrAbort) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        Fixture s(true);
        EXPECT_CALL(*s.client, saveKvCachesForTag(_, _, _, _)).Times(1).WillOnce(Invoke(
            [&](const auto&, const auto&, const auto&, const auto&) {
                ++s.saves;
                return std::make_pair(false, kv_cache_manager::UriStrVec{}); // Explicitly no safety proof.
            }));
        ASSERT_TRUE(s.submit(s.request(true)));
        ASSERT_NO_FATAL_FAILURE(s.expectSettled(true, true));
        EXPECT_EQ(s.saves.load(), 1);
        EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 1u);
        std::lock_guard<std::mutex> lock(s.backend->impl_->completion_resources_->mutex);
        ASSERT_TRUE(s.backend->impl_->completion_resources_->operations);
        EXPECT_EQ(s.backend->impl_->completion_resources_->operations->client, s.client);
        EXPECT_FALSE(s.backend->impl_->completion_resources_->operations->buffers.empty());
    });
}

template<typename Code>
std::vector<std::pair<Code, bool>> projectedSdkFailures() {
    std::vector<std::pair<Code, bool>> cases{{Code::ER_SDKWRITE_ERROR, false},
                                            {Code::ER_INVALID_PARAMS, false},
                                            {Code::ER_SDK_TIMEOUT, false}};
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        cases.emplace_back(Code::ER_SDK_TIMEOUT_BUFFER_SAFE, true);
        cases.emplace_back(Code::ER_SDK_ERROR_BUFFER_SAFE, true);
    }
    return cases;
}

TEST_F(KVCMSubmissionBoundaryTest, DecodeSdkFailuresKeepDestinationAndRetainOnlyUnprovenSources) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    test::withDecodePublicationStartup([&] {
        for (const auto& [code, source_safe] : projectedSdkFailures<kv_cache_manager::ClientErrorCode>()) {
            SCOPED_TRACE(static_cast<int>(code));
            Fixture s(true);
            // Typed mock SDK boundary: no real DMA. Exercise the actual D
            // projection, peer dispatch, completion and source-owner settlement.
            s.client->save_status = code;
            ASSERT_TRUE(s.submit(s.request(true)));
            ASSERT_NO_FATAL_FAILURE(s.expectSettled(/*retained=*/true, /*source_unknown=*/!source_safe));
            EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), source_safe ? 0u : 1u);
            EXPECT_EQ(s.closes.load(), 0); // Source-safe failure still cannot close a started destination.
        }
    });
}

TEST_F(KVCMSubmissionBoundaryTest, ProjectedWireSizeUsesRegisteredContractAndStillRejectsShortPayload) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    test::withDecodePublicationStartup([&] {
        Fixture s(true); // Existing CP2 replicated fixture; no SDK call is submitted below.
        const auto& groups = s.backend->impl_->group_policy_->groups();
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_EQ(groups.begin()->second.tag, "default");
        EXPECT_EQ(groups.begin()->second.block_size_bytes, 64u); // K+V, 8 tokens, head dimension 2, FP16.
        const auto parts = environmentBuffers(s.env, 0, "default", s.env.block_id);
        ASSERT_FALSE(parts.empty());
        size_t total_bytes = 0;
        for (const auto& part : parts) {
            ASSERT_TRUE(part.is_cuda);
            ASSERT_NE(part.addr, nullptr);
            ASSERT_GT(part.size_bytes, 0u);
            total_bytes += part.size_bytes;
        }
        ASSERT_EQ(total_bytes, 64u); // Native K/V may be contiguous; the wire contract is unchanged.
        RemoteOperationRequestPB request;
        request.add_uris("pace://wire-size-check");
        for (int peer : {0, 1}) {
            kv_cache_manager::BlockBuffers buffers;
            ASSERT_TRUE(s.backend->impl_->prepareBlockBuffers(
                request, {"default"}, {s.env.block_id}, buffers, peer));
            ASSERT_EQ(buffers.size(), 1u);
            ASSERT_EQ(buffers.front().iovs.size(), parts.size());
            for (size_t i = 0; i < parts.size(); ++i) {
                EXPECT_EQ(buffers.front().iovs[i].type, kv_cache_manager::MemoryType::GPU);
                EXPECT_EQ(buffers.front().iovs[i].base, parts[i].addr);
                EXPECT_EQ(buffers.front().iovs[i].size, parts[i].size_bytes);
            }
        }
        auto original = s.backend->impl_->file_resolver_;
        block_tree_cache_detail::ScopeRollback restore([&] { s.backend->impl_->file_resolver_.swap(original); });
        s.backend->impl_->file_resolver_ = [resolver = original](int layer, const std::string& tag, int block) {
            auto parts = resolver(layer, tag, block);
            // Remain nonempty/GPU/nonzero: force the wire-size guard itself.
            if (!parts.empty() && parts.front().size_bytes > 1) --parts.front().size_bytes;
            return parts;
        };
        kv_cache_manager::BlockBuffers short_payload;
        EXPECT_FALSE(s.backend->impl_->prepareBlockBuffers(
            request, {"default"}, {s.env.block_id}, short_payload, 0));
        EXPECT_EQ(s.starts.load(), 0);
        EXPECT_EQ(s.saves.load(), 0);
        EXPECT_EQ(s.closes.load(), 0);
    });
}

TEST_F(KVCMSubmissionBoundaryTest, OrdinaryFullPrefixIntersectsGroupsAndEndpointCannotBackOff) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    test::withDecodePublicationStartup([&] {
        // Reuse this target's actual backend-source seam. No copied planner or
        // product hook; selection-only fixture, not a model/SDK transfer test.
        BackendEnvironment env;
        env.cache_config.dtype = DataType::TYPE_FP16;
        env.cache_config.layer_num = 2;
        env.cache_config.seq_size_per_block = 8;
        auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
        env.cache_config.fromGroupedSpecs(
            {test::makeMhaSpec("full0", 8, DataType::TYPE_FP16, 1, 2),
             test::makeMhaSpec("full1", 8, DataType::TYPE_FP16, 1, 2)},
            {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::FULL}, {"full0", "full1"}, {policy, policy});
        env.cache_config.finalizeBlockNums(8, RuntimeConfig{});
        initializeEnvironmentPools(env, "full_prefix_intersection", 8);
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).Times(0);
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::DECODE;
        parallel.prefill_cp_config.prefill_cp_size = 1;
        KVCacheConfig settings;
        settings.kvcm_model_sdk_config = R"([{"type":"pace"}])";
        auto backend = makeBackend(env, parallel, client, nullptr, nullptr, true, settings);
        ASSERT_TRUE(initSingleRank(*backend.backend, env));
        for (int variant = 0; variant < 5; ++variant) {
            SCOPED_TRACE(variant);
            BatchKVCacheResource resource;
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            BlockIndicesType first(4, env.block_id), second(4, env.block_id);
            size_t expected_end = 8;
            if (variant == 0) { first[3] = NULL_BLOCK_IDX; second[1] = NULL_BLOCK_IDX; }
            if (variant == 1) { first[1] = NULL_BLOCK_IDX; second[3] = NULL_BLOCK_IDX; }
            if (variant == 2) { first.resize(2); second[3] = NULL_BLOCK_IDX; expected_end = 16; }
            if (variant == 3) { second[0] = NULL_BLOCK_IDX; expected_end = 0; }
            if (variant == 4) { first[1] = NULL_BLOCK_IDX; expected_end = 0; }
            resource.setBatchBlocks(0, "full0", first);
            resource.setBatchBlocks(0, "full1", second);
            resource.setBatchCacheKeys(0, {101, 102, 103, 104});
            if (variant == 4) {
                // A malformed exact endpoint has a valid shorter FULL prefix,
                // but must be rejected, never relabeled. No pin/content claim.
                auto endpoint = std::make_shared<DecodeEndpoint>();
                endpoint->end = 32;
                endpoint->blocks = {{"full0", first}, {"full1", second}};
                resource.decode_endpoint = std::move(endpoint);
            }
            {
                auto request = backend.backend->impl_->planCompletedDecode(resource, 32);
                if (expected_end) {
                    ASSERT_FALSE(request.empty());
                    EXPECT_EQ(request.published_tokens, expected_end);
                    EXPECT_EQ(request.keys->size(), expected_end / 8);
                    EXPECT_EQ(request.computed_tokens, 32u);
                    for (const auto& handles : request.handles) EXPECT_EQ(handles.size(), 2u);
                } else EXPECT_TRUE(request.empty());
            } // Return any plan-only budget before the next case; no task is submitted.
            const auto usage = backend.backend->impl_->completion_accounting_->snapshot();
            EXPECT_EQ(usage.bytes, 0u);
            EXPECT_EQ(usage.jobs, 0u);
            for (const auto& [tag, pool] : env.pools_by_tag)
                EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u) << tag;
        }
    });
}

} // namespace
} // namespace rtp_llm
