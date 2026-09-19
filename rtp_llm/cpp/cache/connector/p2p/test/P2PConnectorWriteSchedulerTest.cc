#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <thread>

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {
namespace {

class WriteSchedulerService: public RpcService::Service {
public:
    using Handler = std::function<void(
        const P2PConnectorStartWriteRequestPB&, P2PConnectorStartWriteResponsePB&, std::function<bool()>)>;
    Handler           handshake;
    std::atomic<bool> fail_start{false}, lose_handshake{false}, omit_status{false};
    std::atomic<int>  starts{0}, cancels{0}, handshakes{0};

    grpc::Status StartWrite(grpc::ServerContext*                   context,
                            const P2PConnectorStartWriteRequestPB* request,
                            P2PConnectorStartWriteResponsePB*      response) override {
        ++handshakes;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_handshake_ = *request;
        }
        handshake(*request, *response, [context]() { return context->IsCancelled(); });
        return lose_handshake ? grpc::Status(grpc::StatusCode::INTERNAL, "lost handshake") : grpc::Status::OK;
    }

    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto&                 input = request->p2p_request();
        auto&                       state = states_[input.unique_key()];
        if (draining_) {
            state.stopped   = true;
            state.cancelled = true;
        }
        if (input.write_operation() == WRITE_START) {
            ++starts;
            last_start_ = input;
            if (fail_start) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "lost worker START response");
            }
        } else if (input.write_operation() == WRITE_CANCEL) {
            ++cancels;
            state.cancelled = true;
        }
        if (omit_status && input.write_operation() != WRITE_START) {
            return grpc::Status::OK;
        }
        auto* output = response->mutable_p2p_response();
        auto* lease  = output->mutable_lease_status();
        lease->set_sealed(true);
        lease->set_started_ops(1);
        lease->set_finished_ops(state.stopped ? 1 : 0);
        lease->set_stopped(state.stopped);
        output->set_write_success(state.stopped && !state.cancelled && state.success);
        return grpc::Status::OK;
    }

    void completeAll(bool success = true, bool draining = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        draining_ = draining_ || draining;
        for (auto& [key, state] : states_) {
            state.stopped = true;
            state.success = success;
        }
    }
    P2PConnectorBroadcastTpRequestPB lastStart() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_start_;
    }
    P2PConnectorStartWriteRequestPB lastHandshake() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_handshake_;
    }

private:
    struct State {
        bool stopped{false};
        bool success{false};
        bool cancelled{false};
    };
    mutable std::mutex               mutex_;
    std::map<std::string, State>     states_;
    bool                             draining_{false};
    P2PConnectorBroadcastTpRequestPB last_start_;
    P2PConnectorStartWriteRequestPB  last_handshake_;
};

class P2PConnectorWriteSchedulerTest: public ::testing::Test {
protected:
    using Allocator = test::BlockTreeCacheTestAllocator<SingleTypeKVCacheAllocator>;

    void SetUp() override {
        config_    = test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1, 2);
        allocator_ = std::make_shared<Allocator>(config_);
        ASSERT_TRUE(allocator_->init());
        for (int i = 0; i < 2; ++i) {
            launch(prefill_workers_[i], prefill_addrs_);
            launch(decode_workers_[i], decode_addrs_);
        }
        prefill_client_ = std::make_shared<P2PBroadcastClient>(prefill_addrs_, 100);
        decode_client_  = std::make_shared<P2PBroadcastClient>(decode_addrs_, 100);
        ASSERT_TRUE(prefill_client_->init());
        ASSERT_TRUE(decode_client_->init());
        prefill_config_ = schedulerConfig(RoleType::PREFILL, prefill_addrs_);
        decode_config_  = schedulerConfig(RoleType::DECODE, decode_addrs_);
        for (const auto& address : prefill_addrs_) {
            prefill_config_.p2p_worker_addrs.push_back(address + ":12345");
        }
        token_ids_               = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        kv_ready_token_count_    = 12;
        input_length_            = 4;
        routing_.unique_key      = "write-scheduler";
        routing_.request_id      = 17;
        routing_.prefill_tp_size = 2;
        routing_.prefill_cp_size = 1;
        routing_.prefill_addr    = {"127.0.0.1", static_cast<uint32_t>(ports_.front())};
        keys_                    = calculateCacheKeys(token_ids_.data(), 12, 4);
        ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_.front()}).success);
        prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
        ASSERT_TRUE(prefill_->init());
        prefill_workers_[0].handshake = [this](const auto& request, auto& response, auto cancelled) {
            prefill_->handleWrite(request, response, cancelled);
        };
        createDecode();
    }

    void TearDown() override {
        if (load_barrier_) {
            load_barrier_->release();
        }
        for (int i = 0; i < 2; ++i) {
            prefill_workers_[i].omit_status = false;
            decode_workers_[i].omit_status  = false;
        }
        completeAll(true);
        decode_.reset();
        prefill_.reset();
        for (auto& server : servers_) {
            server->Shutdown();
            server->Wait();
        }
    }

    void launch(WriteSchedulerService& service, std::vector<std::string>& addresses) {
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        servers_.push_back(std::move(server));
        ports_.push_back(port);
        addresses.push_back("127.0.0.1:" + std::to_string(port));
    }

    P2PConnectorSchedulerConfig schedulerConfig(RoleType role, const std::vector<std::string>& addresses) {
        P2PConnectorSchedulerConfig config;
        config.topology                                     = config_.topologyPtr();
        config.parallelism_config.tp_size                   = 2;
        config.parallelism_config.role_type                 = role;
        config.role_type                                    = role;
        config.worker_grpc_addrs                            = addresses;
        config.p2p_resource_store_timeout_check_interval_ms = 5;
        config.p2p_cancel_broadcast_timeout_ms              = 100;
        config.p2p_writeback_timeout_ms                     = 4000;
        return config;
    }

    void createDecode() {
        decode_ = std::make_unique<P2PSchedulerDecodeWrite>(decode_config_, decode_client_);
        ASSERT_TRUE(decode_->init());
    }

    static bool await(const std::function<bool()>& predicate, int timeout_ms = 5000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    }

    KVCacheResourcePtr source() {
        auto resource = std::make_shared<KVCacheResource>();
        resource->initGroups(config_.topologyPtr());
        resource->setCacheKeys(calculateCacheKeys(token_ids_.data(), token_ids_.size(), 4));
        resource->mutableBlockIds(0).assign({7, 8, 9, 10});
        return resource;
    }

    std::shared_ptr<P2PConnectorAsyncWriteContext> asyncWrite(KVCacheResourcePtr resource) {
        return decode_->asyncWrite(std::move(resource), token_ids_, input_length_, kv_ready_token_count_, routing_);
    }

    P2PConnectorStartWriteRequestPB request() {
        P2PConnectorStartWriteRequestPB request;
        request.set_unique_key(routing_.unique_key);
        request.set_request_id(17);
        request.set_deadline_ms(currentTimeMs() + 4000);
        request.set_decode_tp_size(2);
        request.set_input_length(4);
        request.set_layout_digest(ShardLayoutFactory::writebackLayoutDigest(*config_.topologyPtr(), 2));
        for (const auto token : token_ids_) {
            request.add_token_ids(token);
        }
        for (const auto key : keys_) {
            request.add_cache_keys(key);
        }
        return request;
    }

    void completeAll(bool draining = false) {
        for (int i = 0; i < 2; ++i) {
            prefill_workers_[i].completeAll(!draining, draining);
            decode_workers_[i].completeAll(!draining, draining);
        }
    }

    P2PConnectorStartWriteResponsePB handleWriteWhileAdmissionBlocked(bool expire) {
        auto input = request();
        if (expire) {
            input.set_deadline_ms(currentTimeMs() + 1000);
        }
        P2PConnectorStartWriteResponsePB response;
        std::atomic<bool>               cancelled{false};
        std::promise<void>              before_admission;
        auto                            reached_admission = before_admission.get_future();
        std::unique_lock<std::mutex>    tree_lock(allocator_->blockTreeCacheOwner()->mutex_);
        auto handling = std::async(std::launch::async, [&]() {
            int checks = 0;
            prefill_->handleWrite(input, response, [&]() {
                // Keep the pre-admission check successful while the tree lock delays admission.
                if (++checks == 2) {
                    before_admission.set_value();
                    return false;
                }
                return cancelled.load();
            });
        });
        const auto reached = reached_admission.wait_for(std::chrono::seconds(2));
        EXPECT_EQ(reached, std::future_status::ready);
        if (reached == std::future_status::ready) {
            if (expire) {
                EXPECT_TRUE(await([&]() { return currentTimeMs() >= input.deadline_ms(); }));
            } else {
                cancelled = true;
            }
        }
        tree_lock.unlock();
        handling.get();
        return response;
    }

    void prepareLocalLoad(block_tree_cache_test::TransferCopyAction action, bool fully_cached = false) {
        prefill_.reset();
        allocator_ = std::make_shared<Allocator>(config_);
        KVCacheConfig tiered;
        tiered.enable_host_cache  = true;
        tiered.host_cache_size_mb = 1;
        allocator_->setBlockTreeCacheConfigForTest(tiered);
        ASSERT_TRUE(allocator_->init());
        ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_[0]}).success);
        const auto                                 cache = allocator_->blockTreeCacheOwner();
        const auto                                 group = cache->groupSets().front();
        const size_t                               count = fully_cached ? keys_.size() : keys_.size() - 1;
        std::vector<std::vector<GroupSetResource>> slots(count, std::vector<GroupSetResource>(1));
        for (size_t i = 1; i < count; ++i) {
            slots[i][0].host_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_NE(slots[i][0].host_block, NULL_BLOCK_IDX);
        }
        cache->tree()->insertNode(CacheKeysType(keys_.begin(), keys_.begin() + count), slots, /*collect_path=*/false);
        for (size_t i = 1; i < count; ++i) {
            group->releaseSingleBlock(Tier::HOST, slots[i][0].host_block, BlockTreeRefType::CACHE);
        }
        load_barrier_ = std::make_shared<block_tree_cache_test::CallbackBarrier>();
        load_engine_  = std::make_shared<block_tree_cache_test::ControlledPerRankBlockTransferEngine>(
            cache->groupSets(), action, load_barrier_);
        cache->transfer_dispatcher_->per_rank_engine_ = load_engine_;
        prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
        ASSERT_TRUE(prefill_->init());
    }

    std::shared_ptr<P2PConnectorAsyncWriteContext> receiveContext() {
        std::lock_guard<std::mutex> lock(prefill_->checker_.mutex_);
        const auto                  it = prefill_->checker_.contexts_.find(routing_.unique_key);
        return it == prefill_->checker_.contexts_.end() ? nullptr : it->second;
    }

    static bool transferStopped(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context) {
        std::lock_guard<std::mutex> lock(context->mutex_);
        return context->transfer_stopped_;
    }

    PlanResult mirroredWritePlan() const {
        const auto src = ShardLayoutFactory::fromTopology(
            *config_.topologyPtr(), decode_config_.parallelism_config, RoleType::DECODE);
        const auto dst = ShardLayoutFactory::peerOf(src, 2, false, RoleType::PREFILL);
        return KVCacheTransferPlanner::plan(src, dst, ShardLayoutFactory::tagsOf(*config_.topologyPtr()));
    }

    CacheConfig                                config_;
    P2PConnectorSchedulerConfig                prefill_config_, decode_config_;
    std::vector<int>                           token_ids_;
    int                                        input_length_{0};
    size_t                                     kv_ready_token_count_{0};
    Meta::P2PRoutingContext                    routing_;
    CacheKeysType                              keys_;
    std::shared_ptr<Allocator>                 allocator_;
    WriteSchedulerService                      prefill_workers_[2], decode_workers_[2];
    std::vector<std::unique_ptr<grpc::Server>> servers_;
    std::vector<int>                           ports_;
    std::vector<std::string>                   prefill_addrs_, decode_addrs_;
    std::shared_ptr<P2PBroadcastClient>        prefill_client_, decode_client_;
    std::unique_ptr<P2PSchedulerPrefillWrite>  prefill_;
    std::unique_ptr<P2PSchedulerDecodeWrite>   decode_;
    std::shared_ptr<block_tree_cache_test::CallbackBarrier>                      load_barrier_;
    std::shared_ptr<block_tree_cache_test::ControlledPerRankBlockTransferEngine> load_engine_;
};

TEST_F(P2PConnectorWriteSchedulerTest, MirroredRoutesAndAllRanksRequiredBeforePublicationAndSourceRelease) {
    const auto expected = mirroredWritePlan();
    ASSERT_TRUE(expected.ok());
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    for (int rank = 0; rank < 2; ++rank) {
        const auto send    = decode_workers_[rank].lastStart();
        const auto receive = prefill_workers_[rank].lastStart();
        EXPECT_EQ(send.type(), WRITE);
        EXPECT_EQ(receive.type(), HANDLE_WRITE);
        EXPECT_EQ(send.plan_digest(), expected.plan.digest());
        EXPECT_EQ(send.deadline_ms(), receive.deadline_ms());
        EXPECT_EQ(send.plan_digest(), receive.plan_digest());
        ASSERT_EQ(send.routes_size(), 1);
        ASSERT_EQ(receive.routes_size(), 1);
        EXPECT_EQ(send.routes(0).route_id(), receive.routes(0).route_id());
        ASSERT_EQ(send.routes(0).layer_blocks_size(), 2);
        for (int layer = 0; layer < 2; ++layer) {
            const auto& s = send.routes(0).layer_blocks(layer);
            const auto& r = receive.routes(0).layer_blocks(layer);
            ASSERT_EQ(s.cache_keys_size(), 2);
            EXPECT_EQ(std::vector<int64_t>(s.cache_keys().begin(), s.cache_keys().end()),
                      std::vector<int64_t>(r.cache_keys().begin(), r.cache_keys().end()));
            for (int i = 0; i < s.cache_keys_size(); ++i) {
                EXPECT_EQ(s.block_ids(i), s.cache_keys(i) == keys_[1] ? 8 : 9);
            }
        }
    }
    prefill_workers_[0].completeAll();
    decode_workers_[0].completeAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    completeAll();
    ASSERT_TRUE(await(
        [&]() { return context->done() && !context->resourceHoldPending() && prefill_->inflightContextCount() == 0; }));
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, KvReadyBoundaryCapsCompleteTokenKeys) {
    kv_ready_token_count_ = 8;
    auto context          = asyncWrite(source());
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[1].starts == 1; }));
    const auto handshake = prefill_workers_[0].lastHandshake();
    EXPECT_EQ(handshake.token_ids_size(), 13);
    ASSERT_EQ(handshake.cache_keys_size(), 2);
    EXPECT_EQ(handshake.cache_keys(1), keys_[1]);
    EXPECT_EQ(decode_workers_[0].lastStart().routes(0).layer_blocks(0).cache_keys_size(), 1);
    completeAll();
    ASSERT_TRUE(await([&]() { return context->done() && prefill_->inflightContextCount() == 0; }));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 2u);
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeSkipsWorkersAndReleasesSource) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    const auto                     free    = allocator_->freeBlocksNum();
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeRejectsDeadlineReachedDuringAdmission) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    const auto free     = allocator_->freeBlocksNum();
    const auto response = handleWriteWhileAdmissionBlocked(/*expire=*/true);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::GENERATE_TIMEOUT));
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeRejectsCancellationDuringAdmission) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    const auto free     = allocator_->freeBlocksNum();
    const auto response = handleWriteWhileAdmissionBlocked(/*expire=*/false);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::CANCELLED));
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyLocalHandshakeCancellationAbortsUnsubmittedLoad) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed, true);
    const auto free     = allocator_->freeBlocksNum();
    const auto response = handleWriteWhileAdmissionBlocked(/*expire=*/false);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::CANCELLED));
    EXPECT_EQ(load_engine_->submittedBatchCount(), 0u);
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, LocalLoadRunsAlongsideReceiveAndPublicationWaitsForBoth) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = receive->resource()->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    EXPECT_EQ(pool->refCount(blocks[0]), 2u);
    for (int rank = 0; rank < 2; ++rank) {
        const auto request = decode_workers_[rank].lastStart();
        ASSERT_EQ(request.routes_size(), 1);
        for (const auto& layer : request.routes(0).layer_blocks()) {
            ASSERT_EQ(layer.cache_keys_size(), 1);
            EXPECT_EQ(layer.cache_keys(0), keys_[2]);
        }
    }
    completeAll();
    ASSERT_TRUE(await([&]() { return transferStopped(receive); }));
    EXPECT_FALSE(receive->done());
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_TRUE(receive->success()) << receive->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
    EXPECT_EQ(pool->refCount(blocks[0]), 1u);
    EXPECT_EQ(pool->refCount(blocks[1]), 1u);
    EXPECT_EQ(pool->refCount(blocks[2]), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LocalLoadFinishesFirstButReceiveStillHoldsResources) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = receive->resource()->blocks(0);
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return allocator_->devicePrefixBlocksForTest(keys_) == 2; }));
    EXPECT_FALSE(receive->done());
    EXPECT_EQ(allocator_->getDeviceBlockPool()->refCount(blocks[1]), 2u);
    completeAll();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_TRUE(receive->success());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LocalLoadFailureCancelsReceiveAndDoesNotPublishTail) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Fail);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = receive->resource()->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return prefill_workers_[0].cancels > 0; }));
    EXPECT_TRUE(receive->done());
    EXPECT_FALSE(receive->success());
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_TRUE(pool->isAllocated(blocks[2]));
    completeAll();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    EXPECT_FALSE(pool->isAllocated(blocks[1]));
    EXPECT_FALSE(pool->isAllocated(blocks[2]));
    EXPECT_EQ(pool->refCount(blocks[0]), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, CancellationKeepsResourceUntilLocalLoadAlsoStops) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = receive->resource()->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    receive->cancel();
    completeAll();
    ASSERT_TRUE(await([&]() { return transferStopped(receive); }));
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_TRUE(pool->isAllocated(blocks[1]));
    EXPECT_TRUE(pool->isAllocated(blocks[2]));
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_FALSE(receive->success());
    EXPECT_FALSE(pool->isAllocated(blocks[2]));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 2u);
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyLocalSuffixSkipsWorkersButWaitsForLoad) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed, true);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return context->done() && load_engine_->submittedBatchCount() == 1; }));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(decode_workers_[0].starts, 0);
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    EXPECT_FALSE(receive->done());
    EXPECT_TRUE(receive->resourceHoldPending());
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_TRUE(receive->success()) << receive->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, ConcurrentWritesJoinLocalLoadAndPublishIdempotently) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto first = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto first_receive = receiveContext();
    ASSERT_NE(first_receive, nullptr);
    const auto first_blocks = first_receive->resource()->blocks(0);

    routing_.unique_key = "second-write-scheduler";
    auto second         = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 2 && decode_workers_[1].starts == 2; }));
    auto second_receive = receiveContext();
    ASSERT_NE(second_receive, nullptr);
    const auto second_blocks = second_receive->resource()->blocks(0);
    EXPECT_EQ(first_blocks[0], second_blocks[0]);
    EXPECT_EQ(first_blocks[1], second_blocks[1]);
    EXPECT_NE(first_blocks[2], second_blocks[2]);
    EXPECT_EQ(load_engine_->submittedBatchCount(), 1u);
    const auto pool = allocator_->getDeviceBlockPool();
    EXPECT_EQ(pool->refCount(first_blocks[0]), 3u);
    EXPECT_EQ(pool->refCount(first_blocks[1]), 3u);

    completeAll();
    load_barrier_->release();
    ASSERT_TRUE(
        await([&]() { return !first_receive->resourceHoldPending() && !second_receive->resourceHoldPending(); }));
    EXPECT_TRUE(first_receive->success()) << first_receive->errorInfo().ToString();
    EXPECT_TRUE(second_receive->success()) << second_receive->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
    EXPECT_EQ(pool->refCount(first_blocks[0]), 1u);
    EXPECT_EQ(pool->refCount(first_blocks[1]), 1u);
    EXPECT_NE(pool->isAllocated(first_blocks[2]), pool->isAllocated(second_blocks[2]));
}

TEST_F(P2PConnectorWriteSchedulerTest, InvalidPrefillAdmissionDoesNotAllocateOrRegister) {
    const auto                                                               free      = allocator_->freeBlocksNum();
    const std::vector<std::function<void(P2PConnectorStartWriteRequestPB&)>> mutations = {
        [](auto& r) { r.set_layout_digest(r.layout_digest() + 1); },
        [](auto& r) { r.set_decode_tp_size(1); },
        [](auto& r) { r.set_cache_keys(2, 999); },
        [](auto& r) { r.add_cache_keys(999); },
        [](auto& r) { r.set_input_length(100); },
        [](auto& r) { r.set_deadline_ms(currentTimeMs() - 1); },
        [](auto& r) { r.set_input_length(12); }};
    for (const auto& mutate : mutations) {
        auto input = request();
        mutate(input);
        P2PConnectorStartWriteResponsePB response;
        prefill_->handleWrite(input, response);
        EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR) << input.DebugString();
    }
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, LostDecodeStartResponseRetainsSourceUntilEveryRankStops) {
    decode_workers_[1].fail_start          = true;
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(
        await([&]() { return context->done() && decode_workers_[0].cancels > 0 && decode_workers_[1].cancels > 0; }));
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(weak.expired());
    EXPECT_TRUE(context->resourceHoldPending());
    completeAll();
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending() && weak.expired(); }));
    EXPECT_FALSE(context->success());
}

TEST_F(P2PConnectorWriteSchedulerTest, PartialReceiveStartFailureOutlivesHandshakeAndBusinessDeadline) {
    prefill_workers_[1].fail_start = true;
    decode_.reset();
    decode_config_.p2p_writeback_timeout_ms = 200;
    createDecode();
    const auto                     free    = allocator_->freeBlocksNum();
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free - 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free - 2);
    completeAll();
    ASSERT_TRUE(await([&]() { return prefill_->inflightContextCount() == 0; }));
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LostHandshakeLeavesPrefillResponsibleForReceiveCleanup) {
    prefill_workers_[0].lose_handshake     = true;
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    completeAll();
    prefill_.reset();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, UnsupportedGroupsAndCpAreRejectedBeforeHandshake) {
    auto groups                         = config_.topologyPtr()->groups();
    groups[0].policy.group_type         = CacheGroupType::SWA;
    groups[0].policy.active_tail_blocks = 2;
    const auto unsupported              = CacheTopology::create(groups, config_.topologyPtr()->layers());
    EXPECT_TRUE(ShardLayoutFactory::validateWritebackLayout(*unsupported, decode_config_.parallelism_config, 2)
                    .hasError());
    auto cp                               = decode_config_.parallelism_config;
    cp.prefill_cp_config.kv_cache_sharded = true;
    EXPECT_TRUE(ShardLayoutFactory::validateWritebackLayout(*config_.topologyPtr(), cp, 2).hasError());
    decode_.reset();
    decode_config_.topology = unsupported;
    createDecode();
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(prefill_workers_[0].handshakes, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, LastSampleAndMtpAcceptanceDoNotOverstateReadyKv) {
    for (const auto& [token_count, ready_count] :
         std::vector<std::pair<int, int>>{{7, 7}, {8, 7}, {9, 8}, {16, 15}, {20, 17}}) {
        token_ids_.clear();
        for (int i = 0; i < token_count; ++i) {
            token_ids_.push_back(i + 1);
        }
        kv_ready_token_count_         = ready_count;
        routing_.unique_key           = "boundary-" + std::to_string(token_count);
        prefill_workers_[0].handshake = [&](const auto& request, auto& response, auto) {
            EXPECT_EQ(request.cache_keys_size(), ready_count / 4);
            response.set_accepted_start_block(request.cache_keys_size());
            const auto plan = mirroredWritePlan();
            response.set_plan_digest(plan.plan.digest());
        };
        auto context = asyncWrite(source());
        ASSERT_TRUE(await([&]() { return context->done(); }));
        EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    }
    EXPECT_EQ(decode_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, InvalidWriteInputAndCancellationSkipHandshake) {
    auto context = decode_->asyncWrite(source(), token_ids_, input_length_, token_ids_.size() + 1, routing_);
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    auto invalid_cp_routing            = routing_;
    invalid_cp_routing.unique_key     += "-invalid-cp";
    invalid_cp_routing.prefill_cp_size = 0;
    context = decode_->asyncWrite(
        source(), token_ids_, input_length_, kv_ready_token_count_, std::move(invalid_cp_routing));
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    auto owner            = source();
    owner->cacheKeys()[1] = 999;
    auto bad_keys_routing = routing_;
    bad_keys_routing.unique_key += "-bad-keys";
    context =
        decode_->asyncWrite(
            std::move(owner), token_ids_, input_length_, kv_ready_token_count_, std::move(bad_keys_routing));
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(prefill_workers_[0].handshakes, 0);
    auto                             req = request();
    P2PConnectorStartWriteResponsePB response;
    prefill_->handleWrite(req, response, []() { return true; });
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, MultipleFullGroupsRollBackEarlierAllocationsOnFailure) {
    auto config                        = config_;
    auto first                         = config.specForGroup(0)->clone();
    auto second                        = first->clone();
    first->tag                         = "first";
    second->tag                        = "second";
    config.use_independent_block_pools = true;
    config.fromGroupedSpecs(
        {first, second}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::FULL}, {"first", "second"});
    config.setGroupBlockLayout({8, 2}, {first->block_size_bytes(), second->block_size_bytes()}, {0, 0});
    using MultiAllocator = test::BlockTreeCacheTestAllocator<HybridPoolKVCacheAllocator>;
    auto allocator = std::make_shared<MultiAllocator>(config, AllocationType::DEVICE, nullptr, 0, RoleType::PREFILL);
    ASSERT_TRUE(allocator->init());
    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    const auto first_free  = pools[0]->freeBlocksNum();
    const auto second_free = pools[1]->freeBlocksNum();
    KVCacheResourcePtr                resource;
    size_t                            start = 0;
    std::shared_ptr<LoadAsyncContext> load_context;
    EXPECT_TRUE(allocator->admitWriteBackDecodeCache(keys_, 0, resource, start, load_context).hasError());
    EXPECT_EQ(resource, nullptr);
    EXPECT_EQ(pools[0]->freeBlocksNum(), first_free);
    EXPECT_EQ(pools[1]->freeBlocksNum(), second_free);
    EXPECT_TRUE(allocator->blockTreeCacheOwner()->tree()->findNode(keys_).empty());
}

TEST_F(P2PConnectorWriteSchedulerTest, InvalidReturnedRangesAndEndpointsNeverReachSenders) {
    const auto plan = mirroredWritePlan();
    const std::vector<std::function<void(P2PConnectorStartWriteResponsePB&)>> mutations = {
        [](auto& r) { r.set_accepted_start_block(-1); },
        [](auto& r) { r.set_accepted_block_count(-1); },
        [](auto& r) { r.set_accepted_block_count(3); },
        [](auto& r) {
            r.set_accepted_start_block(0);
            r.set_accepted_block_count(3);
        },
        [](auto& r) { r.set_accepted_block_count(0); },
        [](auto& r) { r.set_plan_digest(0); },
        [](auto& r) { r.clear_prefill_worker_transfer_addrs(); },
        [](auto& r) { r.set_prefill_worker_transfer_addrs(0, "bad-endpoint"); }};
    for (size_t i = 0; i < mutations.size(); ++i) {
        prefill_workers_[0].handshake = [&, i](const auto&, auto& response, auto) {
            response.set_accepted_start_block(1);
            response.set_accepted_block_count(2);
            response.set_plan_digest(plan.plan.digest());
            response.add_prefill_worker_transfer_addrs("127.0.0.1:12345");
            response.add_prefill_worker_transfer_addrs("[::1]:12345");
            mutations[i](response);
        };
        auto iteration_routing = routing_;
        iteration_routing.unique_key += std::to_string(i);
        auto                           owner   = source();
        std::weak_ptr<KVCacheResource> weak    = owner;
        auto                           context = decode_->asyncWrite(
            std::move(owner), token_ids_, input_length_, kv_ready_token_count_, std::move(iteration_routing));
        ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
        EXPECT_FALSE(context->success()) << i;
    }
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, SlowControlConnectionDoesNotBlockAdmissionOrDeadline) {
    auto first = std::make_shared<P2PConnectorAsyncWriteContext>(
        source(), "slow-control", currentTimeMs() + 200, WRITE, decode_client_, 100);
    auto second = std::make_shared<P2PConnectorAsyncWriteContext>(
        source(), "new-admission", currentTimeMs() + 4000, WRITE, decode_client_, 100);
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init(5));
    ASSERT_TRUE(first->beginKickoff());
    first->setCallResults(std::make_shared<P2PBroadcastClient::Result>("slow-control"));
    std::unique_lock<std::mutex> connection_lock(decode_client_->tp_broadcast_manager_->rpc_pool_->mutex_);
    ASSERT_TRUE(checker.addContext(first));
    EXPECT_TRUE(await([&]() {
        std::lock_guard<std::mutex> lock(first->mutex_);
        return first->control_submitting_;
    }));
    auto admission = std::async(std::launch::async, [&]() { return checker.addContext(second); });
    const auto admission_status = admission.wait_for(std::chrono::milliseconds(100));
    const bool expired = await([&]() { return first->done(); }, 1000);
    connection_lock.unlock();
    EXPECT_EQ(admission_status, std::future_status::ready);
    EXPECT_TRUE(admission.get());
    EXPECT_TRUE(expired);
    EXPECT_TRUE(first->resourceHoldPending());
    completeAll(true);
    checker.stop();
    EXPECT_FALSE(first->resourceHoldPending());
}

}  // namespace
}  // namespace rtp_llm
