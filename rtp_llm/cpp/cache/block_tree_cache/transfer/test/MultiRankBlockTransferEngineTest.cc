#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <chrono>
#include <csignal>
#include <mutex>
#include <sys/resource.h>
#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

struct MultiRankBlockTransferRpcState {
    std::mutex                            mutex;
    std::vector<MemoryOperationRequestPB> requests;
};

struct MultiRankBlockTransferRpcConfig {
    bool                                            has_mem_response;
    MemoryOperationResponsePB::Code                 mem_response_code;
    grpc::Status                                    rpc_status;
    std::shared_ptr<MultiRankBlockTransferRpcState> state{nullptr};
    int                                             sleep_millis{0};
};

class MultiRankBlockTransferRpcService final: public RpcService::Service {
public:
    explicit MultiRankBlockTransferRpcService(const MultiRankBlockTransferRpcConfig& config): config_(config) {}

    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        if (config_.sleep_millis > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.sleep_millis));
        }
        if (config_.state != nullptr && request->has_mem_request()) {
            std::lock_guard<std::mutex> lock(config_.state->mutex);
            config_.state->requests.push_back(request->mem_request());
        }
        if (config_.has_mem_response) {
            response->mutable_mem_response()->set_code(config_.mem_response_code);
        }
        return config_.rpc_status;
    }

private:
    MultiRankBlockTransferRpcConfig config_;
};

class MultiRankBlockTransferRpcServer {
public:
    explicit MultiRankBlockTransferRpcServer(std::unique_ptr<MultiRankBlockTransferRpcService> service):
        service_(std::move(service)) {}

    ~MultiRankBlockTransferRpcServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        return server_ != nullptr && listen_port_ > 0;
    }

    std::string address() const {
        return "127.0.0.1:" + std::to_string(listen_port_);
    }

private:
    std::unique_ptr<MultiRankBlockTransferRpcService> service_;
    std::unique_ptr<grpc::Server>                     server_;
    int                                               listen_port_{0};
};

static std::shared_ptr<BroadcastManager>
makeBroadcastManager(const std::vector<MultiRankBlockTransferRpcConfig>&            configs,
                     std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>>& servers) {
    std::vector<std::string> worker_addrs;
    worker_addrs.reserve(configs.size());
    servers.reserve(configs.size());
    for (const MultiRankBlockTransferRpcConfig& config : configs) {
        std::unique_ptr<MultiRankBlockTransferRpcService> service =
            std::make_unique<MultiRankBlockTransferRpcService>(config);
        std::unique_ptr<MultiRankBlockTransferRpcServer> server =
            std::make_unique<MultiRankBlockTransferRpcServer>(std::move(service));
        if (!server->start()) {
            return nullptr;
        }
        worker_addrs.push_back(server->address());
        servers.push_back(std::move(server));
    }

    std::shared_ptr<BroadcastManager> broadcast_manager = std::make_shared<BroadcastManager>(worker_addrs);
    if (!broadcast_manager->init()) {
        return nullptr;
    }
    return broadcast_manager;
}

static std::unique_ptr<BlockTreeCache> makeBroadcastCache(const std::shared_ptr<BroadcastManager>& broadcast_manager) {
    DeviceBlockPoolPtr            device_pool = makeDevicePool({{256, 0}}, 8, "multi_rank_engine_device");
    std::shared_ptr<FullGroupSet> full =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, makeHostPool(256, 8), nullptr);
    auto topology = block_transfer_engine_test::makeTestTopology(
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 256)});
    full->initialize(0, topology, {0});
    std::vector<GroupSetPtr> groups = {full};
    return makeBlockTreeCacheForTest(std::move(groups),
                                     BlockTreeCacheConfig{},
                                     /*storage_backend=*/nullptr,
                                     broadcast_manager);
}

static std::shared_ptr<FullGroupSet> makeBroadcastGroup(const std::string&             pool_name,
                                                        std::shared_ptr<HostBlockPool> host_pool = nullptr,
                                                        BlockTreeDiskBlockPoolPtr      disk_pool = nullptr) {
    auto device_pool = makeDevicePool({{256, 0}}, 8, pool_name);
    RTP_LLM_CHECK(device_pool != nullptr);
    return std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{std::move(device_pool)}, std::move(host_pool), std::move(disk_pool));
}

static BlockIdxType prepareDeviceTarget(const std::shared_ptr<FullGroupSet>& group) {
    const DeviceBlockPoolPtr& device_pool = group->devicePools().front();
    auto                      topology    = block_transfer_engine_test::makeTestTopology(
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 256)});
    group->initialize(0, topology, {0});
    const auto block = device_pool->malloc();
    if (!block.has_value()) {
        return NULL_BLOCK_IDX;
    }
    device_pool->incRef(*block);
    return *block;
}

static void initializeBroadcastGroups(const std::vector<std::shared_ptr<FullGroupSet>>& groups,
                                      size_t                                            payload_bytes = 256) {
    std::vector<GroupBase> group_bases;
    group_bases.reserve(groups.size());
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        group_bases.push_back(block_transfer_engine_test::makeTestGroupBase(
            defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_bytes));
    }
    auto topology = block_transfer_engine_test::makeTestTopology(std::move(group_bases));
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        groups[group_id]->initialize(group_id, topology, {group_id});
    }
}

static std::vector<TransferDescriptor> makeBroadcastDescriptors() {
    return {TransferDescriptor::hostToDevice(0, 1, {1})};
}

static void expectSingleGroupBlock(const MemoryOperationRequestPB::CopyItem& item,
                                   size_t                                    group_set_id,
                                   int                                       group_id,
                                   BlockIdxType                              block) {
    EXPECT_EQ(item.group_set_id(), group_set_id);
    ASSERT_EQ(item.group_blocks_size(), 1);
    EXPECT_EQ(item.group_blocks(0).group_id(), group_id);
    EXPECT_EQ(item.group_blocks(0).block_id(), block);
}

class MultiRankBlockTransferEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto                     full_group = makeBroadcastGroup("multi_rank_fixture");
        std::vector<GroupSetPtr> groups     = {full_group};
        cache_                              = makeBlockTreeCacheForTest(std::move(groups));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(MultiRankBlockTransferEngineTest, BroadcastManagerStoredCorrectly) {
    // Create a BroadcastManager (no actual RPC connections needed for this test)
    std::vector<std::string> worker_addrs  = {"127.0.0.1:50051", "127.0.0.1:50052"};
    auto                     broadcast_mgr = std::make_shared<BroadcastManager>(worker_addrs);
    ASSERT_TRUE(broadcast_mgr->init());

    auto                     full   = makeBroadcastGroup("broadcast_manager_stored");
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;

    auto cache = makeBlockTreeCacheForTest(std::move(groups), std::move(cfg), nullptr, broadcast_mgr);

    // Verify BroadcastManager is stored (access via internal member)
    EXPECT_EQ(cache->transfer_dispatcher_->multi_rank_engine_->broadcast_manager_, broadcast_mgr);
    EXPECT_EQ(cache->transfer_dispatcher_->multi_rank_engine_->broadcast_manager_->workerNum(), 2u);
}

bool executeAndWait(MultiRankBlockTransferEngine&                 engine,
                    const std::vector<TransferDescriptor>& descriptors,
                    int                                    timeout_ms) {
    auto context = engine.execute(descriptors, timeout_ms);
    context->waitDone();
    return context->success();
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferSucceedsForAllWorkers) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_TRUE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, ExecuteReturnsBeforeSlowWorkersFinish) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, nullptr, /*sleep_millis=*/300},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    auto broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    auto cache = makeBroadcastCache(broadcast_manager);

    auto context = cache->transfer_dispatcher_->multi_rank_engine_->execute(
        makeBroadcastDescriptors(), /*timeout_ms=*/1000);

    EXPECT_FALSE(context->done());
    context->waitDone();
    EXPECT_TRUE(context->success());
}

TEST_F(MultiRankBlockTransferEngineTest, PollingDoneSettlesSuccessAndErrorState) {
    for (const auto response_code : {MemoryOperationResponsePB::OK, MemoryOperationResponsePB::FAILED}) {
        const std::vector<MultiRankBlockTransferRpcConfig> configs = {
            {true, response_code, grpc::Status::OK, nullptr, /*sleep_millis=*/10},
        };
        std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
        auto broadcast_manager = makeBroadcastManager(configs, servers);
        ASSERT_NE(broadcast_manager, nullptr);
        auto cache = makeBroadcastCache(broadcast_manager);
        auto context = cache->transfer_dispatcher_->multi_rank_engine_->execute(
            makeBroadcastDescriptors(), /*timeout_ms=*/500);

        // BroadcastResult completion is driven by waitDone(). Exercise the public
        // polling contract concurrently and verify that observing done also means
        // the success/error state has already been settled.
        std::thread waiter([&context]() { context->waitDone(); });
        const auto  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool        observed_done = false;
        while (!(observed_done = context->done()) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        waiter.join();

        ASSERT_TRUE(observed_done);
        EXPECT_EQ(context->success(), response_code == MemoryOperationResponsePB::OK);
        EXPECT_EQ(context->errorInfo().ok(), response_code == MemoryOperationResponsePB::OK);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferFailsWithoutDispatchOnInvalidBatch) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(*cache->transfer_dispatcher_->multi_rank_engine_, {}, /*timeout_ms=*/500));
    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/0));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferReportsWorkerRpcError) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::OK, grpc::Status(grpc::StatusCode::INTERNAL, "worker failed")},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferReportsRpcDeadline) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, nullptr, /*sleep_millis=*/500},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/50));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferFailsOnWorkerBusinessError) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferFailsWhenAllWorkersReportBusinessError) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK},
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferWaitsForEveryRankBeforeReportingBusinessError) {
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state, /*sleep_millis=*/300},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    const auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/5000));
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    EXPECT_GE(elapsed_ms, 250) << "controller must not report failure before the slow rank finishes";
    std::lock_guard<std::mutex> lock(state->mutex);
    EXPECT_EQ(state->requests.size(), 2u);
}

struct InvalidWorkerResponse {
    bool                            has_response;
    MemoryOperationResponsePB::Code code;
};

class MultiRankBlockTransferInvalidResponseTest:
    public ::testing::TestWithParam<InvalidWorkerResponse> {};

TEST_P(MultiRankBlockTransferInvalidResponseTest, BroadcastTransferRejectsInvalidWorkerResponse) {
    const auto param = GetParam();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {param.has_response, param.code, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidWorkerResponses,
    MultiRankBlockTransferInvalidResponseTest,
    ::testing::Values(InvalidWorkerResponse{false, MemoryOperationResponsePB::CODE_UNSPECIFIED},
                      InvalidWorkerResponse{true, MemoryOperationResponsePB::CODE_UNSPECIFIED},
                      InvalidWorkerResponse{true, static_cast<MemoryOperationResponsePB::Code>(42)}));

class MultiRankBlockTransferGrpcStatusTest: public ::testing::TestWithParam<grpc::StatusCode> {};

TEST_P(MultiRankBlockTransferGrpcStatusTest, BroadcastTransferReportsAnyNonOkStatus) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::OK, grpc::Status(GetParam(), "worker failed")},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

INSTANTIATE_TEST_SUITE_P(NonOkWorkerStatuses,
                         MultiRankBlockTransferGrpcStatusTest,
                         ::testing::Values(grpc::StatusCode::CANCELLED,
                                           grpc::StatusCode::INTERNAL,
                                           grpc::StatusCode::UNAVAILABLE,
                                           grpc::StatusCode::RESOURCE_EXHAUSTED));

static void disableCoreDump() {
    rlimit no_core;
    no_core.rlim_cur = 0;
    no_core.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &no_core);
}

static void broadcastWithNonOkWorkerStatus() {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::OK, grpc::Status(grpc::StatusCode::UNAVAILABLE, "worker gone")},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    disableCoreDump();
    StaticConfig::user_ft_core_dump_on_exception = true;
    (void)executeAndWait(
        *cache->transfer_dispatcher_->multi_rank_engine_, makeBroadcastDescriptors(), /*timeout_ms=*/500);
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferAbortsWithSigabrtWhenCoreDumpEnabled) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    // RTP_LLM_FAIL logs through the project logger, not the child's stderr, so only the
    // exit signal is assertable here.
    EXPECT_EXIT(broadcastWithNonOkWorkerStatus(), ::testing::KilledBySignal(SIGABRT), "");
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastHostLoadCommitsDeviceResource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool> host_pool    = makeHostPool(256, 4);
    std::shared_ptr<FullGroupSet>  group        = makeBroadcastGroup("broadcast_host_load_success", host_pool);
    const BlockIdxType             device_block = prepareDeviceTarget(group);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_host_cache                          = true;
    std::vector<GroupSetPtr>                   groups = {group};
    std::unique_ptr<BlockTreeCache>            cache  = makeBlockTreeCacheForTest(std::move(groups),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].host_block = host_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    BlockTreeMatchResult              match   = cache->match({100});
    std::shared_ptr<LoadAsyncContext> context = std::dynamic_pointer_cast<LoadAsyncContext>(match.async_context);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    context->setTargetBlocks(0, {device_block});
    ASSERT_TRUE(context->commit());
    context->waitDone();
    ASSERT_TRUE(context->success());
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto find_result = cache->tree()->findNode({100});
    ASSERT_FALSE(find_result.empty());
    const GroupSetResource& resource = find_result.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
    releaseDeviceBlocks(*cache, group->devicePools().front(), {device_block});
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2D);
        EXPECT_EQ(worker_request.copy_items(0).mem_block(), host_block);
        expectSingleGroupBlock(worker_request.copy_items(0), 0, 0, device_block);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastHostLoadFailureKeepsSourceResource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool> host_pool    = makeHostPool(256, 4);
    std::shared_ptr<FullGroupSet>  group        = makeBroadcastGroup("broadcast_host_load_failure", host_pool);
    const BlockIdxType             device_block = prepareDeviceTarget(group);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_host_cache                          = true;
    std::vector<GroupSetPtr>                   groups = {group};
    std::unique_ptr<BlockTreeCache>            cache  = makeBlockTreeCacheForTest(std::move(groups),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].host_block = host_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    BlockTreeMatchResult              match   = cache->match({100});
    std::shared_ptr<LoadAsyncContext> context = std::dynamic_pointer_cast<LoadAsyncContext>(match.async_context);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    context->setTargetBlocks(0, {device_block});
    ASSERT_TRUE(context->commit());
    context->waitDone();
    ASSERT_FALSE(context->success());
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto find_result = cache->tree()->findNode({100});
    ASSERT_FALSE(find_result.empty());
    const GroupSetResource& resource = find_result.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_TRUE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(resource.host_block, host_block);
    EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    EXPECT_EQ(host_pool->freeBlocksNum(), 3u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 1u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2D);
        EXPECT_EQ(worker_request.copy_items(0).mem_block(), host_block);
        expectSingleGroupBlock(worker_request.copy_items(0), 0, 0, device_block);
    }
    group->devicePools().front()->decRef(device_block);
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastDiskLoadUsesSingleDirectStage) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 4);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 4, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullGroupSet>           group     = makeBroadcastGroup("broadcast_disk_load", host_pool, disk_pool);
    const BlockIdxType                      device_block = prepareDeviceTarget(group);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    const BlockIdxType disk_block = group->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_host_cache                          = true;
    config.enable_disk_cache                          = true;
    std::vector<GroupSetPtr>                   groups = {group};
    std::unique_ptr<BlockTreeCache>            cache  = makeBlockTreeCacheForTest(std::move(groups),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].disk_slot = disk_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    BlockTreeMatchResult              match   = cache->match({100});
    std::shared_ptr<LoadAsyncContext> context = std::dynamic_pointer_cast<LoadAsyncContext>(match.async_context);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    context->setTargetBlocks(0, {device_block});
    ASSERT_TRUE(context->commit());
    context->waitDone();
    ASSERT_TRUE(context->success());
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto find_result = cache->tree()->findNode({100});
    ASSERT_FALSE(find_result.empty());
    const GroupSetResource& resource = find_result.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::DISK));
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
    releaseDeviceBlocks(*cache, group->devicePools().front(), {device_block});
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::DISK2D);
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        const MemoryOperationRequestPB::CopyItem& request_item = worker_request.copy_items(0);
        EXPECT_EQ(request_item.disk_block(), disk_block);
        expectSingleGroupBlock(request_item, 0, 0, device_block);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastEvictionSuccessCommitsTask) {
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 8);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullGroupSet> full = makeBroadcastGroup("broadcast_eviction_success", host_pool, disk_pool);
    initializeBroadcastGroups({full});
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_device_cache             = false;
    config.enable_host_cache               = true;
    config.enable_disk_cache               = true;
    std::vector<GroupSetPtr>        groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].host_block = host_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    auto before = cache->tree()->findNode({100});
    ASSERT_FALSE(before.empty());
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto after = cache->tree()->findNode({100});
    ASSERT_FALSE(after.empty());
    const GroupSetResource& resource = after.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_TRUE(resource.hasTier(Tier::DISK));
    const BlockIdxType disk_slot = resource.disk_slot;
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2DISK);
        EXPECT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_items(0).mem_block(), host_block);
        EXPECT_EQ(worker_request.copy_items(0).disk_block(), disk_slot);
        EXPECT_EQ(worker_request.copy_items(0).group_set_id(), 0u);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastDeviceEvictionBypassesHostWithD2Disk) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto                                               state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    auto broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    auto full      = makeBroadcastGroup("broadcast_device_to_disk", nullptr, disk_pool);
    initializeBroadcastGroups({full});
    MultiNodeBlocks device = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(device.size(), 1u);
    ASSERT_EQ(device.front().size(), 1u);
    const BlockIdxType device_block = device.front().front();

    BlockTreeCacheConfig config;
    config.enable_device_cache                        = true;
    config.enable_host_cache                          = false;
    config.enable_disk_cache                          = true;
    std::vector<GroupSetPtr>                   groups = {full};
    auto                                       cache  = makeBlockTreeCacheForTest(std::move(groups),
                                           std::move(config),
                                           /*storage_backend=*/nullptr,
                                           broadcast_manager);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = device.front();
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    releaseDeviceBlocks(*cache, full->devicePools().front(), device.front());

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto after = cache->tree()->findNode({100});
    ASSERT_FALSE(after.empty());
    const GroupSetResource& resource = after.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    EXPECT_TRUE(resource.hasTier(Tier::DISK));

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& request : state->requests) {
        EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2DISK);
        ASSERT_EQ(request.copy_items_size(), 1);
        EXPECT_EQ(request.copy_items(0).disk_block(), resource.disk_slot);
        expectSingleGroupBlock(request.copy_items(0), 0, 0, device_block);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastD2DiskFailureRollsBackDeviceSource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto                                               state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK, state},
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    auto broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    auto full      = makeBroadcastGroup("broadcast_device_to_disk_failure", nullptr, disk_pool);
    initializeBroadcastGroups({full});
    MultiNodeBlocks device = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(device.size(), 1u);
    ASSERT_EQ(device.front().size(), 1u);
    const BlockIdxType device_block = device.front().front();

    BlockTreeCacheConfig config;
    config.enable_device_cache                        = true;
    config.enable_host_cache                          = false;
    config.enable_disk_cache                          = true;
    std::vector<GroupSetPtr>                   groups = {full};
    auto                                       cache  = makeBlockTreeCacheForTest(std::move(groups),
                                           std::move(config),
                                           /*storage_backend=*/nullptr,
                                           broadcast_manager);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = device.front();
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    releaseDeviceBlocks(*cache, full->devicePools().front(), device.front());

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto after = cache->tree()->findNode({100});
    ASSERT_FALSE(after.empty());
    const GroupSetResource& resource = after.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_FALSE(resource.hasTier(Tier::DISK));
    EXPECT_TRUE(full->devicePools().front()->isAllocated(device_block));
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& request : state->requests) {
        EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2DISK);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastEvictionFailureRollsBackTask) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, MemoryOperationResponsePB::OK, grpc::Status::OK},
        {true, MemoryOperationResponsePB::FAILED, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 8);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullGroupSet> full = makeBroadcastGroup("broadcast_eviction_failure", host_pool, disk_pool);
    initializeBroadcastGroups({full});
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_device_cache             = false;
    config.enable_host_cache               = true;
    config.enable_disk_cache               = true;
    std::vector<GroupSetPtr>        groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].host_block = host_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    auto before = cache->tree()->findNode({100});
    ASSERT_FALSE(before.empty());
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto after = cache->tree()->findNode({100});
    ASSERT_FALSE(after.empty());
    const GroupSetResource& resource = after.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_TRUE(resource.hasTier(Tier::HOST));
    EXPECT_FALSE(resource.hasTier(Tier::DISK));
    EXPECT_EQ(resource.host_block, host_block);
    EXPECT_EQ(host_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 1u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 0u);
}

TEST_F(MultiRankBlockTransferEngineTest, EncodeEvictionTransferRequestIncludesPrimaryAndCascades) {
    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 8);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullGroupSet> primary_group       = makeBroadcastGroup("broadcast_primary", host_pool, disk_pool);
    std::shared_ptr<FullGroupSet> cascade_group       = makeBroadcastGroup("broadcast_cascade", host_pool, disk_pool);
    initializeBroadcastGroups({primary_group, cascade_group});
    std::vector<GroupSetPtr>        groups = {primary_group, cascade_group};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups));
    ASSERT_NE(cache, nullptr);

    EvictionTask task;
    task.primary_desc.group_set_id  = 0;
    task.primary_desc.source_tier   = Tier::HOST;
    task.primary_desc.target_tier   = Tier::DISK;
    task.primary_desc.source_blocks = {3};
    task.primary_desc.target_blocks = {4};

    TransferDescriptor cascade_desc;
    cascade_desc.group_set_id  = 1;
    cascade_desc.source_tier   = Tier::HOST;
    cascade_desc.target_tier   = Tier::DISK;
    cascade_desc.source_blocks = {5};
    cascade_desc.target_blocks = {6};
    task.cascade_descs.push_back(cascade_desc);

    std::vector<TransferDescriptor> descriptors{task.primary_desc, task.cascade_descs.front()};
    MemoryOperationRequestPB request;
    ASSERT_TRUE(BlockTransferRequestConverter::encodeTransfer(request, descriptors, cache->groupSets()));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(request.copy_items(0).mem_block(), 3);
    EXPECT_EQ(request.copy_items(0).disk_block(), 4);
    EXPECT_EQ(request.copy_items(0).group_set_id(), 0u);
    EXPECT_EQ(request.copy_items(1).mem_block(), 5);
    EXPECT_EQ(request.copy_items(1).disk_block(), 6);
    EXPECT_EQ(request.copy_items(1).group_set_id(), 1u);
}

}  // namespace
}  // namespace rtp_llm
