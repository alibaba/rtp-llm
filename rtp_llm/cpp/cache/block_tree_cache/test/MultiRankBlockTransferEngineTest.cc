#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <mutex>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/PerRankBlockTransferEngineTestUtils.h"
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
    bool                                            mem_response_success;
    grpc::Status                                    rpc_status;
    std::shared_ptr<MultiRankBlockTransferRpcState> state{nullptr};
};

class MultiRankBlockTransferRpcService final: public RpcService::Service {
public:
    explicit MultiRankBlockTransferRpcService(const MultiRankBlockTransferRpcConfig& config): config_(config) {}

    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        if (config_.state != nullptr && request->has_mem_request()) {
            std::lock_guard<std::mutex> lock(config_.state->mutex);
            config_.state->requests.push_back(request->mem_request());
        }
        if (config_.has_mem_response) {
            response->mutable_mem_response()->set_success(config_.mem_response_success);
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
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setHostPool(makeHostPool(256, 8));
    DeviceBlockPoolPtr device_pool = makeDevicePool({{256, 0}}, 8, "multi_rank_engine_device");
    full->setDevicePools({device_pool}, {"tag_0"});
    std::vector<Component> components = {block_transfer_engine_test::makeSchemaComponent(0, 0, "tag_0", {256})};
    setComponentGroupLayoutForTest(*full, {0}, components);
    std::vector<ComponentGroupPtr> groups = {full};
    return makeBlockTreeCacheForTest(std::move(tree),
                                     std::move(groups),
                                     std::move(components),
                                     BlockTreeCacheConfig{},
                                     /*storage_backend=*/nullptr,
                                     broadcast_manager);
}

static BlockIdxType prepareDeviceTarget(const std::shared_ptr<FullComponentGroup>& group,
                                        const std::string&                         pool_name) {
    DeviceBlockPoolPtr device_pool = makeDevicePool({{256, 0}}, 8, pool_name);
    const auto         block       = device_pool->malloc();
    if (!block.has_value()) {
        return NULL_BLOCK_IDX;
    }
    device_pool->incRef(*block, BlockRefType::REQUEST);
    group->setDevicePools({device_pool}, {"tag_0"});
    return *block;
}

static void sealBroadcastLayout(const std::shared_ptr<FullComponentGroup>& group,
                                std::vector<Component>&                    components,
                                size_t                                     payload_bytes = 256) {
    if (group->devicePoolCount() == 0) {
        const std::string tag = "tag_" + std::to_string(group->component_group_id);
        group->setDevicePools(
            {makeDevicePool({{payload_bytes, 0}}, 8, "broadcast_layout_" + std::to_string(group->component_group_id))},
            {tag});
    }
    const int         component_index = static_cast<int>(components.size());
    const std::string component_tag   = group->tags().front();
    components.push_back(block_transfer_engine_test::makeSchemaComponent(
        component_index, group->component_group_id, component_tag, {payload_bytes}));
    setComponentGroupLayoutForTest(*group, {component_index}, components);
}

static std::vector<TransferDescriptor> makeBroadcastDescriptors() {
    return {TransferDescriptor::hostToDevice(0, 1, {1})};
}

static void
expectSingleTaggedBlock(const MemoryOperationRequestPB::CopyItem& item, const std::string& tag, BlockIdxType block) {
    ASSERT_EQ(item.component_group_tags_size(), 1);
    EXPECT_EQ(item.component_group_tags(0), tag);
    ASSERT_EQ(item.tagged_gpu_blocks_size(), 1);
    EXPECT_EQ(item.tagged_gpu_blocks(0).tag(), tag);
    EXPECT_EQ(item.tagged_gpu_blocks(0).block_id(), block);
}

class MultiRankBlockTransferEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(1);
        auto full_group                       = std::make_shared<FullComponentGroup>();
        full_group->component_group_id        = 0;
        std::vector<ComponentGroupPtr> groups = {full_group};
        cache_ = makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::vector<Component>{});
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(MultiRankBlockTransferEngineTest, BroadcastManagerStoredCorrectly) {
    // Create a BroadcastManager (no actual RPC connections needed for this test)
    std::vector<std::string> worker_addrs  = {"127.0.0.1:50051", "127.0.0.1:50052"};
    auto                     broadcast_mgr = std::make_shared<BroadcastManager>(worker_addrs);
    ASSERT_TRUE(broadcast_mgr->init());

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;

    auto cache = makeBlockTreeCacheForTest(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg), nullptr, broadcast_mgr);

    // Verify BroadcastManager is stored (access via internal member)
    EXPECT_EQ(cache->transfer_dispatcher_->multi_rank_engine_->broadcast_manager_, broadcast_mgr);
    EXPECT_EQ(cache->transfer_dispatcher_->multi_rank_engine_->broadcast_manager_->workerNum(), 2u);
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferSucceedsForAllWorkers) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, true, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_TRUE(
        cache->transfer_dispatcher_->multi_rank_engine_->execute(makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferFailsOnWorkerRpcError) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, true, grpc::Status(grpc::StatusCode::INTERNAL, "worker failed")},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(
        cache->transfer_dispatcher_->multi_rank_engine_->execute(makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferFailsOnWorkerBusinessError) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(
        cache->transfer_dispatcher_->multi_rank_engine_->execute(makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastTransferFailsOnMissingMemoryResponse) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {false, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    EXPECT_FALSE(
        cache->transfer_dispatcher_->multi_rank_engine_->execute(makeBroadcastDescriptors(), /*timeout_ms=*/500));
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastHostLoadBackCommitsDeviceSlot) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>      host_pool = makeHostPool(256, 4);
    std::shared_ptr<FullComponentGroup> group     = std::make_shared<FullComponentGroup>();
    group->component_group_id                     = 0;
    group->setHostPool(host_pool);
    const BlockIdxType device_block = prepareDeviceTarget(group, "broadcast_host_load_back_success");
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    std::vector<Component> components;
    sealBroadcastLayout(group, components);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache                 = true;
    config.enable_load_back                    = true;
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(block_tree_cache_test::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeMatchResult match = cache->match({100});
    ASSERT_NE(match.load_back_ticket, nullptr);
    ASSERT_EQ(match.load_back_ticket->items().size(), 1u);
    ASSERT_TRUE(match.load_back_ticket->bindTargetDeviceBlocks(0, {device_block}));
    const auto context = match.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success());
    cache->waitForPendingTasks();

    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);
    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    group->devicePools().front()->decRef(device_block, BlockRefType::REQUEST);
    cache->onBlocksReleased();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2D);
        EXPECT_EQ(worker_request.copy_items(0).mem_block(), host_block);
        expectSingleTaggedBlock(worker_request.copy_items(0), "tag_0", device_block);
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastHostLoadBackFailureKeepsSourceSlot) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, false, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>      host_pool = makeHostPool(256, 4);
    std::shared_ptr<FullComponentGroup> group     = std::make_shared<FullComponentGroup>();
    group->component_group_id                     = 0;
    group->setHostPool(host_pool);
    const BlockIdxType device_block = prepareDeviceTarget(group, "broadcast_host_load_back_failure");
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    std::vector<Component> components;
    sealBroadcastLayout(group, components);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache                 = true;
    config.enable_load_back                    = true;
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(block_tree_cache_test::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeMatchResult match = cache->match({100});
    ASSERT_NE(match.load_back_ticket, nullptr);
    ASSERT_EQ(match.load_back_ticket->items().size(), 1u);
    ASSERT_TRUE(match.load_back_ticket->bindTargetDeviceBlocks(0, {device_block}));
    const auto context = match.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_FALSE(context->success());
    cache->waitForPendingTasks();

    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);
    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_TRUE(slot.has_value(Tier::HOST));
    EXPECT_EQ(slot.host_block, host_block);
    EXPECT_FALSE(slot.has_value(Tier::DEVICE));
    EXPECT_EQ(host_pool->freeBlocksNum(), 3u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 1u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2D);
        EXPECT_EQ(worker_request.copy_items(0).mem_block(), host_block);
        expectSingleTaggedBlock(worker_request.copy_items(0), "tag_0", device_block);
    }
    group->devicePools().front()->decRef(device_block, BlockRefType::REQUEST);
}

TEST_F(MultiRankBlockTransferEngineTest, LoadBackCompletionStateMismatchDoesNotInstallTargetOrClearSource) {
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>      host_pool = makeHostPool(256, 4);
    std::shared_ptr<FullComponentGroup> group     = std::make_shared<FullComponentGroup>();
    group->component_group_id                     = 0;
    group->setHostPool(host_pool);
    const BlockIdxType device_block = prepareDeviceTarget(group, "load_back_completion_state_mismatch");
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    std::vector<Component> components;
    sealBroadcastLayout(group, components);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache                 = true;
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(block_tree_cache_test::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{host_block}}}, BlockRefType::REQUEST);
    ASSERT_TRUE(cache->changeLoadBackStateNolock(
        find_result.matched_node, 0, SlotTransferState::IDLE, SlotTransferState::LOAD_BACK_PENDING));
    cache->evictor_.eraseNode(find_result.matched_node, 0, Tier::HOST);
    ASSERT_TRUE(cache->changeLoadBackStateNolock(
        find_result.matched_node, 0, SlotTransferState::LOAD_BACK_PENDING, SlotTransferState::LOADING_BACK));
    find_result.matched_node->group_slots[0].transfer_state = SlotTransferState::DEMOTING;

    LoadBackTicket::PendingLoadBackItem pending_item;
    pending_item.node                 = find_result.matched_node;
    pending_item.group_id             = 0;
    pending_item.source_tier          = Tier::HOST;
    pending_item.source_blocks        = {host_block};
    pending_item.target_device_blocks = {device_block};
    LoadBackWorker::TaskPtr task = cache->load_back_worker_.createTask({pending_item}, {group});
    ASSERT_NE(task, nullptr);
    cache->runLoadBackTask(task);

    GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::DEMOTING);
    EXPECT_EQ(slot.host_block, host_block);
    EXPECT_FALSE(slot.has_value(Tier::DEVICE));
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    EXPECT_FALSE(group->devicePools()[0]->isAllocated(device_block));

    // The synthetic foreign operation is not completed by this test. Restore
    // its state so cache teardown can drain the remaining source cache hold.
    slot.transfer_state = SlotTransferState::IDLE;
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastDiskLoadBackUsesTwoTransferStages) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 4);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 4, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup>     group     = std::make_shared<FullComponentGroup>();
    group->component_group_id                         = 0;
    group->setHostPool(host_pool);
    group->setDiskPool(disk_pool);
    const BlockIdxType device_block = prepareDeviceTarget(group, "broadcast_disk_load_back");
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    std::vector<Component> components;
    sealBroadcastLayout(group, components);
    const BlockIdxType disk_block = group->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache                 = true;
    config.enable_disk_cache                   = true;
    config.enable_load_back                    = true;
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].disk_slot = disk_block;
    ASSERT_TRUE(block_tree_cache_test::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeMatchResult match = cache->match({100});
    ASSERT_NE(match.load_back_ticket, nullptr);
    ASSERT_EQ(match.load_back_ticket->items().size(), 1u);
    ASSERT_TRUE(match.load_back_ticket->bindTargetDeviceBlocks(0, {device_block}));
    const auto context = match.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success());
    cache->waitForPendingTasks();

    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);
    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::DISK));
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    group->devicePools().front()->decRef(device_block, BlockRefType::REQUEST);
    cache->onBlocksReleased();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 4u);
    BlockIdxType staging_host_block   = NULL_BLOCK_IDX;
    size_t       disk_to_host_count   = 0;
    size_t       host_to_device_count = 0;
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        const MemoryOperationRequestPB::CopyItem& request_item = worker_request.copy_items(0);
        if (worker_request.copy_direction() == MemoryOperationRequestPB::DISK2H) {
            ++disk_to_host_count;
            EXPECT_EQ(request_item.src_disk_slot(), disk_block);
            staging_host_block = request_item.mem_block();
        } else if (worker_request.copy_direction() == MemoryOperationRequestPB::H2D) {
            ++host_to_device_count;
            expectSingleTaggedBlock(request_item, "tag_0", device_block);
            if (!isNullBlockIdx(staging_host_block)) {
                EXPECT_EQ(request_item.mem_block(), staging_host_block);
            }
        }
    }
    EXPECT_EQ(disk_to_host_count, 2u);
    EXPECT_EQ(host_to_device_count, 2u);
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastEvictionSuccessCommitsPlan) {
    std::shared_ptr<MultiRankBlockTransferRpcState>    state   = std::make_shared<MultiRankBlockTransferRpcState>();
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 8);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup>     full      = std::make_shared<FullComponentGroup>();
    full->component_group_id                          = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    std::vector<Component> components;
    sealBroadcastLayout(full, components);
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_device_cache             = false;
    config.enable_memory_cache             = true;
    config.enable_disk_cache               = true;
    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(block_tree_cache_test::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    cache->waitForPendingTasks();

    BlockTreeFindResult after = cache->tree()->findNode({100});
    ASSERT_NE(after.matched_node, nullptr);
    const GroupSlot& slot = after.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_TRUE(slot.has_value(Tier::DISK));
    const BlockIdxType disk_slot = slot.disk_slot;
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2DISK);
        EXPECT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_items(0).src_mem_block(), host_block);
        EXPECT_EQ(worker_request.copy_items(0).disk_slot(), disk_slot);
        ASSERT_EQ(worker_request.copy_items(0).component_group_tags_size(), 1);
        EXPECT_EQ(worker_request.copy_items(0).component_group_tags(0), "tag_0");
    }
}

TEST_F(MultiRankBlockTransferEngineTest, BroadcastEvictionFailureRollsBackPlan) {
    const std::vector<MultiRankBlockTransferRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<MultiRankBlockTransferRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>          host_pool = makeHostPool(256, 8);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup>     full      = std::make_shared<FullComponentGroup>();
    full->component_group_id                          = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    std::vector<Component> components;
    sealBroadcastLayout(full, components);
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_device_cache             = false;
    config.enable_memory_cache             = true;
    config.enable_disk_cache               = true;
    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(block_tree_cache_test::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    cache->waitForPendingTasks();

    BlockTreeFindResult after = cache->tree()->findNode({100});
    ASSERT_NE(after.matched_node, nullptr);
    const GroupSlot& slot = after.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_TRUE(slot.has_value(Tier::HOST));
    EXPECT_FALSE(slot.has_value(Tier::DISK));
    EXPECT_EQ(slot.host_block, host_block);
    EXPECT_EQ(host_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 1u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 0u);
}

TEST_F(MultiRankBlockTransferEngineTest, BuildEvictionTransferRequestIncludesPrimaryAndCascades) {
    std::shared_ptr<HostBlockPool>          host_pool     = makeHostPool(256, 8);
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool     = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup>     primary_group = std::make_shared<FullComponentGroup>();
    primary_group->component_group_id                     = 0;
    primary_group->setHostPool(host_pool);
    primary_group->setDiskPool(disk_pool);
    std::vector<Component> components;
    sealBroadcastLayout(primary_group, components);
    std::shared_ptr<FullComponentGroup> cascade_group = std::make_shared<FullComponentGroup>();
    cascade_group->component_group_id                 = 1;
    cascade_group->setHostPool(host_pool);
    cascade_group->setDiskPool(disk_pool);
    sealBroadcastLayout(cascade_group, components);
    std::vector<ComponentGroupPtr>  groups = {primary_group, cascade_group};
    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups), std::move(components));
    ASSERT_NE(cache, nullptr);

    BlockTreeEvictor::EvictionPlan plan;
    plan.primary.component_group_id = 0;
    plan.primary.source_tier        = Tier::HOST;
    plan.primary.target_tier        = Tier::DISK;
    plan.primary.source_blocks      = {3};
    plan.primary.target_blocks      = {4};

    EvictionMove cascade;
    cascade.component_group_id = 1;
    cascade.source_tier        = Tier::HOST;
    cascade.target_tier        = Tier::DISK;
    cascade.source_blocks      = {5};
    cascade.target_blocks      = {6};
    plan.cascade_moves.push_back(cascade);

    std::vector<TransferDescriptor> descriptors;
    ASSERT_TRUE(cache->buildEvictionTransferBatch(plan, descriptors));
    MemoryOperationRequestPB request;
    for (const TransferDescriptor& descriptor : descriptors) {
        ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(descriptor, cache->componentGroups(), request));
    }
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(request.copy_items(0).src_mem_block(), 3);
    EXPECT_EQ(request.copy_items(0).disk_slot(), 4);
    ASSERT_EQ(request.copy_items(0).component_group_tags_size(), 1);
    EXPECT_EQ(request.copy_items(0).component_group_tags(0), "tag_0");
    EXPECT_EQ(request.copy_items(1).src_mem_block(), 5);
    EXPECT_EQ(request.copy_items(1).disk_slot(), 6);
    ASSERT_EQ(request.copy_items(1).component_group_tags_size(), 1);
    EXPECT_EQ(request.copy_items(1).component_group_tags(0), "tag_1");
}

}  // namespace
}  // namespace rtp_llm
