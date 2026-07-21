#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <mutex>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CopyEngineTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

struct BlockTreeBroadcastRpcState {
    std::mutex                            mutex;
    std::vector<MemoryOperationRequestPB> requests;
};

struct BlockTreeBroadcastRpcConfig {
    bool                                        has_mem_response;
    bool                                        mem_response_success;
    grpc::Status                                rpc_status;
    std::shared_ptr<BlockTreeBroadcastRpcState> state{nullptr};
};

class BlockTreeBroadcastRpcService final: public RpcService::Service {
public:
    explicit BlockTreeBroadcastRpcService(const BlockTreeBroadcastRpcConfig& config): config_(config) {}

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
    BlockTreeBroadcastRpcConfig config_;
};

class BlockTreeBroadcastRpcServer {
public:
    explicit BlockTreeBroadcastRpcServer(std::unique_ptr<BlockTreeBroadcastRpcService> service):
        service_(std::move(service)) {}

    ~BlockTreeBroadcastRpcServer() {
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
    std::unique_ptr<BlockTreeBroadcastRpcService> service_;
    std::unique_ptr<grpc::Server>                 server_;
    int                                           listen_port_{0};
};

class MetadataDeviceBlockPool: public DeviceBlockPool {
public:
    explicit MetadataDeviceBlockPool(const std::shared_ptr<const DeviceBlockPoolConfig>& config):
        DeviceBlockPool(config) {}

    void initMetadata() {
        markInitialized();
    }
};

static std::shared_ptr<BroadcastManager>
makeBroadcastManager(const std::vector<BlockTreeBroadcastRpcConfig>&            configs,
                     std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>>& servers) {
    std::vector<std::string> worker_addrs;
    worker_addrs.reserve(configs.size());
    servers.reserve(configs.size());
    for (const BlockTreeBroadcastRpcConfig& config : configs) {
        std::unique_ptr<BlockTreeBroadcastRpcService> service = std::make_unique<BlockTreeBroadcastRpcService>(config);
        std::unique_ptr<BlockTreeBroadcastRpcServer>  server =
            std::make_unique<BlockTreeBroadcastRpcServer>(std::move(service));
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
    std::vector<ComponentGroupPtr> groups    = {full};
    return makeBlockTreeCacheForTest(std::move(tree),
                                     std::move(groups),
                                     std::vector<Component>{},
                                     BlockTreeCacheConfig{},
                                     /*storage_backend=*/nullptr,
                                     broadcast_manager);
}

static BlockIdxType prepareDeviceTarget(const std::shared_ptr<FullComponentGroup>& group,
                                        const std::string&                         pool_name) {
    MemoryLayoutConfig memory_layout;
    memory_layout.layer_num                = 1;
    memory_layout.block_num                = 9;
    memory_layout.kv_block_pool_size_bytes = 9 * 256;

    std::shared_ptr<DeviceBlockPoolConfig> device_config = std::make_shared<DeviceBlockPoolConfig>();
    device_config->pool_type                             = BlockPoolType::DEVICE;
    device_config->pool_name                             = pool_name;
    device_config->physical_block_count                  = 9;
    device_config->memory_layouts                        = {memory_layout};
    std::shared_ptr<MetadataDeviceBlockPool> device_pool = std::make_shared<MetadataDeviceBlockPool>(device_config);
    device_pool->initMetadata();
    const BlockIdxType device_block = poolMalloc(*device_pool);
    if (isNullBlockIdx(device_block)) {
        return NULL_BLOCK_IDX;
    }
    device_pool->incRef(device_block, BlockRefType::BLOCK_CACHE);
    group->setDevicePools({device_pool});
    return device_block;
}

static void sealBroadcastLayout(const std::shared_ptr<FullComponentGroup>& group,
                                std::vector<Component>&                    components,
                                size_t                                     payload_bytes = 256) {
    if (group->devicePoolCount() == 0) {
        group->setDevicePools({DeviceBlockPoolPtr{}});
    }
    const int component_index = static_cast<int>(components.size());
    components.push_back(copy_engine_test::makeSchemaComponent(
        component_index, group->component_group_id, "broadcast_kv", {payload_bytes}));
    RTP_LLM_CHECK(group->finalizeLayout({component_index}, components));
}

static MemoryOperationRequestPB makeBroadcastRequest() {
    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::H2D);
    MemoryOperationRequestPB::CopyItem* item = request.add_copy_items();
    RTP_LLM_CHECK(item != nullptr);
    item->set_mem_block(1);
    item->add_gpu_blocks(1);
    return request;
}

class BlockTreeCacheBroadcastTest: public ::testing::Test {
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

TEST_F(BlockTreeCacheBroadcastTest, BroadcastManagerStoredCorrectly) {
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
    EXPECT_EQ(cache->broadcast_manager_, broadcast_mgr);
    EXPECT_EQ(cache->broadcast_manager_->workerNum(), 2u);
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastTransferSucceedsForAllWorkers) {
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, true, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    const MemoryOperationRequestPB request = makeBroadcastRequest();
    EXPECT_TRUE(cache->broadcastTransfer(request, /*timeout_ms=*/500));
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastTransferFailsOnWorkerRpcError) {
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, true, grpc::Status(grpc::StatusCode::INTERNAL, "worker failed")},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    const MemoryOperationRequestPB request = makeBroadcastRequest();
    EXPECT_FALSE(cache->broadcastTransfer(request, /*timeout_ms=*/500));
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastTransferFailsOnWorkerBusinessError) {
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    const MemoryOperationRequestPB request = makeBroadcastRequest();
    EXPECT_FALSE(cache->broadcastTransfer(request, /*timeout_ms=*/500));
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastTransferFailsOnMissingMemoryResponse) {
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {false, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeBroadcastCache(broadcast_manager);

    const MemoryOperationRequestPB request = makeBroadcastRequest();
    EXPECT_FALSE(cache->broadcastTransfer(request, /*timeout_ms=*/500));
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastHostLoadBackCommitsDeviceSlot) {
    std::shared_ptr<BlockTreeBroadcastRpcState>    state   = std::make_shared<BlockTreeBroadcastRpcState>();
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
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
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{host_block}}}, BlockRefType::REQUEST);
    ASSERT_TRUE(cache->evictor_.beginLoadBack(find_result.matched_node, 0, Tier::HOST));
    BlockTreeCache::LoadBackItem item{find_result.matched_node, 0, Tier::HOST, {host_block}, {device_block}};
    cache->performLoadBack({item}, /*ctx=*/nullptr);

    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    std::lock_guard<std::mutex> lock(state->mutex);
    ASSERT_EQ(state->requests.size(), 2u);
    for (const MemoryOperationRequestPB& worker_request : state->requests) {
        ASSERT_EQ(worker_request.copy_items_size(), 1);
        EXPECT_EQ(worker_request.copy_direction(), MemoryOperationRequestPB::H2D);
        EXPECT_EQ(worker_request.copy_items(0).mem_block(), host_block);
        ASSERT_EQ(worker_request.copy_items(0).gpu_blocks_size(), 1);
        EXPECT_EQ(worker_request.copy_items(0).gpu_blocks(0), device_block);
    }
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastHostLoadBackFailureKeepsSourceSlot) {
    std::shared_ptr<BlockTreeBroadcastRpcState>    state   = std::make_shared<BlockTreeBroadcastRpcState>();
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, false, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
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
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{host_block}}}, BlockRefType::REQUEST);
    ASSERT_TRUE(cache->evictor_.beginLoadBack(find_result.matched_node, 0, Tier::HOST));
    BlockTreeCache::LoadBackItem item{find_result.matched_node, 0, Tier::HOST, {host_block}, {device_block}};
    cache->performLoadBack({item}, /*ctx=*/nullptr);

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
        EXPECT_EQ(worker_request.copy_items(0).gpu_blocks(0), device_block);
    }
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastDiskLoadBackUsesTwoTransferStages) {
    std::shared_ptr<BlockTreeBroadcastRpcState>    state   = std::make_shared<BlockTreeBroadcastRpcState>();
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>      host_pool = makeHostPool(256, 4);
    std::shared_ptr<DiskBlockPool>      disk_pool = makeDiskPool(256, 4, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> group     = std::make_shared<FullComponentGroup>();
    group->component_group_id                     = 0;
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
    std::vector<ComponentGroupPtr>      groups = {group};
    std::unique_ptr<BlockTreeCache>     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1),
                                                                      std::move(groups),
                                                                      std::move(components),
                                                                      std::move(config),
                                                                      /*storage_backend=*/nullptr,
                                                                      broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].disk_slot = disk_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::DISK, {{disk_block}}}, BlockRefType::REQUEST);
    ASSERT_TRUE(cache->evictor_.beginLoadBack(find_result.matched_node, 0, Tier::DISK));
    BlockTreeCache::LoadBackItem item{find_result.matched_node, 0, Tier::DISK, {disk_block}, {device_block}};
    cache->performLoadBack({item}, /*ctx=*/nullptr);

    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::DISK));
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 0u);
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
            EXPECT_EQ(request_item.gpu_blocks(0), device_block);
            if (!isNullBlockIdx(staging_host_block)) {
                EXPECT_EQ(request_item.mem_block(), staging_host_block);
            }
        }
    }
    EXPECT_EQ(disk_to_host_count, 2u);
    EXPECT_EQ(host_to_device_count, 2u);
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastEvictionSuccessCommitsPlan) {
    std::shared_ptr<BlockTreeBroadcastRpcState>    state   = std::make_shared<BlockTreeBroadcastRpcState>();
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>      host_pool = makeHostPool(256, 8);
    std::shared_ptr<DiskBlockPool>      disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> full      = std::make_shared<FullComponentGroup>();
    full->component_group_id                      = 0;
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
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    cache->insert(nullptr, {200}, std::vector<std::vector<GroupSlot>>(1, std::vector<GroupSlot>(1)));
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
    }
}

TEST_F(BlockTreeCacheBroadcastTest, BroadcastEvictionFailureRollsBackPlan) {
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool>      host_pool = makeHostPool(256, 8);
    std::shared_ptr<DiskBlockPool>      disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> full      = std::make_shared<FullComponentGroup>();
    full->component_group_id                      = 0;
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
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    cache->insert(nullptr, {200}, std::vector<std::vector<GroupSlot>>(1, std::vector<GroupSlot>(1)));
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

TEST_F(BlockTreeCacheBroadcastTest, BuildEvictionTransferRequestIncludesPrimaryAndCascades) {
    std::shared_ptr<HostBlockPool>      host_pool     = makeHostPool(256, 8);
    std::shared_ptr<DiskBlockPool>      disk_pool     = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> primary_group = std::make_shared<FullComponentGroup>();
    primary_group->component_group_id                 = 0;
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

    MemoryOperationRequestPB request;
    ASSERT_TRUE(cache->buildEvictionTransferRequest(plan, request));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(request.copy_items(0).src_mem_block(), 3);
    EXPECT_EQ(request.copy_items(0).disk_slot(), 4);
    EXPECT_EQ(request.copy_items(1).src_mem_block(), 5);
    EXPECT_EQ(request.copy_items(1).disk_slot(), 6);
}

}  // namespace
}  // namespace rtp_llm
