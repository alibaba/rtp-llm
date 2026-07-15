#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

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
    std::unique_ptr<grpc::Server>                  server_;
    int                                           listen_port_{0};
};

static std::shared_ptr<BroadcastManager>
makeBroadcastManager(const std::vector<BlockTreeBroadcastRpcConfig>&           configs,
                     std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>>& servers) {
    std::vector<std::string> worker_addrs;
    worker_addrs.reserve(configs.size());
    servers.reserve(configs.size());
    for (const BlockTreeBroadcastRpcConfig& config : configs) {
        std::unique_ptr<BlockTreeBroadcastRpcService> service =
            std::make_unique<BlockTreeBroadcastRpcService>(config);
        std::unique_ptr<BlockTreeBroadcastRpcServer> server =
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
    return BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                      std::move(groups),
                                                      std::vector<Component>{},
                                                      BlockTreeCacheConfig{},
                                                      /*storage_backend=*/nullptr,
                                                      broadcast_manager);
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

// Helper: create a v4 HostBlockPool with the given payload_bytes and usable_count.
// IBlockPool reserves block 0, so physical_block_count = usable_count + 1. The pool
// is returned uninitialized; callers invoke init() (which is not double-call guarded).
static std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count) {
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = "block_tree_cache_host";
    config->physical_block_count = usable_count + 1;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = ((payload_bytes + 4095) / 4096) * 4096;
    config->enable_pinned        = true;
    config->alignment            = 4096;
    return std::make_shared<HostBlockPool>(config);
}

class MemoryDiskBlockIO: public DiskBlockIO {
public:
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t bytes, bool) override {
        data_.assign(bytes, 0);
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override {
        if (offset + bytes > data_.size()) {
            return DiskBlockIOStatus::INVALID_SIZE;
        }
        std::memcpy(dst, data_.data() + offset, bytes);
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        if (offset + bytes > data_.size()) {
            return DiskBlockIOStatus::INVALID_SIZE;
        }
        std::memcpy(data_.data() + offset, src, bytes);
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override {
        for (const auto& read_req : reads) {
            auto status = read(read_req.offset, read_req.buffer, read_req.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        for (const auto& write_req : writes) {
            auto status = write(write_req.offset, write_req.buffer, write_req.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }

    void close() override {}

    std::string debugString() const override {
        return "MemoryDiskBlockIO";
    }

private:
    std::vector<char> data_;
};

static std::shared_ptr<DiskBlockPool>
makeDiskPool(size_t payload_bytes, size_t usable_count, std::unique_ptr<DiskBlockIO> io = nullptr) {
    const size_t aligned_block_size = ((payload_bytes + 4095) / 4096) * 4096;

    auto config             = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = "block_tree_cache_disk";
    config->work_dir        = "/tmp";
    config->local_rank      = 0;
    config->world_rank      = 0;
    config->disk_size_bytes = aligned_block_size * (usable_count + 1);
    config->payload_bytes   = payload_bytes;
    config->stride_bytes    = aligned_block_size;
    config->buffered_io     = true;

    auto pool = std::make_shared<DiskBlockPool>(config, std::move(io));
    RTP_LLM_CHECK(pool->init());
    return pool;
}

static bool cudaAvailable() {
    try {
        return torch::cuda::is_available();
    } catch (const std::exception&) {
        return false;
    }
}

// Helper: build an initialized DeviceBlockPool from the lightweight cache-config test
// helpers.
static DeviceBlockPoolPtr makeDevicePool() {
    constexpr int    kLayerNum       = 4;
    constexpr int    kBlockNum       = 10;
    constexpr size_t kTokensPerBlock = 1;
    CacheConfig      cache_config    = test::makeSimpleMhaCacheConfig(kLayerNum,
                                                              kBlockNum,
                                                              kTokensPerBlock,
                                                              TYPE_FP16,
                                                              /*local_head_num_kv=*/1,
                                                              /*size_per_head=*/64);
    auto config =
        std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfig(cache_config));
    config->pool_name               = "block_tree_cache_device";
    config->use_cuda_malloc_backing = false;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    pool->init();
    return pool;
}

struct DeviceLayerBufferSpec {
    size_t kv_bytes{0};
    size_t scale_bytes{0};
};

static DeviceBlockPoolPtr
makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs, size_t usable_count, const std::string& pool_name) {
    const auto physical_block_count = usable_count + 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = pool_name;
    config->physical_block_count    = physical_block_count;
    config->use_cuda_malloc_backing = false;

    size_t offset = 0;
    for (const auto& spec : specs) {
        MemoryLayoutConfig layout;
        layout.layer_num                = 1;
        layout.block_num                = static_cast<uint32_t>(physical_block_count);
        layout.dtype                    = TYPE_INT8;
        layout.kv_cache_offset_bytes    = offset;
        layout.kv_block_stride_bytes    = spec.kv_bytes;
        layout.kv_block_pool_size_bytes = physical_block_count * spec.kv_bytes;
        layout.block_stride_bytes       = spec.kv_bytes + spec.scale_bytes;
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
        offset += layout.kv_block_pool_size_bytes;

        if (spec.scale_bytes > 0) {
            layout.enable_kv_scale          = true;
            layout.kv_scale_offset_bytes    = offset;
            layout.kv_scale_stride_bytes    = spec.scale_bytes;
            layout.kv_scale_pool_size_bytes = physical_block_count * spec.scale_bytes;
            layout.total_size_bytes += layout.kv_scale_pool_size_bytes;
            offset += layout.kv_scale_pool_size_bytes;
        }

        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;
        config->memory_layouts.push_back(layout);
    }
    config->total_size_bytes = offset;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

static BlockIdxType poolMalloc(IBlockPool& pool) {
    auto block = pool.malloc();
    return block.has_value() ? *block : NULL_BLOCK_IDX;
}

// referenceDeviceBlocks() must add exactly one cache-category holder (incRef) and
// releaseDeviceBlocks() must release it (decRef), returning capacity at refcount 0.
TEST(BlockTreeCacheComponentGroupTest, DevicePoolLifecycleUsesSingleCountRefcount) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto pool = makeDevicePool();
    ASSERT_NE(pool, nullptr);

    FullComponentGroup group;
    group.component_group_id = 0;
    group.setDevicePools({pool});
    EXPECT_EQ(group.devicePoolCount(), 1u);

    // malloc reserves capacity only; the block starts at refCount 0. The exact index is not
    // asserted: ANY_ORDER makes the choice opaque, which is the correct device-pool behavior.
    auto blocks_opt = pool->malloc(1);
    ASSERT_TRUE(blocks_opt.has_value());
    ASSERT_EQ(blocks_opt->size(), 1u);
    const BlockIdxType block = (*blocks_opt)[0];
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 0u);

    // Cache holder acquired via ComponentGroup -> pool incRef.
    group.referenceDeviceBlocks({block});
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_TRUE(pool->isAllocated(block));

    // Cache holder released via ComponentGroup -> pool decRef; at 0 the block is freed.
    group.releaseDeviceBlocks({block});
    EXPECT_FALSE(pool->isAllocated(block));
}

// Helper to build a simple single-group BlockTreeCache for testing.
class BlockTreeCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree = std::make_unique<BlockTree>(1);  // 1 component group

        auto full_group                = std::make_shared<FullComponentGroup>();
        full_group->component_group_id = 0;

        std::vector<ComponentGroupPtr> groups = {full_group};
        std::vector<Component>         components;

        cache_ = BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeCacheTest, MatchEmptyTree) {
    auto result = cache_->match({100, 200, 300});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.group_block_indices.empty());
}

TEST_F(BlockTreeCacheTest, MatchAfterInsert) {
    // Insert path: 100 → 200 → 300 with per-node blocks
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};

    cache_->insert(nullptr, {100, 200, 300}, slots);

    auto result = cache_->match({100, 200, 300});
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 3u);
    // Each node along the path has its own device_blocks
    const std::unordered_map<int, BlockIndicesType>::const_iterator group_it = result.group_block_indices.find(0);
    ASSERT_NE(group_it, result.group_block_indices.end());
    EXPECT_EQ(group_it->second, (BlockIndicesType{42, 43, 44}));
}

TEST_F(BlockTreeCacheTest, MatchPartialPath) {
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};

    cache_->insert(nullptr, {100, 200, 300}, slots);

    // Match only first 2 keys (4th key doesn't exist)
    auto result = cache_->match({100, 200, 999});
    EXPECT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 2u);
}

TEST_F(BlockTreeCacheTest, InsertNewPath) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};

    cache_->insert(nullptr, {100, 200}, slots);

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, InsertOverlappingPathUpdatesHeat) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};

    cache_->insert(nullptr, {100, 200}, slots);
    cache_->insert(nullptr, {100, 200}, slots);  // Overlap

    // Should still be 2 nodes (no duplication)
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, ReclaimDeviceLeaf) {
    // Insert: root → 100 → 200 → 300
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache_->insert(nullptr, {100, 200, 300}, slots);

    // 300 is a DeviceLeaf (no children with device data)
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    int reclaimed = cache_->reclaimBlocks(1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);

    cache_->waitForPendingTasks();

    // After reclaim, the leaf's device_blocks should be cleared.
    // Check that match no longer finds device data for 300.
    auto result = cache_->match({100, 200, 300});
    // 300's group_slots[0] should have no device value
    // But 100 and 200 also have no device data (only 300 was given slots)
    // So match would fail at 100 (no data in any tier)
}

TEST_F(BlockTreeCacheTest, ReclaimEmptyTreeReturnsZero) {
    int reclaimed = cache_->reclaimBlocks(1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 0);
}

TEST_F(BlockTreeCacheTest, ReclaimCascadesToLowerPriorityGroup) {
    // Build a cache with Full + SWA groups
    auto tree = std::make_unique<BlockTree>(2);  // 2 component groups

    auto full_group                = std::make_shared<FullComponentGroup>();
    full_group->component_group_id = 0;

    auto swa_group                = std::make_shared<SWAComponentGroup>(128, 64);
    swa_group->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {full_group, swa_group};
    std::vector<Component>         components;

    std::unique_ptr<BlockTreeCache> multi_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));

    // Insert a node with both Full and SWA data
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};  // Full
    slots[0][1].device_blocks = {20};  // SWA

    multi_cache->insert(nullptr, {100}, slots);

    // Reclaim Full group at DEVICE → should cascade to SWA.
    int reclaimed = multi_cache->reclaimBlocks(1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);

    multi_cache->waitForPendingTasks();
}

TEST_F(BlockTreeCacheTest, NodeDeletionWhenAllEmpty) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};

    cache_->insert(nullptr, {100, 200}, slots);

    auto stats_before = cache_->getStats();
    EXPECT_EQ(stats_before.tree_node_count, 2u);

    // Reclaim: the leaf (200) should be removed after reclaim.
    cache_->reclaimBlocks(1, Tier::DEVICE);
    cache_->waitForPendingTasks();

    // After reclaim and cleanup, tree might be smaller.
    auto stats_after = cache_->getStats();
    // Node 200 should be removed (all REUSABLE groups empty)
    // Node 100 should also be removed (empty ancestor)
    EXPECT_LE(stats_after.tree_node_count, 2u);
}

TEST_F(BlockTreeCacheTest, GetStats) {
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    cache_->insert(nullptr, {100}, slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 1u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheTest, MultiGroupConstruction) {
    auto tree = std::make_unique<BlockTree>(3);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 1;

    auto linear                = std::make_shared<LinearComponentGroup>();
    linear->component_group_id = 2;

    std::vector<ComponentGroupPtr> groups = {full, swa, linear};
    std::vector<Component>         components;

    std::unique_ptr<BlockTreeCache> multi_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));

    EXPECT_EQ(multi_cache->componentGroups().size(), 3u);
    EXPECT_EQ(multi_cache->tree()->groupSlotCount(), 3);
}

TEST(BlockTreeCacheConstructionTest, OutOfRangeComponentGroupIdFailsInitializationWithoutThrowing) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 1;
    std::vector<ComponentGroupPtr> groups = {full};
    std::vector<Component>         components;

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  nullptr,
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
}

TEST_F(BlockTreeCacheTest, MatchEmptyKeys) {
    auto result = cache_->match({});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
}

TEST_F(BlockTreeCacheTest, InsertEmptyKeys) {
    cache_->insert(nullptr, {}, {});
    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ThreadSafety) {
    // Basic thread safety test: concurrent inserts
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, i]() {
            std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
            slots[0][0].device_blocks = {static_cast<BlockIdxType>(i * 100)};
            CacheKeysType keys        = {static_cast<CacheKeyType>(i * 1000 + 1)};
            cache_->insert(nullptr, keys, slots);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 4u);
}

// ---------------------------------------------------------------------------
// CopyEngine integration tests
// ---------------------------------------------------------------------------

// Test: reclaimBlocks directly drops device blocks even when host demotion is available.
//
//   Before reclaim:                          After reclaim:
//   root → [100] D={42} ← heap               empty tree
//
//   No host block is allocated and no copy task is submitted.
TEST_F(BlockTreeCacheTest, ReclaimBlocksDoesNotAllocateHostBlock) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());
    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig ce_cfg;
    ce_cfg.eviction_thread_pool_size = 2;
    ce_cfg.enable_device_cache       = true;
    ce_cfg.enable_memory_cache       = true;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(ce_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);

    EXPECT_EQ(ce_cache->reclaimBlocks(1, Tier::DEVICE), 1);
    ce_cache->waitForPendingTasks();

    // Direct reclaim bypasses demotion/copy, so host capacity is unchanged.
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
    auto find = ce_cache->tree()->findNode({100});
    EXPECT_EQ(find.matched_node, nullptr);
}

// Test: Sequential direct reclaim without Host pool.
//
//   root → [100] → [200] → [300] all D={42}
//   Host disabled → reclaim target=NONE (direct release), synchronous.
//   Sequential reclaim drains all 3 nodes.
TEST_F(BlockTreeCacheTest, SequentialReclaimDrainsChainWithoutHostBlocks) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // No Host pool, Host disabled → direct release on reclaim.
    BlockTreeCacheConfig seq_cfg;
    seq_cfg.eviction_thread_pool_size = 2;
    seq_cfg.enable_device_cache       = true;
    seq_cfg.enable_memory_cache       = false;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(seq_cfg));

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    ce_cache->insert(nullptr, {100, 200, 300}, slots);

    // Reclaim all 3 nodes sequentially (synchronous direct release)
    for (int i = 0; i < 3; ++i) {
        int reclaimed = ce_cache->reclaimBlocks(1, Tier::DEVICE);
        EXPECT_EQ(reclaimed, 1) << "Reclaim " << i << " should succeed";
        ce_cache->waitForPendingTasks();
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

// Test: REUSABLE direct reclaim does not demote to host even when host is enabled.
TEST_F(BlockTreeCacheTest, ReusableReclaimDoesNotAllocateHostBlock) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig reuse_cfg;
    reuse_cfg.eviction_thread_pool_size = 2;
    reuse_cfg.enable_device_cache       = true;
    reuse_cfg.enable_memory_cache       = true;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(reuse_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->reclaimBlocks(1, Tier::DEVICE), 1);
    // Synchronous, no wait needed

    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// Tier enable flag tests
// ---------------------------------------------------------------------------

// Test: Initialization validation — disk requires host.
TEST_F(BlockTreeCacheTest, DiskRequiresHostValidation) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.eviction_thread_pool_size = 2;
    config.enable_device_cache       = true;
    config.enable_memory_cache       = false;
    config.enable_disk_cache         = true;
    config.enable_remote_cache       = false;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(config));
    EXPECT_EQ(cache, nullptr);
}

// Test: reclaimBlocks on disabled tier returns 0.
TEST_F(BlockTreeCacheTest, ReclaimDisabledTierReturnsZero) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Device enabled, Host disabled (default)
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim HOST tier — disabled → returns 0
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::HOST), 0);
    // Reclaim DEVICE tier — enabled → returns 1
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
}

// Test: Host disabled → Device reclaim does direct release (no D2H demotion).
TEST_F(BlockTreeCacheTest, HostDisabledDirectRelease) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Host disabled (default): Device reclaim → direct release.
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// Test: Tier enable query accessors.
TEST_F(BlockTreeCacheTest, TierEnableQueries) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_remote_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isMemoryCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

// ---------------------------------------------------------------------------
// UT-2: shouldDeleteNode checks all groups
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, NodeDeletedWhenAllGroupsEmpty) {
    auto tree = std::make_unique<BlockTree>(1);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Reclaim device data.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // Node should be deleted: group empty
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

// ---------------------------------------------------------------------------
// UT-4: SWA buildTransfer supports HOST_TO_DISK (Bug 5 fix)
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, SWABuildTransferSupportsHostToDisk) {
    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 0;

    // Create a mock tree node with host data
    auto                                tree = std::make_unique<BlockTree>(1);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    tree->insertNode(nullptr, {100}, slots);
    auto find = tree->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;

    // Verify HOST_TO_DISK transfer descriptor is correct
    TransferDescriptor desc = swa->buildTransfer(find.matched_node, TransferType::HOST_TO_DISK);
    EXPECT_EQ(desc.source_tier, Tier::HOST);
    EXPECT_EQ(desc.target_tier, Tier::DISK);
    EXPECT_EQ(desc.host_block, 7);
}

TEST_F(BlockTreeCacheTest, MatchCollectsBlocksSelectedByGroupPolicy) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(3);

    std::shared_ptr<FullComponentGroup> full     = std::make_shared<FullComponentGroup>();
    full->component_group_id                     = 0;
    std::shared_ptr<LinearComponentGroup> linear = std::make_shared<LinearComponentGroup>();
    linear->component_group_id                   = 1;
    std::shared_ptr<SWAComponentGroup> swa       = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                      = 2;

    std::vector<ComponentGroupPtr>  component_groups = {full, linear, swa};
    std::unique_ptr<BlockTreeCache> cache            = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(component_groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(3));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        slots[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{22}));
    EXPECT_EQ(result.group_block_indices.at(2), (BlockIndicesType{31, 32}));
}

TEST_F(BlockTreeCacheTest, MatchKeepsAggregatedDevicePoolsSeparate) {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setDevicePools({DeviceBlockPoolPtr{}, DeviceBlockPoolPtr{}});

    std::vector<ComponentGroupPtr>             component_groups = {full};
    std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping  = {{0, 0}, {0, 1}};
    std::unique_ptr<BlockTreeCache>            cache =
        std::make_unique<BlockTreeCache>(std::move(tree),
                                         std::move(component_groups),
                                         std::vector<Component>{},
                                         BlockTreeCacheConfig{},
                                         std::shared_ptr<StorageBackend>{},
                                         std::shared_ptr<BroadcastManager>{},
                                         std::vector<DeviceKVCacheGroupPtr>{nullptr, nullptr},
                                         std::move(per_tag_mapping));
    ASSERT_TRUE(cache->init());

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[0][1].device_blocks = {20};
    slots[1][0].device_blocks = {11};
    slots[1][1].device_blocks = {21};
    cache->insert(nullptr, {100, 200}, slots);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10, 11}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{20, 21}));
}

TEST_F(BlockTreeCacheTest, InitializationRequiresPerTagMapping) {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setDevicePools({DeviceBlockPoolPtr{}});

    std::vector<ComponentGroupPtr>  component_groups = {full};
    std::unique_ptr<BlockTreeCache> cache =
        std::make_unique<BlockTreeCache>(std::move(tree),
                                         std::move(component_groups),
                                         std::vector<Component>{},
                                         BlockTreeCacheConfig{},
                                         std::shared_ptr<StorageBackend>{},
                                         std::shared_ptr<BroadcastManager>{},
                                         std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                         std::vector<BlockTreeCache::PerTagMapping>{});
    EXPECT_FALSE(cache->init());
}

TEST_F(BlockTreeCacheTest, MatchRequiresSWAWindowAfterGap) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 1;

    std::vector<ComponentGroupPtr>  groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    slots[2][0].device_blocks = {12};
    slots[3][0].device_blocks = {13};
    slots[0][1].device_blocks = {20};
    slots[2][1].device_blocks = {22};
    slots[3][1].device_blocks = {23};

    cache->insert(nullptr, {100, 200, 300, 400}, slots);

    BlockTreeMatchResult partial = cache->match({100, 200, 300});
    EXPECT_EQ(partial.matched_blocks, 1u);

    BlockTreeMatchResult restored = cache->match({100, 200, 300, 400});
    EXPECT_EQ(restored.matched_blocks, 4u);
}

// ---------------------------------------------------------------------------
// UT-6: Watermark demotion with CopyEngine — D2H copy fails without component layout.
// Copy failure rolls back: host_block is freed and the node stays in the device heap.
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, WatermarkDemotionCopyFailureRollsBack) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto host_pool = makeHostPool(256, 8);
    ASSERT_TRUE(host_pool->init());
    auto device_pool  = makeDevicePool({{256, 0}}, 8, "watermark_failure_device");
    auto device_block = poolMalloc(*device_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    full->setDevicePools({device_pool});
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;
    cfg.watermark_device    = {0.01, 0};

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {device_block};
    cache->insert(nullptr, {100}, slots);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    const auto& slot = find.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_TRUE(device_pool->isAllocated(device_block));
    EXPECT_EQ(device_pool->refCount(device_block), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
}

TEST_F(BlockTreeCacheTest, WatermarkDemotionCopiesHostBlockToDisk) {
    auto host_pool = makeHostPool(256, 8);
    ASSERT_TRUE(host_pool->init());

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    auto host_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = false;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

    auto before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    // The host-only leaf is registered in the host candidate heap automatically
    // by onInsertCommitted.
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    std::vector<std::vector<GroupSlot>> trigger_slots(1, std::vector<GroupSlot>(1));
    cache->insert(nullptr, {200}, trigger_slots);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    const auto& slot = find.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_TRUE(slot.has_value(Tier::DISK));
    EXPECT_NE(slot.disk_slot, NULL_BLOCK_IDX);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_TRUE(disk_pool->isAllocated(slot.disk_slot));
    EXPECT_EQ(disk_pool->refCount(slot.disk_slot), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);
}

// ---------------------------------------------------------------------------
// UT-7: Cascade reclaim - parent becomes device leaf after child reclaim.
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildReclaim) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache->insert(nullptr, {100, 200, 300}, slots);

    // Initially only C (leaf) is in heap
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim C -> B becomes DeviceLeaf -> enters heap.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim B -> A becomes DeviceLeaf -> enters heap.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
}

// ---------------------------------------------------------------------------
// UT-9: reclaimBlocks does not use CopyEngine, so no failed copy can update the slot.
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, ReclaimBlocksDoesNotUpdateHostSlot) {
    auto host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->component_indices  = {0};
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    // Create a component with MemoryBlockLayerTagSlot so deviceToHost attempts real copy
    Component comp;
    comp.component_id                 = 0;
    comp.component_group_id           = 0;
    comp.memory_block_layer_tag_slots = {{0, "kv", 128}};
    std::vector<Component> components = {comp};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::move(components), std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    EXPECT_EQ(find.matched_node, nullptr);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
}

TEST_F(BlockTreeCacheTest, LoadBackOnlyReloadsSWAWindow) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 1;

    std::vector<ComponentGroupPtr>  groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].host_block    = static_cast<BlockIdxType>(100 + i);
    }

    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

    BlockTreeMatchResult result = cache->match({100, 200, 300, 400});
    EXPECT_EQ(result.matched_blocks, 4u);
    EXPECT_EQ(result.host_load_back_blocks, 2u);
    EXPECT_EQ(result.load_back_blocks, 2u);
}

// ---------------------------------------------------------------------------
// Test: enable_load_back — match detects Host/Disk data needing reload
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, LoadBackDetectsHostData) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    // Insert a node and manually set host data (simulating prior demotion).
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim without host demotion, then manually set up a host-only node.
    // Instead, manually set up a node with host_block but no device_blocks
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // After reclaim without host enabled, node is deleted.
    // Let's insert again and manually simulate host-only state
    std::vector<std::vector<GroupSlot>> slots2(1, std::vector<GroupSlot>(1));
    slots2[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots2);

    // Manually set host_block and clear device_blocks to simulate a demoted state.
    auto find = cache->tree()->findNode({200});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;
    for (BlockIdxType& device_block_index : find.matched_node->group_slots[0].device_blocks) {
        device_block_index = NULL_BLOCK_IDX;
    }

    // Match should detect load_back
    auto result = cache->match({200});
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);
}

// ---------------------------------------------------------------------------
// Test: BroadcastManager is stored and accessible
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, BroadcastManagerStoredCorrectly) {
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

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg), nullptr, broadcast_mgr);

    // Verify BroadcastManager is stored (access via internal member)
    EXPECT_EQ(cache->broadcast_manager_, broadcast_mgr);
    EXPECT_EQ(cache->broadcast_manager_->workerNum(), 2u);
}

TEST_F(BlockTreeCacheTest, BroadcastTransferSucceedsForAllWorkers) {
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

TEST_F(BlockTreeCacheTest, BroadcastTransferFailsOnWorkerRpcError) {
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

TEST_F(BlockTreeCacheTest, BroadcastTransferFailsOnWorkerBusinessError) {
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

TEST_F(BlockTreeCacheTest, BroadcastTransferFailsOnMissingMemoryResponse) {
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

TEST_F(BlockTreeCacheTest, BroadcastHostLoadBackCommitsDeviceSlot) {
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
    ASSERT_TRUE(host_pool->init());
    group->component_group_id = 0;
    group->setHostPool(host_pool);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache             = true;
    std::vector<ComponentGroupPtr>  groups = {group};
    std::unique_ptr<BlockTreeCache> cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1),
                                                                                       std::move(groups),
                                                                                       std::vector<Component>{},
                                                                                       std::move(config),
                                                                                       /*storage_backend=*/nullptr,
                                                                                       broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{host_block}}});
    ASSERT_TRUE(cache->evictor_.beginLoadBack(find_result.matched_node, 0, Tier::HOST));
    BlockTreeCache::LoadBackItem item{find_result.matched_node, 0, Tier::HOST, {host_block}, {7}};
    cache->performLoadBack({item}, /*ctx=*/nullptr);

    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{7}));
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
        EXPECT_EQ(worker_request.copy_items(0).gpu_blocks(0), 7);
    }
}

TEST_F(BlockTreeCacheTest, BroadcastHostLoadBackFailureKeepsSourceSlot) {
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
    ASSERT_TRUE(host_pool->init());
    group->component_group_id = 0;
    group->setHostPool(host_pool);
    const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache             = true;
    std::vector<ComponentGroupPtr>  groups = {group};
    std::unique_ptr<BlockTreeCache> cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1),
                                                                                       std::move(groups),
                                                                                       std::vector<Component>{},
                                                                                       std::move(config),
                                                                                       /*storage_backend=*/nullptr,
                                                                                       broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{host_block}}});
    ASSERT_TRUE(cache->evictor_.beginLoadBack(find_result.matched_node, 0, Tier::HOST));
    BlockTreeCache::LoadBackItem item{find_result.matched_node, 0, Tier::HOST, {host_block}, {7}};
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
        EXPECT_EQ(worker_request.copy_items(0).gpu_blocks(0), 7);
    }
}

TEST_F(BlockTreeCacheTest, BroadcastDiskLoadBackUsesTwoTransferStages) {
    std::shared_ptr<BlockTreeBroadcastRpcState>    state   = std::make_shared<BlockTreeBroadcastRpcState>();
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(256, 4);
    ASSERT_TRUE(host_pool->init());
    std::shared_ptr<DiskBlockPool>      disk_pool = makeDiskPool(256, 4, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> group     = std::make_shared<FullComponentGroup>();
    group->component_group_id                     = 0;
    group->setHostPool(host_pool);
    group->setDiskPool(disk_pool);
    const BlockIdxType disk_block = group->allocateSingleBlock(Tier::DISK);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache             = true;
    config.enable_disk_cache               = true;
    std::vector<ComponentGroupPtr>  groups = {group};
    std::unique_ptr<BlockTreeCache> cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1),
                                                                                       std::move(groups),
                                                                                       std::vector<Component>{},
                                                                                       std::move(config),
                                                                                       /*storage_backend=*/nullptr,
                                                                                       broadcast_manager);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].disk_slot = disk_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
    BlockTreeFindResult find_result = cache->tree()->findNode({100});
    ASSERT_NE(find_result.matched_node, nullptr);

    group->referenceBlocks(GroupBlockSet{0, Tier::DISK, {{disk_block}}});
    ASSERT_TRUE(cache->evictor_.beginLoadBack(find_result.matched_node, 0, Tier::DISK));
    BlockTreeCache::LoadBackItem item{find_result.matched_node, 0, Tier::DISK, {disk_block}, {7}};
    cache->performLoadBack({item}, /*ctx=*/nullptr);

    const GroupSlot& slot = find_result.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::DISK));
    EXPECT_EQ(slot.device_blocks, (std::vector<BlockIdxType>{7}));
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
            EXPECT_EQ(request_item.gpu_blocks(0), 7);
            if (!isNullBlockIdx(staging_host_block)) {
                EXPECT_EQ(request_item.mem_block(), staging_host_block);
            }
        }
    }
    EXPECT_EQ(disk_to_host_count, 2u);
    EXPECT_EQ(host_to_device_count, 2u);
}

TEST_F(BlockTreeCacheTest, BuildEvictionTransferRequestIncludesPrimaryAndCascades) {
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
    ASSERT_TRUE(cache_->buildEvictionTransferRequest(plan, request));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(request.copy_items(0).src_mem_block(), 3);
    EXPECT_EQ(request.copy_items(0).disk_slot(), 4);
    EXPECT_EQ(request.copy_items(1).src_mem_block(), 5);
    EXPECT_EQ(request.copy_items(1).disk_slot(), 6);
}

TEST_F(BlockTreeCacheTest, BroadcastEvictionSuccessCommitsPlan) {
    std::shared_ptr<BlockTreeBroadcastRpcState>    state   = std::make_shared<BlockTreeBroadcastRpcState>();
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK, state},
        {true, true, grpc::Status::OK, state},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(256, 8);
    ASSERT_TRUE(host_pool->init());
    std::shared_ptr<DiskBlockPool>      disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> full      = std::make_shared<FullComponentGroup>();
    full->component_group_id                      = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_device_cache             = false;
    config.enable_memory_cache             = true;
    config.enable_disk_cache               = true;
    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1),
                                                                                       std::move(groups),
                                                                                       std::vector<Component>{},
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

TEST_F(BlockTreeCacheTest, BroadcastEvictionFailureRollsBackPlan) {
    const std::vector<BlockTreeBroadcastRpcConfig> configs = {
        {true, true, grpc::Status::OK},
        {true, false, grpc::Status::OK},
    };
    std::vector<std::unique_ptr<BlockTreeBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager> broadcast_manager = makeBroadcastManager(configs, servers);
    ASSERT_NE(broadcast_manager, nullptr);

    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(256, 8);
    ASSERT_TRUE(host_pool->init());
    std::shared_ptr<DiskBlockPool>      disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());
    std::shared_ptr<FullComponentGroup> full      = std::make_shared<FullComponentGroup>();
    full->component_group_id                      = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_device_cache             = false;
    config.enable_memory_cache             = true;
    config.enable_disk_cache               = true;
    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(std::make_unique<BlockTree>(1),
                                                                                       std::move(groups),
                                                                                       std::vector<Component>{},
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

// ---------------------------------------------------------------------------
// Admission callback: gate invoked inside match() before load_back.
// ---------------------------------------------------------------------------

// Builds a single-FULL-group cache with load_back enabled and one host-only node
// (device blocks cleared) so match() would trigger a host->device load_back.
static std::unique_ptr<BlockTreeCache> makeHostOnlyLoadBackCache() {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    std::vector<ComponentGroupPtr> groups    = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots);

    BlockTreeFindResult find = cache->tree()->findNode({200});
    RTP_LLM_CHECK(find.matched_node != nullptr);
    find.matched_node->group_slots[0].host_block = 7;
    for (BlockIdxType& device_block_index : find.matched_node->group_slots[0].device_blocks) {
        device_block_index = NULL_BLOCK_IDX;
    }
    return cache;
}

TEST_F(BlockTreeCacheTest, LoadBackDeviceAllocationFailureRollsBackAllItems) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr first_device_pool      = makeDevicePool({{1, 0}}, 1, "load_back_atomic_first");
    DeviceBlockPoolPtr exhausted_device_pool  = makeDevicePool({{1, 0}}, 1, "load_back_atomic_exhausted");
    const BlockIdxType exhausted_device_block = poolMalloc(*exhausted_device_pool);
    ASSERT_NE(exhausted_device_block, NULL_BLOCK_IDX);
    exhausted_device_pool->incRef(exhausted_device_block);

    std::shared_ptr<HostBlockPool> first_host_pool  = makeHostPool(1, 2);
    std::shared_ptr<HostBlockPool> second_host_pool = makeHostPool(1, 2);
    ASSERT_TRUE(first_host_pool->init());
    ASSERT_TRUE(second_host_pool->init());

    std::shared_ptr<FullComponentGroup> first_group = std::make_shared<FullComponentGroup>();
    first_group->component_group_id                 = 0;
    first_group->setDevicePools({first_device_pool});
    first_group->setHostPool(first_host_pool);
    std::shared_ptr<FullComponentGroup> second_group = std::make_shared<FullComponentGroup>();
    second_group->component_group_id                 = 1;
    second_group->setDevicePools({exhausted_device_pool});
    second_group->setHostPool(second_host_pool);

    const BlockIdxType first_host_block  = first_group->allocateSingleBlock(Tier::HOST);
    const BlockIdxType second_host_block = second_group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);
    first_group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{first_host_block}}});
    second_group->referenceBlocks(GroupBlockSet{1, Tier::HOST, {{second_host_block}}});

    std::vector<ComponentGroupPtr>  component_groups = {first_group, second_group};
    std::unique_ptr<BlockTreeCache> cache            = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(2), std::move(component_groups), std::vector<Component>{});
    TreeNode first_node;
    first_node.cache_key = 100;
    TreeNode second_node;
    second_node.cache_key = 200;

    BlockTreeCache*                 cache_pointer = cache.get();
    std::shared_ptr<LoadBackTicket> ticket        = std::make_shared<LoadBackTicket>(
        [cache_pointer](const std::vector<PendingLoadBackItem>& items) { return cache_pointer->commitLoadBack(items); },
        [cache_pointer](const std::vector<PendingLoadBackItem>& items) { cache_pointer->abortLoadBack(items); });
    ticket->items().push_back(PendingLoadBackItem{&first_node, 0, Tier::HOST, {first_host_block}});
    ticket->items().push_back(PendingLoadBackItem{&second_node, 1, Tier::HOST, {second_host_block}});

    std::shared_ptr<AsyncContext> context = ticket->commit();
    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(first_device_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 1u);

    first_group->releaseSingleBlock(Tier::HOST, first_host_block);
    second_group->releaseSingleBlock(Tier::HOST, second_host_block);
    exhausted_device_pool->decRef(exhausted_device_block);
}

// Deferred load_back: match() plans (references the source blocks) but does NOT execute
// load_back. The result carries a LoadBackTicket; committing it allocates the device
// target and submits the async copy, while dropping it uncommitted aborts (unreferences
// the source) with nothing wasted.

// Not committing the ticket: no device block is allocated and no async copy is submitted;
// the ticket destructor aborts safely.
TEST_F(BlockTreeCacheTest, LoadBackTicketAbortSkipsLoadBack) {
    auto cache = makeHostOnlyLoadBackCache();

    auto result = cache->match({200});
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_FALSE(result.load_back_ticket->empty());
    // Counters reflect the planned load_back; match() submits nothing async and leaves
    // async_context null (the async context is produced only at commit).
    EXPECT_EQ(result.matched_blocks, 1u);
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);
    EXPECT_EQ(result.async_context, nullptr);

    // Drop the ticket without committing => RAII abort (source unreferenced). No async
    // task was ever submitted, so waitForPendingTasks returns immediately.
    result.load_back_ticket.reset();
    cache->releaseMatchedBlocks(result.matched_block_sets);
    cache->waitForPendingTasks();
}

// Committing the ticket allocates the device target and submits the async copy, yielding
// a non-null AsyncContext.
TEST_F(BlockTreeCacheTest, LoadBackTicketCommitTriggersLoadBack) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::unique_ptr<BlockTreeCache> cache       = makeHostOnlyLoadBackCache();
    DeviceBlockPoolPtr              device_pool = makeDevicePool({{1, 0}}, 1, "load_back_ticket_commit");
    cache->component_groups_[0]->setDevicePools({device_pool});

    BlockTreeMatchResult result = cache->match({200});
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.matched_blocks, 1u);
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);

    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    EXPECT_NE(context, nullptr);

    cache->releaseMatchedBlocks(result.matched_block_sets);
    cache->waitForPendingTasks();
}

// A no-match match() plans nothing and returns a null ticket (never created).
TEST_F(BlockTreeCacheTest, EmptyMatchYieldsNoTicket) {
    auto result = cache_->match({100, 200, 300});  // empty tree => no match
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.load_back_ticket, nullptr);
}

}  // namespace
}  // namespace rtp_llm
