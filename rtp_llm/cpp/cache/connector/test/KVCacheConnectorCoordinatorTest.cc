#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/memory/test/mock/MockKVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnector.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"

namespace rtp_llm {
namespace test {

namespace {

class TestMeta final: public Meta {
public:
    explicit TestMeta(bool enable_memory_cache): enable_memory_cache_(enable_memory_cache) {}
    ~TestMeta() override = default;

    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }

private:
    bool enable_memory_cache_{false};
};

}  // namespace

class KVCacheConnectorCoordinatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();

        cache_config_.layer_num        = 1;
        cache_config_.layer_all_num    = 1;
        cache_config_.block_num        = 10;
        cache_config_.block_size_bytes = 1024;
        cache_config_.dtype            = rtp_llm::TYPE_FP16;
        cache_config_.layer_to_group_id.assign(static_cast<size_t>(cache_config_.layer_all_num), 0);

        kv_cache_config_.memory_cache_size_mb         = 100;
        kv_cache_config_.memory_cache_sync_timeout_ms = 1000;

        device_    = createDevice();
        allocator_ = std::make_shared<MockKVCacheAllocator>(cache_config_, device_);
        // KVCacheConnectorCoordinator::asyncRead/asyncWrite logs free/available blocks via KVCacheAllocator.
        // Those methods assume allocator_->block_pool_ is non-null. In UT we use a mock allocator, so set a
        // minimal BlockPool here to avoid crashes/hangs in tests that exercise coordinator paths.
        {
            // NOTE: use the 4-arg overload to avoid requiring cache_config_.cache_specs in unit tests.
            const size_t block_stride_bytes =
                cache_config_.block_size_bytes / static_cast<size_t>(std::max(1u, cache_config_.layer_all_num));
            auto pool_config = BlockPoolConfigHelper::createConfig(
                cache_config_.layer_all_num, cache_config_.block_num, block_stride_bytes, cache_config_.dtype);
            auto pool = std::make_shared<BlockPool>(pool_config, device_, AllocationType::HOST);
            RTP_LLM_CHECK(pool->init());
            allocator_->block_pool_ = pool;
        }

        coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(
            cache_config_, kv_cache_config_, runtime_config_, allocator_, device_);
    }

    // In production, KVCacheAllocator::incrKVCacheRef() typically returns a shared_ptr with a custom deleter that
    // decrements the ref-count via KVCacheAllocator::decrKVCacheRef(). Our gmock allocator does not provide that,
    // so tests that validate ref-counting must simulate it explicitly.
    std::shared_ptr<KVCacheResource> makeResourceWithAutoDecr() {
        // IMPORTANT: Use weak_ptr to avoid a reference cycle:
        // allocator_ (mock) -> EXPECT_CALL action -> returned resource -> deleter -> allocator_.
        std::weak_ptr<MockKVCacheAllocator> allocator_weak = allocator_;
        auto                                owned          = std::make_shared<KVCacheResource>();
        return std::shared_ptr<KVCacheResource>(owned.get(), [owned, allocator_weak](KVCacheResource*) mutable {
            if (auto allocator = allocator_weak.lock()) {
                allocator->decrKVCacheRef(*owned);
            }
            owned.reset();
        });
    }

    void TearDown() override {
        if (coordinator_) {
            // Ensure all internal contexts/connectors are released before gmock leak checker runs at program exit.
            coordinator_->stop_.store(true);
            {
                std::lock_guard<std::mutex> lock(coordinator_->update_mutex_);
                coordinator_->fused_async_read_context_list_.clear();
                coordinator_->fused_async_write_context_list_.clear();
            }
            if (coordinator_->update_thread_) {
                coordinator_->update_thread_->stop();
                coordinator_->update_thread_.reset();  // break shared_from_this cycle if any
            }
            coordinator_->connectors_.clear();
            coordinator_.reset();
        }
        allocator_.reset();
        device_ = nullptr;
    }

private:
    DeviceBase* createDevice() const {
        DeviceResourceConfig device_resource_config;
        // On shared GPUs, free memory can be < 1GB. The default in DeviceResourceConfig is -1GB,
        // which can make target_track_bytes negative and throw in TrackerAllocator.
        // Use 0 to fall back to DeviceFactory default (-512MB).
        device_resource_config.device_reserve_memory_bytes = 2048000000;
        device_resource_config.host_reserve_memory_bytes   = 2048000000;
        ModelSpecificConfig model_specific_config;

        DeviceFactory::initDevices(ParallelismConfig{},
                                   ModelConfig{},
                                   EPLBConfig{},
                                   FMHAConfig{},
                                   device_resource_config,
                                   MoeConfig{},
                                   SpeculativeExecutionConfig{},
                                   MiscellaneousConfig{},
                                   ProfilingDebugLoggingConfig{},
                                   HWKernelConfig{},
                                   ConcurrencyConfig{},
                                   FfnDisAggregateConfig{},
                                   RuntimeConfig{},
                                   model_specific_config);
        return DeviceFactory::getDefaultDevice();
    }

    std::shared_ptr<FusedAsyncReadContext> makeFusedReadContextAndExpectAsyncRead() {
        // Coordinator now uses connector index order (vector) and passes read range explicitly.
        auto connector0           = std::make_shared<MockKVCacheConnector>();
        auto connector1           = std::make_shared<MockKVCacheConnector>();
        coordinator_->connectors_ = {connector0, connector1};

        auto resource = std::make_shared<KVCacheResource>();
        resource->setDeviceReuseBlockNum(1);

        auto match_ctx0 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
        auto match_ctx1 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();

        ON_CALL(*match_ctx0, done()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx0, success()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx0, matchedBlockCount()).WillByDefault(testing::Return(3));

        ON_CALL(*match_ctx1, done()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx1, success()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx1, matchedBlockCount()).WillByDefault(testing::Return(5));

        auto read_ctx0 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
        auto read_ctx1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
        ON_CALL(*read_ctx0, done()).WillByDefault(testing::Return(false));
        ON_CALL(*read_ctx0, success()).WillByDefault(testing::Return(true));
        ON_CALL(*read_ctx1, done()).WillByDefault(testing::Return(false));
        ON_CALL(*read_ctx1, success()).WillByDefault(testing::Return(true));

        auto meta = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);

        testing::InSequence s;
        EXPECT_CALL(*connector0,
                    asyncRead(testing::Eq(resource),
                              testing::Eq(meta),
                              testing::Truly([match_ctx0](const std::shared_ptr<AsyncMatchContext>& mc) {
                                  return mc.get() == match_ctx0.get();
                              }),
                              /*start_read_block_index=*/1,
                              /*read_block_num=*/2))
            .WillOnce(testing::Return(read_ctx0));
        EXPECT_CALL(*connector1,
                    asyncRead(testing::Eq(resource),
                              testing::Eq(meta),
                              testing::Truly([match_ctx1](const std::shared_ptr<AsyncMatchContext>& mc) {
                                  return mc.get() == match_ctx1.get();
                              }),
                              /*start_read_block_index=*/3,
                              /*read_block_num=*/2))
            .WillOnce(testing::Return(read_ctx1));

        auto fused_match_ctx =
            std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_ctx0, match_ctx1});
        return std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource, meta);
    }

private:
    CacheConfig                                  cache_config_;
    KVCacheConfig                                kv_cache_config_;
    RuntimeConfig                                runtime_config_;
    DeviceBase*                                  device_{nullptr};
    std::shared_ptr<MockKVCacheAllocator>        allocator_;
    std::shared_ptr<KVCacheConnectorCoordinator> coordinator_;
};

TEST_F(KVCacheConnectorCoordinatorTest, Init_ReturnFalse_WhenMemoryConfigInvalid) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.layer_all_num    = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;
    cache_config.layer_to_group_id.assign(static_cast<size_t>(cache_config.layer_all_num), 0);

    kv_cache_config.enable_memory_cache = true;
    kv_cache_config.reuse_cache = true;  // coordinator init only enables memory connector when reuse_cache is true
    kv_cache_config.memory_cache_size_mb         = 1;
    kv_cache_config.memory_cache_sync_timeout_ms = 0;  // invalid => RTP_LLM_CHECK throws

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, allocator_, device_);

    EXPECT_THROW(coordinator->init(), std::runtime_error);
    EXPECT_EQ(coordinator->update_thread_, nullptr);  // should not start update thread if memory init failed
}

TEST_F(KVCacheConnectorCoordinatorTest, Init_ReturnTrue_WhenMemorySkipped_AndStopsUpdateThread) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.layer_all_num    = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;
    cache_config.layer_to_group_id.assign(static_cast<size_t>(cache_config.layer_all_num), 0);

    kv_cache_config.enable_memory_cache = false;  // skip memory connector in init

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, allocator_, device_);

    EXPECT_TRUE(coordinator->init());
    ASSERT_NE(coordinator->update_thread_, nullptr);
    coordinator->update_thread_->stop();
    coordinator->update_thread_.reset();  // break shared_ptr cycle from shared_from_this()
}

TEST_F(KVCacheConnectorCoordinatorTest, Init_ReturnFalse_WhenMemoryEnabledButSizeInvalid) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.layer_all_num    = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;
    cache_config.layer_to_group_id.assign(static_cast<size_t>(cache_config.layer_all_num), 0);

    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.reuse_cache                  = true;
    kv_cache_config.memory_cache_size_mb         = 0;     // invalid
    kv_cache_config.memory_cache_sync_timeout_ms = 1000;  // valid

    // Even with empty worker_grpc_addrs, init should fail early due to invalid size.
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, allocator_, device_);

    EXPECT_THROW(coordinator->init(), std::runtime_error);
    EXPECT_EQ(coordinator->update_thread_, nullptr);  // should not start update thread if memory init failed
    EXPECT_TRUE(coordinator->connectors_.empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, Init_ReturnTrue_WhenMemoryEnabled_HappyPath_AndStopsUpdateThread) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num     = 1;
    cache_config.layer_all_num = 1;
    cache_config.block_num     = 1;
    // Keep block size reasonably large so block_num doesn't explode in createBlockPool().
    cache_config.block_size_bytes = 1024;
    cache_config.dtype            = rtp_llm::TYPE_FP16;
    cache_config.layer_to_group_id.assign(static_cast<size_t>(cache_config.layer_all_num), 0);

    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.reuse_cache                  = true;
    kv_cache_config.memory_cache_size_mb         = 1;
    kv_cache_config.memory_cache_sync_timeout_ms = 1;
    runtime_config.worker_grpc_addrs             = {"127.0.0.1:12345"};

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, allocator_, device_);

    EXPECT_TRUE(coordinator->init());
    ASSERT_NE(coordinator->update_thread_, nullptr);
    ASSERT_EQ(coordinator->connectors_.size(), 1u);

    coordinator->update_thread_->stop();
    coordinator->update_thread_.reset();  // break shared_ptr cycle from shared_from_this()
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenStop) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.layer_all_num    = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;
    cache_config.layer_to_group_id.assign(static_cast<size_t>(cache_config.layer_all_num), 0);

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    coordinator->stop_.store(true);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_EQ(coordinator->asyncRead(ctx), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenConnectorContextNull) {
    coordinator_->allocator_ = allocator_;
    auto ctx                 = coordinator_->asyncRead(/*connector_context=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenCacheKeysEmpty) {
    auto mock_connector       = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {mock_connector};
    coordinator_->allocator_  = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    // leave cacheKeys empty to hit the early return
    auto                  rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    std::shared_ptr<Meta> meta   = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(meta));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncMatch(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncRead(rw_ctx);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenIncrKVCacheRefReturnsNull) {
    auto mock_connector       = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {mock_connector};
    coordinator_->allocator_  = allocator_;

    // Coordinator logs free/available blocks before calling incrKVCacheRef().
    // MockKVCacheAllocator doesn't initialize its internal BlockPool unless we set it up explicitly.
    // Without this, allocator_->freeBlocksNum() / availableBlocksNum() will dereference a null BlockPool and
    // the test process can crash/hang.
    {
        auto pool_config = BlockPoolConfigHelper::createConfig(cache_config_.layer_all_num,
                                                               /*block_num=*/1,
                                                               /*block_stride_bytes=*/cache_config_.block_size_bytes,
                                                               /*dtype=*/cache_config_.dtype);
        auto pool        = std::make_shared<BlockPool>(pool_config, device_, AllocationType::HOST);
        ASSERT_TRUE(pool->init());
        allocator_->block_pool_ = pool;
    }

    KVCacheResource resource;
    resource.initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto                  rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    std::shared_ptr<Meta> meta   = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(meta));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(nullptr));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncMatch(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncRead(rw_ctx);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnFusedContext_WhenNoConnectors) {
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config_, KVCacheConfig{}, RuntimeConfig{}, allocator_, device_);
    coordinator->connectors_.clear();  // explicitly ensure "no connectors registered"

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});

    // No connectors registered: asyncRead() still returns a fused read context; it will contain zero match contexts
    // and will be processed/cleaned up by the coordinator update loop if enabled.
    // Use a plain shared_ptr here to avoid custom-deleter side effects in this no-connector path.
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    // Don't let gmock keep a ref to `resource` until program exit.
    // gmock actions are stored as const; use a shared holder to release the ref after first call.
    auto resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *resource_holder;
            resource_holder->reset();
            return out;
        }));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    std::shared_ptr<Meta> meta = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    ON_CALL(*ctx, meta()).WillByDefault(testing::ReturnRef(meta));

    // No connectors registered => returns a fused context and enqueues it.
    auto async_ctx = coordinator->asyncRead(ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_read_context_list_.size(), 1u);

    // Important: coordinator dtor waits until both lists become empty. Clear under lock to avoid races with the dtor.
    {
        std::lock_guard<std::mutex> lock(coordinator->update_mutex_);
        coordinator->fused_async_read_context_list_.clear();
        coordinator->fused_async_write_context_list_.clear();
    }
    async_ctx.reset();
    coordinator.reset();
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnContextAndEnqueue_WhenHasMatchContext) {
    // Use fixture allocator_/device_ so allocator has a valid BlockPool.
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config_, KVCacheConfig{}, RuntimeConfig{}, allocator_, device_);

    auto mock_connector      = std::make_shared<testing::NiceMock<MockKVCacheConnector>>();
    coordinator->connectors_ = {mock_connector};

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = makeResourceWithAutoDecr();

    // Don't let gmock keep a ref to `resource` until program exit.
    auto resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *resource_holder;
            resource_holder->reset();
            return out;
        }));

    auto match_ctx = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    EXPECT_CALL(*mock_connector, asyncMatch(testing::Eq(resource), testing::_)).WillOnce(testing::Return(match_ctx));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    std::shared_ptr<Meta> null_meta;
    ON_CALL(*ctx, meta()).WillByDefault(testing::ReturnRef(null_meta));

    auto async_ctx = coordinator->asyncRead(ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_read_context_list_.size(), 1);

    // Release: clear list and release returned ctx to trigger deleter => decrKVCacheRef once.
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    coordinator->fused_async_read_context_list_.clear();
    async_ctx.reset();
    coordinator->connectors_.clear();
    mock_connector.reset();
    resource.reset();     // trigger auto-decr while allocator_ is alive
    coordinator.reset();  // ensure no lingering references before next tests
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenStop) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.layer_all_num    = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;
    cache_config.layer_to_group_id.assign(static_cast<size_t>(cache_config.layer_all_num), 0);

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    coordinator->stop_.store(true);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_EQ(coordinator->asyncWrite(ctx), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenConnectorContextNull) {
    coordinator_->allocator_ = allocator_;
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    auto ctx = coordinator_->asyncWrite(/*connector_context=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenCacheKeysEmpty) {
    auto mock_connector       = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {mock_connector};
    coordinator_->allocator_  = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    // leave cacheKeys empty
    auto                  rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    std::shared_ptr<Meta> meta   = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(meta));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncWrite(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncWrite(rw_ctx);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenIncrKVCacheRefReturnsNull) {
    auto mock_connector       = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {mock_connector};
    coordinator_->allocator_  = allocator_;

    // Build a connector context with non-empty cache keys.
    auto ctx_resource = std::make_shared<KVCacheResource>();
    ctx_resource->initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    ctx_resource->cacheKeys()    = CacheKeysType{1, 2, 3};
    auto                  rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    std::shared_ptr<Meta> meta   = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(meta));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(*ctx_resource));

    // Simulate allocator refusing to create a referenced resource (e.g. no valid blocks).
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(nullptr));
    // Must not call decr on a null resource.
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(0);
    // Must not call connector->asyncWrite with a null resource.
    EXPECT_CALL(*mock_connector, asyncWrite(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncWrite(rw_ctx);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnFusedContext_WhenMemoryCacheDisabledInMeta) {
    auto mock_connector       = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {mock_connector};
    coordinator_->allocator_  = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource        = makeResourceWithAutoDecr();
    auto selected_resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(selected_resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([selected_resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *selected_resource_holder;
            selected_resource_holder->reset();
            return out;
        }));
    EXPECT_CALL(*mock_connector, asyncWrite(testing::Eq(selected_resource), testing::_))
        .WillOnce(testing::Return(nullptr));

    auto                  rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    std::shared_ptr<Meta> meta   = std::make_shared<TestMeta>(/*enable_memory_cache=*/false);
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(meta));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    auto async_ctx = coordinator_->asyncWrite(rw_ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator_->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    coordinator_->fused_async_write_context_list_.clear();
    async_ctx.reset();
    coordinator_->connectors_.clear();
    mock_connector.reset();
    selected_resource.reset();  // trigger auto-decr while allocator_ is alive
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnFusedContext_WhenConnectorReturnsNullContext) {
    auto mock_connector       = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {mock_connector};
    coordinator_->allocator_  = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource        = makeResourceWithAutoDecr();
    auto selected_resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(selected_resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([selected_resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *selected_resource_holder;
            selected_resource_holder->reset();
            return out;
        }));
    EXPECT_CALL(*mock_connector, asyncWrite(testing::Eq(selected_resource), testing::_))
        .WillOnce(testing::Return(nullptr));

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));
    std::shared_ptr<Meta> null_meta;
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(null_meta));

    auto async_ctx = coordinator_->asyncWrite(rw_ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator_->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    coordinator_->fused_async_write_context_list_.clear();
    async_ctx.reset();
    coordinator_->connectors_.clear();
    mock_connector.reset();
    selected_resource.reset();  // trigger auto-decr while allocator_ is alive
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnFusedContext_WhenNoConnectors) {
    // No connectors registered: asyncWrite() still returns a fused context (with zero inner contexts) and enqueues it.
    // NOTE: coordinator->connectors_ is not expected to contain nullptr in production; keep the test aligned with the
    // current contract and avoid dereferencing null connectors.
    coordinator_->connectors_.clear();
    coordinator_->allocator_ = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, cache_config_.layer_all_num, cache_config_.layer_to_group_id);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource        = makeResourceWithAutoDecr();
    auto selected_resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(selected_resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([selected_resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *selected_resource_holder;
            selected_resource_holder->reset();
            return out;
        }));

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));
    std::shared_ptr<Meta> null_meta;
    ON_CALL(*rw_ctx, meta()).WillByDefault(testing::ReturnRef(null_meta));

    auto async_ctx = coordinator_->asyncWrite(rw_ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator_->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    coordinator_->fused_async_write_context_list_.clear();
    async_ctx.reset();
    coordinator_->connectors_.clear();
    selected_resource.reset();  // trigger auto-decr while allocator_ is alive
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnFusedContext_WhenNoConnectors_NewCoordinator) {
    // Use fixture allocator_/device_ so allocator has a valid BlockPool.
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config_, KVCacheConfig{}, RuntimeConfig{}, allocator_, device_);

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = makeResourceWithAutoDecr();

    auto resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *resource_holder;
            resource_holder->reset();
            return out;
        }));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    std::shared_ptr<Meta> null_meta;
    ON_CALL(*ctx, meta()).WillByDefault(testing::ReturnRef(null_meta));

    auto async_ctx = coordinator->asyncWrite(ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    coordinator->fused_async_write_context_list_.clear();
    async_ctx.reset();
    coordinator->connectors_.clear();
    resource.reset();     // trigger auto-decr while allocator_ is alive
    coordinator.reset();  // ensure no lingering references before next tests
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnContextAndEnqueue_WhenHasWriteContext) {
    // Use fixture allocator_/device_ so allocator has a valid BlockPool.
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config_, KVCacheConfig{}, RuntimeConfig{}, allocator_, device_);

    auto mock_connector      = std::make_shared<testing::NiceMock<MockKVCacheConnector>>();
    coordinator->connectors_ = {mock_connector};

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = makeResourceWithAutoDecr();

    auto resource_holder = std::make_shared<std::shared_ptr<KVCacheResource>>(resource);
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_))
        .WillOnce(testing::Invoke([resource_holder](const KVCacheResource&, const CacheKeysType&) {
            auto out = *resource_holder;
            resource_holder->reset();
            return out;
        }));

    auto write_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_connector, asyncWrite(testing::Eq(resource), testing::_)).WillOnce(testing::Return(write_ctx));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    std::shared_ptr<Meta> null_meta;
    ON_CALL(*ctx, meta()).WillByDefault(testing::ReturnRef(null_meta));

    auto async_ctx = coordinator->asyncWrite(ctx);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    coordinator->fused_async_write_context_list_.clear();
    async_ctx.reset();
    coordinator->connectors_.clear();
    mock_connector.reset();
    resource.reset();  // trigger auto-decr while allocator_ is alive
    write_ctx.reset();
    coordinator.reset();  // ensure no lingering references before next tests
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_SkipsNonMatchAndUpdatesReuseAndSetsFusedReadContext) {
    auto fused_read_ctx = makeFusedReadContextAndExpectAsyncRead();
    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 2);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_SetsEmptyFusedReadContext_WhenNoAsyncReadTriggered) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setDeviceReuseBlockNum(5);

    // One non-match context should be skipped by dynamic_pointer_cast.
    auto non_match_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();

    // One match context with matchedBlockCount <= reuse should be skipped.
    auto match_ctx = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*match_ctx, matchedBlockCount()).WillByDefault(testing::Return(5));

    // Keep connector count aligned with match contexts.
    coordinator_->connectors_ = {nullptr, nullptr};

    auto fused_match_ctx =
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{non_match_ctx, match_ctx});
    auto meta           = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource, meta);

    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_TRUE(fused_read_ctx->fusedReadContext()->contexts().empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_DoesNotUpdateReuse_WhenAsyncReadReturnsNull) {
    auto connector0           = std::make_shared<MockKVCacheConnector>();
    auto connector1           = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {connector0, connector1};

    auto resource = std::make_shared<KVCacheResource>();
    resource->setDeviceReuseBlockNum(1);

    auto match_ctx1 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    auto match_ctx2 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*match_ctx1, matchedBlockCount()).WillByDefault(testing::Return(3));
    ON_CALL(*match_ctx2, matchedBlockCount()).WillByDefault(testing::Return(5));

    auto fused_match_ctx =
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_ctx1, match_ctx2});
    auto meta           = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource, meta);

    auto read_ctx2 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_ctx2, done()).WillByDefault(testing::Return(false));
    ON_CALL(*read_ctx2, success()).WillByDefault(testing::Return(true));

    testing::InSequence s;
    // First match triggers asyncRead but returns nullptr -> reuse_num must NOT update.
    EXPECT_CALL(*connector0, asyncRead(testing::Eq(resource), testing::Eq(meta), testing::_, 1, 2))
        .WillOnce(testing::Return(std::shared_ptr<AsyncContext>{nullptr}));
    // Second should still use reuse=1 (not 3): start=1, size=4 (matched=5).
    EXPECT_CALL(*connector1, asyncRead(testing::Eq(resource), testing::Eq(meta), testing::_, 1, 4))
        .WillOnce(testing::Return(read_ctx2));

    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 1);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_UsesConnectorByIndex) {
    auto connector0           = std::make_shared<MockKVCacheConnector>();
    auto connector1           = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_ = {connector0, connector1};

    auto resource = std::make_shared<KVCacheResource>();
    resource->setDeviceReuseBlockNum(0);

    auto match0 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    auto match1 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*match0, matchedBlockCount()).WillByDefault(testing::Return(2));
    ON_CALL(*match1, matchedBlockCount()).WillByDefault(testing::Return(4));

    auto fused_match_ctx =
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match0, match1});
    auto meta           = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource, meta);

    auto read0 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto read1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read0, done()).WillByDefault(testing::Return(false));
    ON_CALL(*read0, success()).WillByDefault(testing::Return(true));
    ON_CALL(*read1, done()).WillByDefault(testing::Return(false));
    ON_CALL(*read1, success()).WillByDefault(testing::Return(true));

    testing::InSequence s;
    EXPECT_CALL(*connector0, asyncRead(testing::Eq(resource), testing::Eq(meta), testing::_, 0, 2))
        .WillOnce(testing::Return(read0));
    EXPECT_CALL(*connector1, asyncRead(testing::Eq(resource), testing::Eq(meta), testing::_, 2, 2))
        .WillOnce(testing::Return(read1));

    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 2);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_Throws_WhenSizeMismatch) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setDeviceReuseBlockNum(0);

    // connectors_ is empty but match contexts has one entry -> should fail check.
    coordinator_->connectors_.clear();
    auto missing = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*missing, matchedBlockCount()).WillByDefault(testing::Return(1));

    auto fused_match_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{missing});
    auto meta            = std::make_shared<TestMeta>(/*enable_memory_cache=*/true);
    auto fused_read_ctx  = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource, meta);

    EXPECT_THROW(coordinator_->asyncReadAfterMatch(fused_read_ctx), std::runtime_error);
}

}  // namespace test
}  // namespace rtp_llm