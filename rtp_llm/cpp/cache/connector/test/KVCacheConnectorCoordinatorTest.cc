#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/memory/test/mock/MockKVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnector.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

class KVCacheConnectorCoordinatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        cache_config_.layer_num        = 1;
        cache_config_.block_num        = 10;
        cache_config_.block_size_bytes = 1024;

        kv_cache_config_.memory_block_cache_size_mb         = 100;
        kv_cache_config_.memory_block_cache_sync_timeout_ms = 1000;

        device_    = createDevice();
        allocator_ = std::make_shared<MockKVCacheAllocator>(cache_config_, device_);

        coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(
            cache_config_, kv_cache_config_, runtime_config_, nullptr, nullptr);
    }

    void TearDown() override {
        if (coordinator_) {
            coordinator_->fused_async_read_context_list_.clear();
            coordinator_->fused_async_write_context_list_.clear();
        }
    }

private:
    DeviceBase* createDevice() const {
        DeviceFactory::initDevices(ParallelismConfig{},
                                   ModelConfig{},
                                   EPLBConfig{},
                                   FMHAConfig{},
                                   DeviceResourceConfig{},
                                   MoeConfig{},
                                   SpeculativeExecutionConfig{},
                                   MiscellaneousConfig{},
                                   ProfilingDebugLoggingConfig{},
                                   HWKernelConfig{},
                                   ConcurrencyConfig{},
                                   FfnDisAggregateConfig{},
                                   RuntimeConfig{});
        return DeviceFactory::getDefaultDevice();
    }

    std::shared_ptr<FusedAsyncReadContext> makeFusedReadContextAndExpectAsyncRead() {
        auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
        coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;

        auto resource = std::make_shared<KVCacheResource>();
        resource->setReuseBlocksNum(1);

        auto non_match_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
        ON_CALL(*non_match_ctx, done()).WillByDefault(testing::Return(true));
        ON_CALL(*non_match_ctx, success()).WillByDefault(testing::Return(true));

        auto match_ctx1  = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
        auto match_ctx2  = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
        auto small_match = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();

        // small_match: matched <= reuse => skipped
        ON_CALL(*small_match, done()).WillByDefault(testing::Return(true));
        ON_CALL(*small_match, success()).WillByDefault(testing::Return(true));
        ON_CALL(*small_match, matchedBlockCount()).WillByDefault(testing::Return(1));
        ON_CALL(*small_match, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));

        ON_CALL(*match_ctx1, done()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx1, success()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx1, matchedBlockCount()).WillByDefault(testing::Return(3));
        ON_CALL(*match_ctx1, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));

        ON_CALL(*match_ctx2, done()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx2, success()).WillByDefault(testing::Return(true));
        ON_CALL(*match_ctx2, matchedBlockCount()).WillByDefault(testing::Return(5));
        ON_CALL(*match_ctx2, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));

        auto read_ctx1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
        auto read_ctx2 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
        ON_CALL(*read_ctx1, done()).WillByDefault(testing::Return(false));
        ON_CALL(*read_ctx1, success()).WillByDefault(testing::Return(true));
        ON_CALL(*read_ctx2, done()).WillByDefault(testing::Return(false));
        ON_CALL(*read_ctx2, success()).WillByDefault(testing::Return(true));

        testing::InSequence s;
        EXPECT_CALL(
            *mock_connector,
            asyncRead(testing::Eq(resource),
                      testing::Truly([](const std::shared_ptr<KVCacheConnector::Meta>& meta) {
                          const auto [start, size] = meta->blockRange();
                          return start == 1 && size == 2;
                      }),
                      testing::Truly([match_ctx1](const std::shared_ptr<KVCacheConnector::AsyncMatchContext>& mc) {
                          return mc.get() == match_ctx1.get();
                      })))
            .WillOnce(testing::Return(read_ctx1));
        EXPECT_CALL(
            *mock_connector,
            asyncRead(testing::Eq(resource),
                      testing::Truly([](const std::shared_ptr<KVCacheConnector::Meta>& meta) {
                          const auto [start, size] = meta->blockRange();
                          return start == 3 && size == 2;
                      }),
                      testing::Truly([match_ctx2](const std::shared_ptr<KVCacheConnector::AsyncMatchContext>& mc) {
                          return mc.get() == match_ctx2.get();
                      })))
            .WillOnce(testing::Return(read_ctx2));

        auto fused_match_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{
            non_match_ctx,
            small_match,
            match_ctx1,
            match_ctx2,
        });
        auto fused_read_ctx  = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource);
        return fused_read_ctx;
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
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    kv_cache_config.enable_memory_cache                = true;
    kv_cache_config.memory_block_cache_size_mb         = 1;
    kv_cache_config.memory_block_cache_sync_timeout_ms = 0;  // invalid => initMemoryConnector() returns false

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, std::shared_ptr<KVCacheAllocator>{}, nullptr);

    EXPECT_FALSE(coordinator->init());
    EXPECT_EQ(coordinator->update_thread_, nullptr);  // should not start update thread if memory init failed
}

TEST_F(KVCacheConnectorCoordinatorTest, Init_ReturnTrue_WhenMemorySkipped_AndStopsUpdateThread) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    kv_cache_config.enable_memory_cache = false;  // skip memory connector in init

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, std::shared_ptr<KVCacheAllocator>{}, nullptr);

    EXPECT_TRUE(coordinator->init());
    ASSERT_NE(coordinator->update_thread_, nullptr);
    coordinator->update_thread_->stop();
    coordinator->update_thread_.reset();  // break shared_ptr cycle from shared_from_this()
}

TEST_F(KVCacheConnectorCoordinatorTest, InitMemoryConnector_ReturnFalse_WhenSizeInvalid) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    kv_cache_config.enable_memory_cache                = true;
    kv_cache_config.memory_block_cache_size_mb         = 0;     // invalid
    kv_cache_config.memory_block_cache_sync_timeout_ms = 1000;  // valid

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, std::shared_ptr<KVCacheAllocator>{}, nullptr);

    EXPECT_FALSE(coordinator->initMemoryConnector());
    EXPECT_EQ(coordinator->memory_connector_, nullptr);
    EXPECT_EQ(coordinator->connectors_.count(KVCacheConnector::ConnectorType::Memory), 0);
}

TEST_F(KVCacheConnectorCoordinatorTest, InitMemoryConnector_ReturnFalse_WhenTimeoutInvalid) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    kv_cache_config.memory_block_cache_size_mb         = 1;  // valid (>0)
    kv_cache_config.memory_block_cache_sync_timeout_ms = 0;  // invalid

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, std::shared_ptr<KVCacheAllocator>{}, nullptr);

    EXPECT_FALSE(coordinator->initMemoryConnector());
    EXPECT_EQ(coordinator->memory_connector_, nullptr);
    EXPECT_EQ(coordinator->connectors_.count(KVCacheConnector::ConnectorType::Memory), 0);
}

TEST_F(KVCacheConnectorCoordinatorTest, InitMemoryConnector_ReturnTrue) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    kv_cache_config.memory_block_cache_size_mb         = 1;
    kv_cache_config.memory_block_cache_sync_timeout_ms = 1;

    // KVCacheMemoryConnector::init() requires non-empty tp_addrs_ (TpBroadcastManager::init checks this).
    runtime_config.worker_grpc_addrs = {"127.0.0.1:12345"};

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, std::shared_ptr<KVCacheAllocator>{}, nullptr);

    EXPECT_TRUE(coordinator->initMemoryConnector());
    ASSERT_NE(coordinator->memory_connector_, nullptr);
    ASSERT_EQ(coordinator->connectors_.count(KVCacheConnector::ConnectorType::Memory), 1);
    EXPECT_EQ(coordinator->connectors_.at(KVCacheConnector::ConnectorType::Memory).get(),
              coordinator->memory_connector_.get());
}

TEST_F(KVCacheConnectorCoordinatorTest, InitUpdateThread_ReturnTrue_AndCanStop) {
    CacheConfig   cache_config;
    KVCacheConfig kv_cache_config;
    RuntimeConfig runtime_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    kv_cache_config.memory_block_cache_size_mb = 0;  // skip memory connector in init()

    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, kv_cache_config, runtime_config, std::shared_ptr<KVCacheAllocator>{}, nullptr);

    EXPECT_TRUE(coordinator->initUpdateThread());
    ASSERT_NE(coordinator->update_thread_, nullptr);
    coordinator->update_thread_->stop();
    coordinator->update_thread_.reset();  // break shared_ptr cycle from shared_from_this()
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenStop) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    coordinator->stop_.store(true);

    auto ctx  = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    auto meta = std::shared_ptr<KVCacheConnectorCoordinator::Meta>{nullptr};

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_EQ(coordinator->asyncRead(ctx, meta), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenAllocatorNull) {
    coordinator_->allocator_.reset();
    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    auto ctx    = coordinator_->asyncRead(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenConnectorContextNull) {
    coordinator_->allocator_ = allocator_;
    auto ctx                 = coordinator_->asyncRead(/*connector_context=*/nullptr, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenCacheKeysEmpty) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;
    coordinator_->allocator_                                           = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    // leave cacheKeys empty to hit the early return
    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncMatch(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncRead(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenIncrKVCacheRefReturnsNull) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;
    coordinator_->allocator_                                           = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(nullptr));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncMatch(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncRead(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnNull_WhenNoMatchContexts) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});

    auto resource = std::make_shared<KVCacheResource>();
    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(resource));
    EXPECT_CALL(*allocator, decrKVCacheRef(testing::_)).Times(1);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    ON_CALL(*ctx, enableMemoryCache()).WillByDefault(testing::Return(true));

    // No connectors registered => contexts empty => decr + nullptr
    EXPECT_EQ(coordinator->asyncRead(ctx, nullptr), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncRead_ReturnContextAndEnqueue_WhenHasMatchContext) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    auto mock_connector = std::make_shared<testing::NiceMock<MockKVCacheConnector>>();
    coordinator->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = std::make_shared<KVCacheResource>();

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(resource));

    auto match_ctx = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    EXPECT_CALL(*mock_connector, asyncMatch(testing::Eq(resource), testing::_)).WillOnce(testing::Return(match_ctx));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    ON_CALL(*ctx, enableMemoryCache()).WillByDefault(testing::Return(true));

    auto async_ctx = coordinator->asyncRead(ctx, nullptr);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_read_context_list_.size(), 1);

    // Release: clear list and release returned ctx to trigger deleter => decrKVCacheRef once.
    EXPECT_CALL(*allocator, decrKVCacheRef(testing::_)).Times(1);
    coordinator->fused_async_read_context_list_.clear();
    async_ctx.reset();
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenStop) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    coordinator->stop_.store(true);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_EQ(coordinator->asyncWrite(ctx, nullptr), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenAllocatorNull) {
    coordinator_->allocator_.reset();
    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    auto ctx    = coordinator_->asyncWrite(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenConnectorContextNull) {
    coordinator_->allocator_ = allocator_;
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    auto ctx = coordinator_->asyncWrite(/*connector_context=*/nullptr, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenCacheKeysEmpty) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;
    coordinator_->allocator_                                           = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    // leave cacheKeys empty
    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncWrite(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncWrite(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenIncrKVCacheRefReturnsNull) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;
    coordinator_->allocator_                                           = allocator_;

    // Build a connector context with non-empty cache keys.
    auto ctx_resource = std::make_shared<KVCacheResource>();
    ctx_resource->initGroups(1, /*layer_num=*/1);
    ctx_resource->cacheKeys() = CacheKeysType{1, 2, 3};
    auto rw_ctx               = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(*ctx_resource));

    // Simulate allocator refusing to create a referenced resource (e.g. no valid blocks).
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(nullptr));
    // Must not call decr on a null resource.
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(0);
    // Must not call connector->asyncWrite with a null resource.
    EXPECT_CALL(*mock_connector, asyncWrite(testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncWrite(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenEnableMemoryCacheFalse) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;
    coordinator_->allocator_                                           = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource = std::make_shared<KVCacheResource>();
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(selected_resource));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    EXPECT_CALL(*mock_connector, asyncWrite(testing::_, testing::_)).Times(0);

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(false));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    auto ctx = coordinator_->asyncWrite(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenConnectorReturnsNullContext) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;
    coordinator_->allocator_                                           = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource = std::make_shared<KVCacheResource>();
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(selected_resource));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    EXPECT_CALL(*mock_connector, asyncWrite(testing::Eq(selected_resource), testing::_))
        .WillOnce(testing::Return(nullptr));

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    auto ctx = coordinator_->asyncWrite(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenMemoryConnectorNull) {
    // Connector entry exists but is null; should be skipped.
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = nullptr;
    coordinator_->allocator_                                           = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource = std::make_shared<KVCacheResource>();
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(selected_resource));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    auto ctx = coordinator_->asyncWrite(rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnNull_WhenNoWriteContexts) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = std::make_shared<KVCacheResource>();

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(resource));
    EXPECT_CALL(*allocator, decrKVCacheRef(testing::_)).Times(1);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    ON_CALL(*ctx, enableMemoryCache()).WillByDefault(testing::Return(true));

    EXPECT_EQ(coordinator->asyncWrite(ctx, nullptr), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWrite_ReturnContextAndEnqueue_WhenHasWriteContext) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    auto mock_connector = std::make_shared<testing::NiceMock<MockKVCacheConnector>>();
    coordinator->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = std::make_shared<KVCacheResource>();

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(resource));

    auto write_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_connector, asyncWrite(testing::Eq(resource), testing::_)).WillOnce(testing::Return(write_ctx));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));
    ON_CALL(*ctx, enableMemoryCache()).WillByDefault(testing::Return(true));

    auto async_ctx = coordinator->asyncWrite(ctx, nullptr);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator, decrKVCacheRef(testing::_)).Times(1);
    coordinator->fused_async_write_context_list_.clear();
    async_ctx.reset();
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenStop) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    coordinator->stop_.store(true);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_EQ(coordinator->asyncWriteByLayer(0, ctx, nullptr), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenAllocatorNull) {
    coordinator_->allocator_.reset();
    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    auto ctx    = coordinator_->asyncWriteByLayer(/*layer_id=*/0, rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenConnectorContextNull) {
    coordinator_->allocator_ = allocator_;
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    auto ctx = coordinator_->asyncWriteByLayer(/*layer_id=*/0, /*connector_context=*/nullptr, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenCacheKeysEmpty) {
    auto mock_connector                                             = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::P2P] = mock_connector;
    coordinator_->allocator_                                        = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    // leave cacheKeys empty
    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncWriteByLayer(testing::_, testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncWriteByLayer(/*layer_id=*/0, rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenIncrKVCacheRefReturnsNull) {
    auto mock_connector                                             = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::P2P] = mock_connector;
    coordinator_->allocator_                                        = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, enableMemoryCache()).WillByDefault(testing::Return(true));
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(nullptr));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(0);
    EXPECT_CALL(*mock_connector, asyncWriteByLayer(testing::_, testing::_, testing::_)).Times(0);

    auto ctx = coordinator_->asyncWriteByLayer(/*layer_id=*/0, rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenP2PConnectorReturnsNullContext) {
    auto mock_p2p                                                   = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::P2P] = mock_p2p;
    coordinator_->allocator_                                        = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource = std::make_shared<KVCacheResource>();
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(selected_resource));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);
    EXPECT_CALL(*mock_p2p, asyncWriteByLayer(0, testing::Eq(selected_resource), testing::_))
        .WillOnce(testing::Return(nullptr));

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    auto ctx = coordinator_->asyncWriteByLayer(/*layer_id=*/0, rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenP2PConnectorNull) {
    coordinator_->connectors_[KVCacheConnector::ConnectorType::P2P] = nullptr;
    coordinator_->allocator_                                        = allocator_;

    KVCacheResource resource;
    resource.initGroups(1, /*layer_num=*/1);
    resource.cacheKeys() = CacheKeysType{1, 2, 3};

    auto selected_resource = std::make_shared<KVCacheResource>();
    EXPECT_CALL(*allocator_, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(selected_resource));
    EXPECT_CALL(*allocator_, decrKVCacheRef(testing::_)).Times(1);

    auto rw_ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*rw_ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(resource));

    auto ctx = coordinator_->asyncWriteByLayer(/*layer_id=*/0, rw_ctx, /*meta=*/nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnNull_WhenNoP2PContexts) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    // Register a Memory connector only; asyncWriteByLayer only uses P2P.
    coordinator->connectors_[KVCacheConnector::ConnectorType::Memory] =
        std::make_shared<testing::NiceMock<MockKVCacheConnector>>();

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = std::make_shared<KVCacheResource>();

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(resource));
    EXPECT_CALL(*allocator, decrKVCacheRef(testing::_)).Times(1);

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));

    EXPECT_EQ(coordinator->asyncWriteByLayer(0, ctx, nullptr), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncWriteByLayer_ReturnContextAndEnqueue_WhenHasP2PContext) {
    CacheConfig cache_config;
    cache_config.layer_num        = 1;
    cache_config.block_num        = 1;
    cache_config.block_size_bytes = 1;

    auto allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_config, nullptr);
    auto coordinator = std::make_shared<KVCacheConnectorCoordinator>(
        cache_config, KVCacheConfig{}, RuntimeConfig{}, allocator, nullptr);

    auto p2p_connector = std::make_shared<testing::NiceMock<MockKVCacheConnector>>();
    coordinator->connectors_[KVCacheConnector::ConnectorType::P2P] = p2p_connector;

    auto req_resource = KVCacheResource{};
    req_resource.cacheKeys().assign({1, 2, 3});
    auto resource = std::make_shared<KVCacheResource>();

    EXPECT_CALL(*allocator, incrKVCacheRef(testing::_, testing::_)).WillOnce(testing::Return(resource));

    auto write_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*p2p_connector, asyncWriteByLayer(7, testing::Eq(resource), testing::_))
        .WillOnce(testing::Return(write_ctx));

    auto ctx = std::make_shared<testing::NiceMock<MockKVCacheConnectorReadWriteContext>>();
    ON_CALL(*ctx, kvCacheResource()).WillByDefault(testing::ReturnRef(req_resource));

    auto async_ctx = coordinator->asyncWriteByLayer(7, ctx, nullptr);
    ASSERT_NE(async_ctx, nullptr);
    EXPECT_EQ(coordinator->fused_async_write_context_list_.size(), 1);

    EXPECT_CALL(*allocator, decrKVCacheRef(testing::_)).Times(1);
    coordinator->fused_async_write_context_list_.clear();
    async_ctx.reset();
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessReadContexts_EraseReadContext_WhenDone) {
    // Make a FusedAsyncReadContext that reports done() == true
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);
    // fused_match_context_ == nullptr => FusedAsyncReadContext::done() returns true
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(nullptr, resource);
    coordinator_->fused_async_read_context_list_.push_back(fused_read_ctx);

    coordinator_->processReadContexts();
    EXPECT_TRUE(coordinator_->fused_async_read_context_list_.empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessReadContexts_EraseReadContext_WhenMatchDoneButNotSuccess) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);
    auto done_fail_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*done_fail_ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*done_fail_ctx, success()).WillByDefault(testing::Return(false));
    auto match_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{done_fail_ctx});
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(match_ctx, resource);
    coordinator_->fused_async_read_context_list_.push_back(fused_read_ctx);

    coordinator_->processReadContexts();

    EXPECT_TRUE(coordinator_->fused_async_read_context_list_.empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessReadContexts_KeepReadContext_WhenMatchNotDone) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);

    auto not_done_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*not_done_ctx, done()).WillByDefault(testing::Return(false));
    // success() should not be consulted when done() == false, but keep it defined.
    ON_CALL(*not_done_ctx, success()).WillByDefault(testing::Return(true));

    auto match_ctx      = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{not_done_ctx});
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(match_ctx, resource);
    coordinator_->fused_async_read_context_list_.push_back(fused_read_ctx);

    coordinator_->processReadContexts();

    EXPECT_EQ(coordinator_->fused_async_read_context_list_.size(), 1);
    EXPECT_EQ(fused_read_ctx->fusedReadContext(), nullptr);
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessReadContexts_MatchSuccess_StartsAsyncReadAndSetsFusedReadContext) {
    auto fused_read_ctx = makeFusedReadContextAndExpectAsyncRead();
    coordinator_->fused_async_read_context_list_.push_back(fused_read_ctx);

    coordinator_->processReadContexts();

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 2);
    EXPECT_EQ(coordinator_->fused_async_read_context_list_.size(), 1);

    // Second round should NOT trigger another asyncReadAfterMatch, since fusedReadContext() is already set.
    coordinator_->processReadContexts();
    EXPECT_EQ(coordinator_->fused_async_read_context_list_.size(), 1);
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessReadContexts_KeepReadContext_WhenReadContextNotDone) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);

    // Make match done + success
    auto match_done_ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_done_ok, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_done_ok, success()).WillByDefault(testing::Return(true));
    auto match_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_done_ok});

    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(match_ctx, resource);

    // Provide a read context that is NOT done.
    auto read_not_done = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_not_done, done()).WillByDefault(testing::Return(false));
    ON_CALL(*read_not_done, success()).WillByDefault(testing::Return(true));
    fused_read_ctx->setFusedReadContext(
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_not_done}));

    coordinator_->fused_async_read_context_list_.push_back(fused_read_ctx);

    coordinator_->processReadContexts();

    EXPECT_EQ(coordinator_->fused_async_read_context_list_.size(), 1);
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessReadContexts_EraseReadContext_WhenReadContextDone) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);

    // Make match done + success
    auto match_done_ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_done_ok, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_done_ok, success()).WillByDefault(testing::Return(true));
    auto match_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_done_ok});

    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(match_ctx, resource);

    // Provide a read context that IS done.
    auto read_done = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_done, done()).WillByDefault(testing::Return(true));
    ON_CALL(*read_done, success()).WillByDefault(testing::Return(true));
    fused_read_ctx->setFusedReadContext(
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_done}));

    coordinator_->fused_async_read_context_list_.push_back(fused_read_ctx);

    coordinator_->processReadContexts();

    EXPECT_TRUE(coordinator_->fused_async_read_context_list_.empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessWriteContexts_EraseWriteContext_WhenDone) {
    auto done_ok_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*done_ok_ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*done_ok_ctx, success()).WillByDefault(testing::Return(true));
    auto write_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{done_ok_ctx});
    coordinator_->fused_async_write_context_list_.push_back(write_ctx);

    coordinator_->processWriteContexts();

    EXPECT_TRUE(coordinator_->fused_async_write_context_list_.empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, ProcessWriteContexts_KeepWriteContext_WhenNotDone) {
    auto not_done_ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*not_done_ok, done()).WillByDefault(testing::Return(false));
    ON_CALL(*not_done_ok, success()).WillByDefault(testing::Return(true));
    auto write_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{not_done_ok});
    coordinator_->fused_async_write_context_list_.push_back(write_ctx);

    coordinator_->processWriteContexts();

    EXPECT_EQ(coordinator_->fused_async_write_context_list_.size(), 1);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_SkipsNonMatchAndUpdatesReuseAndSetsFusedReadContext) {
    auto fused_read_ctx = makeFusedReadContextAndExpectAsyncRead();
    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 2);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_SetsEmptyFusedReadContext_WhenNoAsyncReadTriggered) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(5);

    // One non-match context should be skipped by dynamic_pointer_cast.
    auto non_match_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();

    // One match context with matchedBlockCount <= reuse should be skipped.
    auto match_ctx = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*match_ctx, matchedBlockCount()).WillByDefault(testing::Return(5));
    ON_CALL(*match_ctx, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));

    auto fused_match_ctx =
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{non_match_ctx, match_ctx});
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource);

    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_TRUE(fused_read_ctx->fusedReadContext()->contexts().empty());
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_DoesNotUpdateReuse_WhenAsyncReadReturnsNull) {
    auto mock_connector                                                = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mock_connector;

    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(1);

    auto match_ctx1 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    auto match_ctx2 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*match_ctx1, matchedBlockCount()).WillByDefault(testing::Return(3));
    ON_CALL(*match_ctx1, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));
    ON_CALL(*match_ctx2, matchedBlockCount()).WillByDefault(testing::Return(5));
    ON_CALL(*match_ctx2, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));

    auto fused_match_ctx =
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_ctx1, match_ctx2});
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource);

    auto read_ctx2 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_ctx2, done()).WillByDefault(testing::Return(false));
    ON_CALL(*read_ctx2, success()).WillByDefault(testing::Return(true));

    testing::InSequence s;
    // First match triggers asyncRead but returns nullptr -> reuse_num must NOT update.
    EXPECT_CALL(*mock_connector,
                asyncRead(testing::Eq(resource),
                          testing::Truly([](const std::shared_ptr<KVCacheConnector::Meta>& meta) {
                              const auto [start, size] = meta->blockRange();
                              return start == 1 && size == 2;  // reuse=1, matched=3
                          }),
                          testing::_))
        .WillOnce(testing::Return(std::shared_ptr<AsyncContext>{nullptr}));
    // Second meta should still use reuse=1 (not 3): start=1, size=4 (matched=5).
    EXPECT_CALL(*mock_connector,
                asyncRead(testing::Eq(resource),
                          testing::Truly([](const std::shared_ptr<KVCacheConnector::Meta>& meta) {
                              const auto [start, size] = meta->blockRange();
                              return start == 1 && size == 4;
                          }),
                          testing::_))
        .WillOnce(testing::Return(read_ctx2));

    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 1);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_UsesConnectorByType) {
    auto mem_connector                                                 = std::make_shared<MockKVCacheConnector>();
    auto remote_connector                                              = std::make_shared<MockKVCacheConnector>();
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Memory] = mem_connector;
    coordinator_->connectors_[KVCacheConnector::ConnectorType::Remote] = remote_connector;

    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);

    auto mem_match = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    auto rem_match = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*mem_match, matchedBlockCount()).WillByDefault(testing::Return(2));
    ON_CALL(*mem_match, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Memory));
    ON_CALL(*rem_match, matchedBlockCount()).WillByDefault(testing::Return(4));
    ON_CALL(*rem_match, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Remote));

    auto fused_match_ctx =
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{mem_match, rem_match});
    auto fused_read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource);

    auto mem_read_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto rem_read_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*mem_read_ctx, done()).WillByDefault(testing::Return(false));
    ON_CALL(*mem_read_ctx, success()).WillByDefault(testing::Return(true));
    ON_CALL(*rem_read_ctx, done()).WillByDefault(testing::Return(false));
    ON_CALL(*rem_read_ctx, success()).WillByDefault(testing::Return(true));

    testing::InSequence s;
    EXPECT_CALL(*mem_connector,
                asyncRead(testing::Eq(resource),
                          testing::Truly([](const std::shared_ptr<KVCacheConnector::Meta>& meta) {
                              const auto [start, size] = meta->blockRange();
                              return start == 0 && size == 2;
                          }),
                          testing::_))
        .WillOnce(testing::Return(mem_read_ctx));
    EXPECT_CALL(*remote_connector,
                asyncRead(testing::Eq(resource),
                          testing::Truly([](const std::shared_ptr<KVCacheConnector::Meta>& meta) {
                              const auto [start, size] = meta->blockRange();
                              return start == 2 && size == 2;
                          }),
                          testing::_))
        .WillOnce(testing::Return(rem_read_ctx));

    coordinator_->asyncReadAfterMatch(fused_read_ctx);

    ASSERT_NE(fused_read_ctx->fusedReadContext(), nullptr);
    EXPECT_EQ(fused_read_ctx->fusedReadContext()->contexts().size(), 2);
}

TEST_F(KVCacheConnectorCoordinatorTest, AsyncReadAfterMatch_Throws_WhenConnectorMissing) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->setReuseBlocksNum(0);

    auto missing = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
    ON_CALL(*missing, matchedBlockCount()).WillByDefault(testing::Return(1));
    ON_CALL(*missing, connectorType()).WillByDefault(testing::Return(KVCacheConnector::ConnectorType::Remote));

    auto fused_match_ctx = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{missing});
    auto fused_read_ctx  = std::make_shared<FusedAsyncReadContext>(fused_match_ctx, resource);

    EXPECT_THROW(coordinator_->asyncReadAfterMatch(fused_read_ctx), std::out_of_range);
}

TEST_F(KVCacheConnectorCoordinatorTest, BroadcastTp_ReturnFalse_WhenRequestInvalid) {
    BroadcastTpRequestPB  request;
    BroadcastTpResponsePB response;
    // request.has_mem_request() is false
    EXPECT_FALSE(coordinator_->broadcastTp(request, response));
}

TEST_F(KVCacheConnectorCoordinatorTest, BroadcastTp_ReturnFalse_WhenMemoryConnectorNull) {
    BroadcastTpRequestPB request;
    request.mutable_mem_request();
    BroadcastTpResponsePB response;

    coordinator_->memory_connector_.reset();
    EXPECT_FALSE(coordinator_->broadcastTp(request, response));
    EXPECT_FALSE(response.mem_response().success());
}

TEST_F(KVCacheConnectorCoordinatorTest, BroadcastTp_ReturnTrue_WhenSuccess) {
    BroadcastTpRequestPB request;
    request.mutable_mem_request();
    BroadcastTpResponsePB response;

    auto mock_mem_connector = std::make_shared<MockKVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, device_, std::vector<std::string>{}, nullptr);
    coordinator_->memory_connector_ = mock_mem_connector;

    EXPECT_CALL(*mock_mem_connector, copyCache(testing::_, testing::_)).WillOnce(testing::Return(true));
    EXPECT_TRUE(coordinator_->broadcastTp(request, response));
}

TEST_F(KVCacheConnectorCoordinatorTest, ClearMemoryCache_ReturnVoid_WhenMemoryConnectorNull) {
    coordinator_->memory_connector_.reset();
    coordinator_->clearMemoryCache();  // Should not crash
}

TEST_F(KVCacheConnectorCoordinatorTest, ClearMemoryCache_ReturnVoid_WhenSuccess) {
    auto mock_mem_connector = std::make_shared<MockKVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, device_, std::vector<std::string>{}, nullptr);
    coordinator_->memory_connector_ = mock_mem_connector;

    EXPECT_CALL(*mock_mem_connector, clearCache()).Times(1);
    coordinator_->clearMemoryCache();
}

}  // namespace test
}  // namespace rtp_llm
