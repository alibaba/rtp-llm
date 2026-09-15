
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class StreamCacheResourceTest: public DeviceTestBase {
protected:
    StreamCacheResourceTest(): perf_scope("PERF_TEST", "1") {}

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(/*layer_num=*/3,
                                              /*block_num=*/9,
                                              /*tokens_per_block=*/2,
                                              rtp_llm::DataType::TYPE_INT8);
    }

    void prepareResource(bool reuse_cache = false, RoleType role_type = RoleType::PDFUSION) {
        prepareResourceWithInputTokens(/*input_tokens=*/{1, 2, 3, 4, 5, 6}, reuse_cache, role_type);
    }

    void prepareHybridResource(bool reuse_cache = false, RoleType role_type = RoleType::PDFUSION) {
        prepareHybridResourceWithInputTokens(/*input_tokens=*/{1, 2, 3, 4, 5, 6}, reuse_cache, role_type);
    }

    void prepareResourceWithInputTokens(const std::vector<int>& input_tokens,
                                        bool                    reuse_cache = false,
                                        RoleType                role_type   = RoleType::PDFUSION) {
        prepareResourceWithCacheConfig(init_config(), input_tokens, reuse_cache, role_type);
    }

    void prepareHybridResourceWithInputTokens(const std::vector<int>& input_tokens,
                                              bool                    reuse_cache = false,
                                              RoleType                role_type   = RoleType::PDFUSION) {
        prepareResourceWithCacheConfig(test::makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                                            /*block_num=*/9,
                                                                            /*tokens_per_block=*/2,
                                                                            rtp_llm::DataType::TYPE_FP16,
                                                                            /*group_layer_num=*/2),
                                       input_tokens,
                                       reuse_cache,
                                       role_type);
    }

    void prepareResourceWithCacheConfig(const CacheConfig&      cache_config,
                                        const std::vector<int>& input_tokens,
                                        bool                    reuse_cache,
                                        RoleType                role_type) {
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        ASSERT_TRUE(cache_manager_->init());
        ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;
        resource_context.role_type     = role_type;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_tokens.begin(), input_tokens.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig model_config;
        model_config.attn_config.tokens_per_block = 2;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        stream_                  = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
        stream_->generate_status_->status = StreamState::RUNNING;
    }

    void checkBlockFunc(BatchKVCacheResource& batch_resource, int outter_size, int inner_size) {
        ASSERT_EQ(batch_resource.batchSize(), outter_size);
        for (int i = 0; i < outter_size; ++i) {
            ASSERT_EQ(batch_resource.blocks(i).size(), inner_size);
        }
    };

#define CHECK_BLOCK(block_vec, outter_size, inner_size)                                                                \
    do {                                                                                                               \
        SCOPED_TRACE("checkBlockFunc");                                                                                \
        checkBlockFunc(block_vec, outter_size, inner_size);                                                            \
    } while (0)

protected:
    autil::EnvGuard                 perf_scope;
    GenerateStreamPtr               stream_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

TEST_F(StreamCacheResourceTest, testAllocateResource) {
    prepareResource();

    auto& resource = stream_->streamCacheResource();

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
    ASSERT_EQ(resource.curBlocksNum(), 3);
    auto& blocks = resource.kvCacheMutable();
    CHECK_BLOCK(blocks, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 3);

    CHECK_BLOCK(blocks, 2, 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);

    CHECK_BLOCK(blocks, 2, 0);
}

// TEST_F(StreamCacheResourceTest, testFallbackWithFastGen) {
//     prepareResource();
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
//     auto& resource            = stream_->streamCacheResource();
//     stream_->enable_fast_gen_ = true;

//     // first chunk: 分块场景下 current_chunk_len 会被设置为 >0
//     int token_capacity = 4;
//     ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 6);
//     ASSERT_GT(stream_->currentChunkLen(), 0);

//     int old_max_blocks = resource.maxBlockSize();
//     int released       = resource.tryReleaseKVBlock(old_max_blocks);
//     stream_->setPaused();

//     ASSERT_EQ(released, old_max_blocks);
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
//     // fast_gen 模式下，fallback 之后 chunk 长度会被重置为 0
//     ASSERT_EQ(stream_->currentChunkLen(), 0);
// }

// TEST_F(StreamCacheResourceTest, testReleaseSequenceKVCache) {
//     prepareResource();
//     auto& resource = stream_->streamCacheResource();

//     int token_capacity = 1000;
//     ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->setSeqLength(7);
//     stream_->setIsContextStream(false);
//     ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 3);
//     ASSERT_EQ(resource.maxBlockSize(), 4);

//     auto status = resource.releaseSequenceKVCache(7, 7);
//     ASSERT_TRUE(status.ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
// }

// TEST_F(StreamCacheResourceTest, testQueryLevelReuseCacheControl) {
//     // Test query-level reuse_cache control when engine-level is enabled
//     prepareResource(true);  // Enable engine-level reuse_cache
//     auto& resource = stream_->streamCacheResource();

//     // Test with query-level reuse_cache = true
//     stream_->generate_input_->generate_config->reuse_cache = true;
//     int token_capacity                                     = 1000;
//     ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     // Test with query-level reuse_cache = false
//     stream_->releaseResource();
//     // Re-initialize batch resource after release
//     resource.init(stream_->currentBatchSize());
//     size_t baseline_free_blocks                            = cache_manager_->freeBlocksNum();
//     stream_->generate_input_->generate_config->reuse_cache = false;
//     ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(),
//               baseline_free_blocks >= 3 ? baseline_free_blocks - 3 : baseline_free_blocks);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->releaseResource();
// }

// TEST_F(StreamCacheResourceTest, testQueryLevelReuseCacheMasterSwitch) {
//     // Test that query-level reuse_cache is ignored when engine-level is disabled
//     prepareResource(false);  // Disable engine-level reuse_cache
//     auto& resource = stream_->streamCacheResource();

//     // Test with query-level reuse_cache = true, but should be ignored
//     stream_->generate_input_->generate_config->reuse_cache = true;
//     int token_capacity                                     = 1000;
//     ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     // Test with query-level reuse_cache = false, should also be ignored
//     stream_->releaseResource();
//     // Re-initialize batch resource after release
//     resource.init(stream_->currentBatchSize());
//     stream_->generate_input_->generate_config->reuse_cache = false;
//     ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->releaseResource();
// }

TEST_F(StreamCacheResourceTest, testStreamCacheResourceReuseCacheMethod) {
    // engine=true, query=true -> true
    prepareResource(true);
    auto& resource                                         = stream_->streamCacheResource();
    stream_->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(resource.reuseCache());

    // engine=true, query=false -> false
    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(resource.reuseCache());

    // engine=false, query=true -> false
    resource.resource_context_.reuse_cache                 = false;
    stream_->generate_input_->generate_config->reuse_cache = true;
    ASSERT_FALSE(resource.reuseCache());

    // engine=false, query=false -> false
    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(resource.reuseCache());
}

TEST_F(StreamCacheResourceTest, testLoadCacheSync_PreservesRequestId_AndUpdatesReuseLen) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    stream_->generate_input_->request_id = 1234567890123LL;

    // Enable query-level reuse_cache and memory_cache so meta(enableMemoryCache) should be true.
    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    // Build a FusedAsyncReadContext that is immediately done/success and has reuse blocks set.
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto fused_match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto kv_resource = std::make_shared<KVCacheResource>();
    kv_resource->setDeviceReuseBlockNum(1);
    kv_resource->setMemoryReuseBlockNum(1);

    std::shared_ptr<Meta> meta;
    auto                  load_ctx = std::make_shared<FusedAsyncReadContext>(fused_match, kv_resource, meta);
    // Important: FusedAsyncReadContext::waitDone() waits for read context to be set (it can be nullptr).
    load_ctx->setFusedReadContext(nullptr);

    std::shared_ptr<KVCacheConnectorReadWriteContext> captured_ctx;
    EXPECT_CALL(*mock_coord, asyncRead(testing::_))
        .WillOnce(testing::Invoke([&](const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
            captured_ctx = connector_context;
            return std::static_pointer_cast<AsyncContext>(load_ctx);
        }));

    ASSERT_TRUE(resource.initKVBlock(/*reserve_step=*/0).ok());
    resource.loadCacheSync();
    ASSERT_NE(captured_ctx, nullptr);
    ASSERT_NE(captured_ctx->meta(), nullptr);
    EXPECT_TRUE(captured_ctx->meta()->enableMemoryCache());
    EXPECT_EQ(captured_ctx->meta()->request_id(), stream_->streamId());

    // seq_size_per_block = 2 in init_config()
    const int expected_total_reuse_len  = (1 + 1) * resource.seqSizePerBlock();
    const int expected_memory_reuse_len = 1 * resource.seqSizePerBlock();
    EXPECT_EQ(stream_->initialReuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->reuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->localReuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->memoryReuseLength(), expected_memory_reuse_len);
}

TEST_F(StreamCacheResourceTest, testDecodeInitKVBlock_DisablesDeviceCacheOnlyForFirstMalloc) {
    prepareHybridResource(/*reuse_cache=*/true, RoleType::DECODE);
    cache_manager_->config_.disable_decode_first_malloc_device_reuse = true;
    auto& resource = stream_->streamCacheResource();

    // Enable query-level reuse/device cache, but decode initKVBlock should still force device cache off.
    stream_->generate_input_->generate_config->reuse_cache         = true;
    stream_->generate_input_->generate_config->enable_device_cache = true;
    resource.resource_context_.enable_device_cache                 = true;

    // initKVBlock() -> loadCacheSync() -> asyncRead.
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    resource.resource_context_.enable_memory_cache                 = true;

    auto allocator             = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;
    EXPECT_CALL(*mock_coord, asyncRead(testing::_)).WillOnce(testing::Return(nullptr));

    testing::InSequence seq;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            EXPECT_FALSE(info.enable_device_cache);
            return {true, 0};
        }));

    EXPECT_CALL(*allocator, incrMalloc(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // initKVBlock should force-disable device cache on the first malloc for decode role.
            EXPECT_FALSE(info.enable_device_cache);
            // Simulate a successful allocation so subsequent calls go through incrMalloc path.
            for (int b = 0; b < info.batch_kv_cache_resource->batchSize(); ++b) {
                auto& block_ids = info.batch_kv_cache_resource->mutableBlockIds(b, /*group_id=*/0);
                block_ids.assign(BlockIndicesType{/*block=*/1});
            }
            return {true, 0};
        }))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // incrKVBlock should respect runtime config: reuseCache() && enableDeviceCache().
            EXPECT_TRUE(info.enable_device_cache);
            return {true, 0};
        }));

    ASSERT_TRUE(resource.initKVBlock(/*reserve_step=*/0).ok());
    resource.asyncLoadCache();
    resource.loadCacheDone();
    ASSERT_TRUE(resource.incrKVBlock(/*reserve_step=*/0).ok());
}

TEST_F(StreamCacheResourceTest, testTryReleaseKVBlock_TriggersStoreCacheAsync_WhenFinishedAndReuseCache) {
    // Use incrKVBlock() to avoid loadCacheSync() noise; we only want to validate storeCacheAsync path.
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    stream_->generate_input_->request_id = 1234567890123LL;

    stream_->generate_input_->generate_config->reuse_cache = true;

    // Enable memory cache gate just to validate meta(enableMemoryCache) is true.
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    // Device cache disabled: new behavior still stores to connector but skips insertIntoCache.
    resource.resource_context_.enable_device_cache                 = false;
    stream_->generate_input_->generate_config->enable_device_cache = false;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);

    cache_manager_->coordinator_ = mock_coord;

    std::shared_ptr<KVCacheConnectorReadWriteContext> captured_ctx;
    auto store_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_coord, asyncWrite(testing::_))
        .WillOnce(testing::Invoke([&](const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
            captured_ctx = connector_context;
            return store_ctx;
        }));

    ASSERT_TRUE(resource.incrKVBlock(/*reserve_step=*/0).ok());
    ASSERT_GT(resource.curBlocksNum(), 0);

    stream_->generate_status_->status = StreamState::FINISHED;
    stream_->fillSubGenerateStatus(StreamState::FINISHED);
    const int blocks = resource.curBlocksNum();
    ASSERT_EQ(resource.tryReleaseKVBlock(blocks), blocks);

    ASSERT_NE(captured_ctx, nullptr);
    ASSERT_NE(captured_ctx->meta(), nullptr);
    EXPECT_TRUE(captured_ctx->meta()->enableMemoryCache());
    EXPECT_EQ(captured_ctx->meta()->generateStream(), nullptr);
    EXPECT_EQ(captured_ctx->meta()->request_id(), stream_->streamId());
}

TEST_F(StreamCacheResourceTest, testTryReleaseKVBlock_DoesNotStoreCacheAsync_WhenNotFinished) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);

    cache_manager_->coordinator_ = mock_coord;

    EXPECT_CALL(*mock_coord, asyncWrite(testing::_)).Times(0);

    ASSERT_TRUE(resource.incrKVBlock(/*reserve_step=*/0).ok());
    const int blocks = resource.curBlocksNum();
    ASSERT_GT(blocks, 0);

    // Stream is still running -> should not store to connector.
    ASSERT_EQ(resource.tryReleaseKVBlock(blocks), blocks);
}

TEST_F(StreamCacheResourceTest, testTryReleaseKVBlock_TieredMemoryCache_EvictsDeviceBlocksWithSeparateMeta) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    resource.resource_context_.enable_remote_cache                 = true;
    stream_->generate_input_->generate_config->enable_remote_cache = true;
    resource.resource_context_.enable_tiered_memory_cache          = true;
    resource.resource_context_.device_cache_min_free_blocks        = 8;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);

    cache_manager_->coordinator_ = mock_coord;

    std::vector<std::shared_ptr<KVCacheConnectorReadWriteContext>> captured_ctxs;
    auto store_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_coord, asyncWrite(testing::_))
        .Times(2)
        .WillRepeatedly(
            testing::Invoke([&](const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
                captured_ctxs.push_back(connector_context);
                return store_ctx;
            }));

    ASSERT_TRUE(resource.incrKVBlock(/*reserve_step=*/0).ok());
    ASSERT_GT(resource.curBlocksNum(), 0);

    stream_->generate_status_->status = StreamState::FINISHED;
    stream_->fillSubGenerateStatus(StreamState::FINISHED);
    const int blocks = resource.curBlocksNum();
    ASSERT_EQ(resource.tryReleaseKVBlock(blocks), blocks);

    ASSERT_EQ(captured_ctxs.size(), 2u);
    ASSERT_NE(captured_ctxs[0], nullptr);
    ASSERT_NE(captured_ctxs[1], nullptr);
    EXPECT_FALSE(captured_ctxs[0]->meta()->enableMemoryCache());
    EXPECT_TRUE(captured_ctxs[0]->meta()->enableRemoteCache());
    EXPECT_TRUE(captured_ctxs[1]->meta()->enableMemoryCache());
    EXPECT_FALSE(captured_ctxs[1]->meta()->enableRemoteCache());
    EXPECT_FALSE(captured_ctxs[1]->kvCacheResource().cacheKeys().empty());
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
}

// ============================================================================
// asyncLoadCache() and loadCacheDone() tests
// ============================================================================

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_NoReuseCache_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // Without reuse_cache, asyncLoadCache should return false
    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_ReuseCacheNoConnector_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    // reuse_cache=true but neither memory_cache nor remote_cache enabled
    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = false;
    stream_->generate_input_->generate_config->enable_memory_cache = false;
    resource.resource_context_.enable_remote_cache                 = false;
    stream_->generate_input_->generate_config->enable_remote_cache = false;

    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_WithMemoryCache_SubmitsLoad) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    stream_->generate_input_->request_id = 1234567890123LL;

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    auto mock_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(testing::Return(false));
    EXPECT_CALL(*mock_coord, asyncRead(testing::_))
        .WillOnce(testing::Invoke([&](const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
            EXPECT_EQ(connector_context->meta()->request_id(), stream_->streamId());
            return std::static_pointer_cast<AsyncContext>(mock_ctx);
        }));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.asyncLoadCache());

    // Second call is idempotent - already has load_cache_context_
    ASSERT_TRUE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_CoordinatorReturnsNull_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    // Coordinator returns nullptr (no connector available)
    EXPECT_CALL(*mock_coord, asyncRead(testing::_)).WillOnce(testing::Return(nullptr));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_NoContext_ReturnsTrue) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No load_cache_context_ -> immediately done
    ASSERT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, MemoryReuseFailureFallsBackToDevicePrefixWithoutRetry) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    const auto allocated_blocks = resource.kvCache().blocks(0);
    const auto free_blocks      = cache_manager_->freeBlocksNum();
    resource.kvCacheMutable().cacheResource(0).setDeviceReuseBlockNum(1);

    auto failed = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*failed, done()).WillByDefault(testing::Return(true));
    ON_CALL(*failed, success()).WillByDefault(testing::Return(false));
    ON_CALL(*failed, errorInfo())
        .WillByDefault(testing::Return(ErrorInfo(ErrorCode::KV_CACHE_REUSE_ERROR, "rank 1 memory copy failed")));
    auto fused_match     = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{});
    auto loaded_resource = std::make_shared<KVCacheResource>();
    loaded_resource->setDeviceReuseBlockNum(1);
    // Another connector can succeed beyond the failed memory prefix. Its
    // accounting must not turn that disconnected suffix into a reuse hit.
    loaded_resource->setMemoryReuseBlockNum(1);
    loaded_resource->setRemoteReuseBlockNum(1);
    auto load_context = std::make_shared<FusedAsyncReadContext>(fused_match, loaded_resource, nullptr);
    load_context->setFusedReadContext(
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{failed}));
    resource.load_cache_context_ = load_context;
    resource.load_cache_once_.store(true);
    resource.resource_context_.load_cache_retry_times = 5;
    stream_->setReuseLength(4);
    stream_->setInitialReuseLength(4);
    stream_->setMemoryReuseLength(2);
    stream_->setRemoteReuseLength(2);
    stream_->setMtpTokenIndex(4);

    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(resource.load_cache_retry_count_, 0);
    EXPECT_EQ(resource.load_cache_context_, nullptr);
    EXPECT_TRUE(resource.load_cache_once_.load());
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->initialReuseLength(), 2);
    EXPECT_EQ(stream_->localReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_EQ(stream_->memoryReuseLength(), 0);
    EXPECT_EQ(stream_->remoteReuseLength(), 0);
    EXPECT_EQ(stream_->mtp_token_index_, 2);
    EXPECT_EQ(loaded_resource->reuseBlockNum(), 1);
    EXPECT_EQ(resource.kvCache().blocks(0), allocated_blocks);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_blocks);
    EXPECT_TRUE(resource.incrKVBlock().ok());
    stream_->releaseResource();
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8);
}

TEST_F(StreamCacheResourceTest, Dsv4MemoryReuseFallbackPreservesDevicePrefix) {
    constexpr int block_tokens = 128;
    ModelConfig   model;
    model.num_layers                        = 5;
    model.hidden_size                       = 32;
    model.max_seq_len                       = 2048;
    model.attn_config.tokens_per_block      = block_tokens;
    model.attn_config.head_num              = 4;
    model.attn_config.kv_head_num           = 1;
    model.attn_config.size_per_head         = 8;
    model.attn_config.rope_head_dim         = 4;
    model.attn_config.sliding_window        = 128;
    model.attn_config.indexer_head_dim      = 8;
    model.attn_config.indexer_head_num      = 2;
    model.attn_config.indexer_topk          = 16;
    model.attn_config.o_groups              = 2;
    model.attn_config.o_lora_rank           = 16;
    model.attn_config.layer_compress_ratios = {4, 128, 4, 128, 0};
    KVCacheConfig kv_config;
    kv_config.seq_size_per_block     = block_tokens;
    kv_config.dsv4_fixed_pool_blocks = 16;
    auto config        = HybridPoolConfigCreator::createConfig(model, ParallelismConfig{}, kv_config, false, 0);
    config.block_num   = 16;
    ASSERT_EQ(config.linear_step, 1);
    cache_manager_     = std::make_shared<KVCacheManager>(config, false, nullptr, kv_config);
    ASSERT_TRUE(cache_manager_->init());
    auto allocator = std::dynamic_pointer_cast<HybridPoolKVCacheAllocator>(cache_manager_->allocator_);
    ASSERT_NE(allocator, nullptr);
    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    ASSERT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
    ASSERT_EQ(config.group_region_names[4], KVCacheRegionName::CSA_STATE);
    ASSERT_EQ(config.group_region_names[6], KVCacheRegionName::SWA_KV);

    ResourceContext context;
    context.cache_manager       = cache_manager_;
    context.reuse_cache         = true;
    context.enable_device_cache = true;
    context.role_type           = RoleType::PDFUSION;
    auto make_stream            = [&](int tokens) {
        std::vector<int32_t> input_tokens(tokens);
        for (int i = 0; i < tokens; ++i) {
            input_tokens[i] = i + 1;
        }
        auto input                                   = std::make_shared<GenerateInput>();
        input->input_ids                             = torch::tensor(input_tokens, torch::kInt32);
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->num_return_sequences = 1;
        input->generate_config->reuse_cache          = true;
        auto stream = std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setReserveStep(1);
        return stream;
    };

    // A shorter request publishes a reusable fixed-pool tail at position 2.
    auto  seed          = make_stream(3 * block_tokens + 1);
    auto& seed_resource = seed->streamCacheResource();
    ASSERT_TRUE(seed_resource.initKVBlock(1).ok());
    BlockIndicesType trusted_tails(7, NULL_BLOCK_IDX);
    for (int gid : {3, 4, 6}) {
        trusted_tails[gid] = seed_resource.kvCache().blocks(0, gid).at(2);
        ASSERT_FALSE(isNullBlockIdx(trusted_tails[gid]));
        ASSERT_GT(trusted_tails[gid], 0);
    }
    cache_manager_->insertIntoCache(
        InsertInfo{seed_resource.batch_kv_cache_resource_, seed->completeTokenIdsPtr(), false});
    seed->releaseResource();

    stream_        = make_stream(5 * block_tokens + 1);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock(1).ok());
    ASSERT_TRUE(stream_->isContextStream());
    ASSERT_EQ(resource.kvCache().cacheResource(0).deviceReuseBlockNum(), 3);
    ASSERT_EQ(stream_->reuseLength(), 3 * block_tokens);
    std::vector<BlockIndicesType> allocated_groups;
    for (int gid = 0; gid < 7; ++gid) {
        allocated_groups.push_back(resource.kvCache().blocks(0, gid));
    }
    for (int gid : {3, 4, 6}) {
        SCOPED_TRACE(gid);
        const auto& blocks = allocated_groups[gid];
        ASSERT_EQ(blocks.size(), 6u);
        EXPECT_FALSE(isNullBlockIdx(blocks[0]));
        EXPECT_FALSE(isNullBlockIdx(blocks[1]));
        EXPECT_EQ(blocks[2], trusted_tails[gid]);
        for (size_t pos = 3; pos < blocks.size(); ++pos) {
            ASSERT_FALSE(isNullBlockIdx(blocks[pos]));
            ASSERT_GT(blocks[pos], 0);
            EXPECT_NE(blocks[pos], trusted_tails[gid]);
        }
        const auto& pool = allocator->groupBlockPools()[gid];
        ASSERT_EQ(pool->request_ref_counter_.getRefCounter(trusted_tails[gid]), 1);
        ASSERT_EQ(pool->block_cache_ref_counter_.getRefCounter(trusted_tails[gid]), 1);
        ASSERT_EQ(pool->requestRefBlocksNum(), 6u);
    }

    auto failed = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*failed, done()).WillByDefault(testing::Return(true));
    ON_CALL(*failed, success()).WillByDefault(testing::Return(false));
    ON_CALL(*failed, errorInfo())
        .WillByDefault(testing::Return(ErrorInfo(ErrorCode::KV_CACHE_REUSE_ERROR, "rank 1 memory copy failed")));
    auto loaded_resource = std::make_shared<KVCacheResource>(resource.kvCache().cacheResource(0));
    loaded_resource->setMemoryReuseBlockNum(1);
    loaded_resource->setRemoteReuseBlockNum(1);
    auto fused_match  = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{});
    auto load_context = std::make_shared<FusedAsyncReadContext>(fused_match, loaded_resource, nullptr);
    load_context->setFusedReadContext(
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{failed}));
    resource.load_cache_context_ = load_context;
    resource.load_cache_once_.store(true);
    resource.resource_context_.load_cache_retry_times = 5;
    stream_->setReuseLength(5 * block_tokens);
    stream_->setInitialReuseLength(5 * block_tokens);
    stream_->setMemoryReuseLength(block_tokens);
    stream_->setRemoteReuseLength(block_tokens);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->reportEvent(StreamEvents::CanRun);

    ASSERT_EQ(stream_->moveToNext(), StreamState::WAITING);
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(stream_->reuseLength(), 3 * block_tokens);
    EXPECT_EQ(loaded_resource->reuseBlockNum(), 3);
    EXPECT_EQ(stream_->memoryReuseLength(), 0);
    EXPECT_EQ(stream_->remoteReuseLength(), 0);
    EXPECT_EQ(resource.load_cache_retry_count_, 0);
    EXPECT_EQ(resource.load_cache_context_, nullptr);
    // Exercise PDFUSION's real WAITING -> incrKVBlock -> RUNNING path.
    ASSERT_EQ(stream_->moveToNext(), StreamState::RUNNING);
    for (int gid : {0, 1, 2, 3, 4, 6}) {
        EXPECT_EQ(resource.kvCache().blocks(0, gid), allocated_groups[gid]) << "gid=" << gid;
    }
    // HCA_STATE does not reuse cached state and keeps the ordinary active-tail policy.
    EXPECT_EQ(resource.kvCache().blocks(0, 5).back(), allocated_groups[5].back());
    for (int gid : {3, 4, 6}) {
        const auto& pool = allocator->groupBlockPools()[gid];
        EXPECT_EQ(pool->request_ref_counter_.getRefCounter(trusted_tails[gid]), 1) << "gid=" << gid;
        EXPECT_EQ(pool->requestRefBlocksNum(), 6u) << "gid=" << gid;
    }

    // With step=1, reusable blocks remain owned until the request releases them.
    stream_->setIsContextStream(false);
    stream_->setSeqLength(5 * block_tokens + 2);
    ASSERT_TRUE(resource.incrKVBlock(1).ok());
    for (int gid : {3, 4, 6}) {
        SCOPED_TRACE(gid);
        EXPECT_EQ(resource.kvCache().blocks(0, gid), allocated_groups[gid]);
        const auto& pool = allocator->groupBlockPools()[gid];
        EXPECT_EQ(pool->request_ref_counter_.getRefCounter(trusted_tails[gid]), 1);
        EXPECT_EQ(pool->requestRefBlocksNum(), 6u);
        // Cache ownership is independent of the request's completed use.
        EXPECT_EQ(pool->block_cache_ref_counter_.getRefCounter(trusted_tails[gid]), 1);
    }
    stream_->releaseResource();
    for (const auto& pool : allocator->groupBlockPools()) {
        EXPECT_EQ(pool->requestRefBlocksNum(), 0u);
        EXPECT_EQ(pool->connectorRefBlocksNum(), 0u);
    }
}

TEST_F(StreamCacheResourceTest, MemoryReuseFailureWithoutDevicePrefixRestoresZeroReuse) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    auto failed = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*failed, done()).WillByDefault(testing::Return(true));
    ON_CALL(*failed, success()).WillByDefault(testing::Return(false));
    ON_CALL(*failed, errorInfo())
        .WillByDefault(testing::Return(ErrorInfo(ErrorCode::KV_CACHE_REUSE_ERROR, "memory copy failed")));
    stream_->setReuseLength(4);
    stream_->setInitialReuseLength(4);
    stream_->setLocalReuseLength(4);
    stream_->setMemoryReuseLength(4);
    stream_->setRemoteReuseLength(2);
    stream_->setMtpTokenIndex(4);
    // Exercise the common completion path used by synchronous cache loading.
    EXPECT_TRUE(resource.finishLoadCache(failed, false));
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(stream_->initialReuseLength(), 0);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->deviceReuseLength(), 0);
    EXPECT_EQ(stream_->memoryReuseLength(), 0);
    EXPECT_EQ(stream_->remoteReuseLength(), 0);
    EXPECT_EQ(stream_->mtp_token_index_, 0);
    EXPECT_EQ(resource.load_cache_retry_count_, 0);
    stream_->releaseResource();
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8);
}

TEST_F(StreamCacheResourceTest, OrdinaryCopyFailureDoesNotResetReuseLengths) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    auto failed = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*failed, success()).WillByDefault(testing::Return(false));
    ON_CALL(*failed, errorInfo()).WillByDefault(testing::Return(ErrorInfo::OkStatus()));
    stream_->setReuseLength(4);
    stream_->setInitialReuseLength(4);
    stream_->setLocalReuseLength(4);
    stream_->setMemoryReuseLength(4);
    stream_->setMtpTokenIndex(4);
    EXPECT_TRUE(resource.finishLoadCache(failed, false));
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->initialReuseLength(), 4);
    EXPECT_EQ(stream_->localReuseLength(), 4);
    EXPECT_EQ(stream_->memoryReuseLength(), 4);
    EXPECT_EQ(stream_->mtp_token_index_, 4);
}

TEST_F(StreamCacheResourceTest, SyncAndAsyncLoadHandleCompletedFailuresConsistently) {
    for (bool asynchronous : {false, true}) {
        for (auto code : {ErrorCode::KV_CACHE_REUSE_ERROR, ErrorCode::EXECUTION_EXCEPTION, ErrorCode::NONE_ERROR}) {
            SCOPED_TRACE(::testing::Message() << asynchronous << ":" << static_cast<int>(code));
            prepareResource(/*reuse_cache=*/true);
            auto& resource = stream_->streamCacheResource();
            ASSERT_TRUE(resource.initKVBlock().ok());
            resource.resource_context_.enable_memory_cache                 = true;
            resource.resource_context_.load_cache_retry_times              = 3;
            stream_->generate_input_->generate_config->enable_memory_cache = true;
            resource.batch_kv_cache_resource_->cacheResource(0).setDeviceReuseBlockNum(1);
            auto coordinator =
                std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                                     cache_manager_->kv_cache_config_,
                                                                                     cache_manager_->runtime_config_,
                                                                                     cache_manager_->allocator_);
            ON_CALL(*coordinator, hasActiveConnectors()).WillByDefault(testing::Return(true));
            cache_manager_->coordinator_ = coordinator;
            auto matched                 = std::make_shared<testing::NiceMock<MockAsyncMatchContext>>();
            ON_CALL(*matched, done()).WillByDefault(testing::Return(true));
            ON_CALL(*matched, success()).WillByDefault(testing::Return(true));
            ON_CALL(*matched, matchedBlockCount())
                .WillByDefault(testing::Return(code == ErrorCode::NONE_ERROR ? 0 : 1));
            auto rejected = std::make_shared<testing::NiceMock<MockAsyncContext>>();
            ON_CALL(*rejected, done()).WillByDefault(testing::Return(true));
            ON_CALL(*rejected, success()).WillByDefault(testing::Return(false));
            ON_CALL(*rejected, errorInfo()).WillByDefault(testing::Return(ErrorInfo(code, "load failed")));
            auto loaded = std::make_shared<KVCacheResource>();
            loaded->setDeviceReuseBlockNum(1);
            loaded->setMemoryReuseBlockNum(2);
            auto context = std::make_shared<FusedAsyncReadContext>(
                std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{matched}),
                loaded,
                nullptr);
            context->setFusedReadContext(
                std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{rejected}));
            EXPECT_CALL(*coordinator, asyncRead(testing::_)).Times(1).WillOnce(testing::Return(context));
            stream_->setReuseLength(4);
            stream_->setInitialReuseLength(4);
            stream_->setMemoryReuseLength(2);
            stream_->setMtpTokenIndex(4);
            if (asynchronous) {
                ASSERT_TRUE(resource.asyncLoadCache());
                EXPECT_TRUE(resource.loadCacheDone());
                EXPECT_TRUE(resource.loadCacheDone());
            } else {
                resource.loadCacheSync();
                resource.loadCacheSync();
            }
            const bool fallback = code == ErrorCode::KV_CACHE_REUSE_ERROR;
            EXPECT_EQ(stream_->reuseLength(), fallback ? 2 : 4);
            EXPECT_EQ(stream_->initialReuseLength(), fallback ? 2 : 4);
            EXPECT_EQ(stream_->memoryReuseLength(), fallback ? 0 : 2);
            EXPECT_EQ(stream_->mtp_token_index_, fallback ? 2 : 4);
            EXPECT_EQ(stream_->hasError(), code == ErrorCode::EXECUTION_EXCEPTION);
            EXPECT_EQ(resource.load_cache_retry_count_, 0);
            EXPECT_EQ(resource.load_cache_context_, nullptr);
            stream_->releaseResource();
        }
    }
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_Pending_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    auto mock_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(testing::Return(false));
    EXPECT_CALL(*mock_coord, asyncRead(testing::_))
        .WillOnce(testing::Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.asyncLoadCache());
    // Still pending
    ASSERT_FALSE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_Done_ReturnsTrue_ClearsContext) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    // Build a complete FusedAsyncReadContext that is immediately done
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto fused_match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto kv_resource = std::make_shared<KVCacheResource>();
    kv_resource->setDeviceReuseBlockNum(1);
    kv_resource->setMemoryReuseBlockNum(1);

    std::shared_ptr<Meta> meta;
    auto                  load_ctx = std::make_shared<FusedAsyncReadContext>(fused_match, kv_resource, meta);
    load_ctx->setFusedReadContext(nullptr);

    EXPECT_CALL(*mock_coord, asyncRead(testing::_))
        .WillOnce(testing::Return(std::static_pointer_cast<AsyncContext>(load_ctx)));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.asyncLoadCache());
    ASSERT_TRUE(resource.loadCacheDone());

    // After loadCacheDone, the context should be cleared
    ASSERT_EQ(resource.load_cache_context_, nullptr);

    // Subsequent call returns true (no context)
    ASSERT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_ThenLoadCacheDone_UpdatesReuseLength) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto fused_match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto kv_resource = std::make_shared<KVCacheResource>();
    kv_resource->setDeviceReuseBlockNum(1);
    kv_resource->setMemoryReuseBlockNum(1);

    std::shared_ptr<Meta> meta;
    auto                  load_ctx = std::make_shared<FusedAsyncReadContext>(fused_match, kv_resource, meta);
    load_ctx->setFusedReadContext(nullptr);

    EXPECT_CALL(*mock_coord, asyncRead(testing::_))
        .WillOnce(testing::Return(std::static_pointer_cast<AsyncContext>(load_ctx)));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.asyncLoadCache());
    ASSERT_TRUE(resource.loadCacheDone());

    // Verify reuse lengths are updated
    // seq_size_per_block = 2 (from init_config)
    const int total_reuse_len  = (1 + 1) * resource.seqSizePerBlock();
    const int memory_reuse_len = 1 * resource.seqSizePerBlock();
    EXPECT_EQ(stream_->initialReuseLength(), total_reuse_len);
    EXPECT_EQ(stream_->reuseLength(), total_reuse_len);
    EXPECT_EQ(stream_->memoryReuseLength(), memory_reuse_len);
}

TEST_F(StreamCacheResourceTest, testP2PSideChannelRestoresZeroFirstTokenAndMtpState) {
    prepareResourceWithInputTokens({1, 2, 3}, /*reuse_cache=*/true);
    stream_->vocab_size_ = 16;
    auto& resource = stream_->streamCacheResource();

    auto kv_resource = std::make_shared<KVCacheResource>();
    // Reuse accounting in this case comes from the P2P side-channel payload.
    // Keep connector-local counters at zero so they do not preempt that payload.

    auto server_call_result                             = std::make_shared<PrefillLoadCaller::Result>();
    server_call_result->side_channel_payload.has_data   = true;
    server_call_result->side_channel_payload.first_token_id = 0;
    server_call_result->side_channel_payload.total_reuse_len = 2;
    server_call_result->side_channel_payload.local_reuse_len = 2;
    server_call_result->side_channel_payload.propose_tokens  = {0, 7};
    TensorPbConvert::torchToPb(&server_call_result->side_channel_payload.propose_probs,
                               torch::tensor({{0.1f, 0.2f, 0.7f}}, torch::kFloat32));
    TensorPbConvert::torchToPb(&server_call_result->side_channel_payload.propose_hidden,
                               torch::tensor({{0.3f, 0.4f}}, torch::kFloat32));

    auto p2p_ctx      = std::make_shared<P2PConnectorAsyncReadContext>(kv_resource,
        std::shared_ptr<P2PBroadcastClient::Result>(),
        server_call_result,
        std::shared_ptr<DecodeSchedulerMetricsCollector>(),
        /*transfer_not_done_hold_ms=*/0);
    auto read_context = std::make_shared<FusedAsyncReadContext>(
        std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{}), kv_resource, nullptr);
    read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(
        std::vector<std::shared_ptr<AsyncContext>>{std::static_pointer_cast<AsyncContext>(p2p_ctx)}));

    p2p_ctx->done_ = true;
    p2p_ctx->success_ = true;
    resource.load_cache_context_ = read_context;
    ASSERT_TRUE(resource.loadCacheDone());
    ASSERT_TRUE(resource.loadCacheDone());

    EXPECT_EQ(stream_->completeTokenIdsVec(), std::vector<int>({1, 2, 3, 0}));
    auto sp_output_buffer = stream_->getSPOutputBuffer();
    ASSERT_TRUE(sp_output_buffer != nullptr);
    EXPECT_EQ(sp_output_buffer->tokens.cpu()[0][0].item<int32_t>(), 0);
    EXPECT_EQ(sp_output_buffer->tokens.cpu()[0][1].item<int32_t>(), 7);
    ASSERT_TRUE(sp_output_buffer->all_probs.defined());
    ASSERT_TRUE(sp_output_buffer->hidden_states.defined());
    EXPECT_TRUE(stream_->getAcceptTokensGpu().defined());
    EXPECT_TRUE(stream_->getAcceptLenGpu().defined());
    EXPECT_TRUE(stream_->getProposeTokensGpu().defined());
    EXPECT_TRUE(stream_->getDraftAllProbsGpu().defined());
    EXPECT_TRUE(stream_->getLastHiddenStatesGpu().defined());
}

TEST_F(StreamCacheResourceTest, testInitKVBlock_SecondCallDoesNotOverwriteReuseLength) {
    // Simulates PD separation: initKVBlock called twice on the same stream.
    // First call sets reuse length via asyncLoadCache+loadCacheDone; second call should NOT overwrite it with 0.
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_);
    ON_CALL(*mock_coord, hasActiveConnectors()).WillByDefault(testing::Return(true));
    cache_manager_->coordinator_ = mock_coord;

    // First call: asyncLoadCache returns reuse blocks (memory=1, device=2)
    auto match_child1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child1, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child1, success()).WillByDefault(testing::Return(true));
    auto fused_match1 = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child1});

    auto kv_resource1 = std::make_shared<KVCacheResource>();
    kv_resource1->setDeviceReuseBlockNum(1);
    kv_resource1->setMemoryReuseBlockNum(1);

    std::shared_ptr<Meta> meta1;
    auto                  load_ctx1 = std::make_shared<FusedAsyncReadContext>(fused_match1, kv_resource1, meta1);
    load_ctx1->setFusedReadContext(nullptr);

    // Second call: load_cache_once_ prevents re-issue (no asyncRead call expected)
    // loadCacheSync runs once inside initKVBlock; second initKVBlock skips async read (load_cache_once_).
    EXPECT_CALL(*mock_coord, asyncRead(testing::_))
        .WillOnce(testing::Return(std::static_pointer_cast<AsyncContext>(load_ctx1)));

    // First initKVBlock + asyncLoadCache + loadCacheDone: sets reuse lengths
    ASSERT_TRUE(resource.initKVBlock(/*reserve_step=*/0).ok());
    ASSERT_GT(resource.curBlocksNum(), 0);
    ASSERT_TRUE(resource.asyncLoadCache());
    ASSERT_TRUE(resource.loadCacheDone());

    const int expected_total_reuse_len  = (1 + 1) * resource.seqSizePerBlock();
    const int expected_memory_reuse_len = 1 * resource.seqSizePerBlock();
    EXPECT_EQ(stream_->reuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->memoryReuseLength(), expected_memory_reuse_len);

    // Second initKVBlock + asyncLoadCache + loadCacheDone: load_cache_once_ prevents re-issue.
    // The once-per-lifecycle guard means the second asyncLoadCache() returns false (skipped),
    // which inherently preserves the reuse lengths set by the first load.
    ASSERT_TRUE(resource.initKVBlock(/*reserve_step=*/0).ok());
    ASSERT_TRUE(resource.asyncLoadCache());
    ASSERT_TRUE(resource.loadCacheDone());

    EXPECT_EQ(stream_->reuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->initialReuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->localReuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->memoryReuseLength(), expected_memory_reuse_len);
    EXPECT_EQ(stream_->deviceReuseLength(), expected_total_reuse_len - expected_memory_reuse_len);
}

TEST_F(StreamCacheResourceTest, testWaitLoadCacheDone_ZeroReuseLen_DoesNotOverwriteExisting) {
    // Directly tests that completion with total_reuse_len == 0 preserves existing values.
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    // Pre-set reuse lengths on the stream (simulating a prior successful loadCacheSync)
    stream_->setReuseLength(4);
    stream_->setInitialReuseLength(4);
    stream_->setLocalReuseLength(3);
    stream_->setMemoryReuseLength(1);
    stream_->setRemoteReuseLength(1);
    stream_->setMtpTokenIndex(4);

    // Build a FusedAsyncReadContext with 0 reuse blocks
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto fused_match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto                  kv_resource = std::make_shared<KVCacheResource>();
    std::shared_ptr<Meta> meta;
    auto                  load_ctx = std::make_shared<FusedAsyncReadContext>(fused_match, kv_resource, meta);
    load_ctx->setFusedReadContext(nullptr);

    // Apply the completed result directly
    EXPECT_TRUE(resource.finishLoadCache(load_ctx, false));

    // All values should be preserved — not overwritten with 0
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->initialReuseLength(), 4);
    EXPECT_EQ(stream_->localReuseLength(), 3);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_EQ(stream_->memoryReuseLength(), 1);
    EXPECT_EQ(stream_->remoteReuseLength(), 1);
    EXPECT_EQ(stream_->getMtpTokenIndex(), 4);
}

}  // namespace rtp_llm
