
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

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
        cache_manager_ =
            std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        ASSERT_TRUE(cache_manager_->init());
        ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;
        resource_context.role_type     = role_type;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        auto                vec               = input_tokens;  // keep alive until stream is constructed
        std::vector<size_t> shape             = {vec.size()};
        generate_input->input_ids =
            std::make_unique<rtp_llm::Buffer>(rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(vec.data()));
        generate_input->generate_config = generate_config;
        ModelConfig model_config;
        model_config.attn_config.tokens_per_block = 2;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        stream_                  = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
        stream_->setRunning();
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

TEST_F(StreamCacheResourceTest, testInitKVBlock_TriggersLoadCacheSync_AndUpdatesReuseLen) {
    // initKVBlock() calls incrKVBlock() then loadCacheSync() internally.
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    // Enable query-level reuse_cache and memory_cache so meta(enableMemoryCache) should be true.
    stream_->generate_input_->generate_config->reuse_cache         = true;
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_,
                                                                             device_);
    cache_manager_->coordinator_ = mock_coord;

    // Build a FusedAsyncReadContext that is immediately done/success and has reuse blocks set.
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto fused_match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto kv_resource = std::make_shared<KVCacheResource>();
    kv_resource->setDeviceReuseBlockNum(2);
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
    ASSERT_NE(captured_ctx, nullptr);
    ASSERT_NE(captured_ctx->meta(), nullptr);
    EXPECT_TRUE(captured_ctx->meta()->enableMemoryCache());

    // seq_size_per_block = 2 in init_config()
    const int expected_total_reuse_len  = (2 + 1) * resource.seqSizePerBlock();
    const int expected_memory_reuse_len = 1 * resource.seqSizePerBlock();
    EXPECT_EQ(stream_->initialReuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->reuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->localReuseLength(), expected_total_reuse_len);
    EXPECT_EQ(stream_->memoryReuseLength(), expected_memory_reuse_len);
}

TEST_F(StreamCacheResourceTest, testDecodeInitKVBlock_DisablesDeviceCacheOnlyForFirstMalloc) {
    prepareHybridResource(/*reuse_cache=*/true, RoleType::DECODE);
    auto& resource = stream_->streamCacheResource();

    // Enable query-level reuse/device cache, but decode initKVBlock should still force device cache off.
    stream_->generate_input_->generate_config->reuse_cache         = true;
    stream_->generate_input_->generate_config->enable_device_cache = true;
    resource.resource_context_.enable_device_cache                 = true;

    // Enable memory cache so initKVBlock will call asyncLoadCache -> asyncRead.
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    resource.resource_context_.enable_memory_cache                 = true;

    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_, device_);
    cache_manager_->allocator_ = allocator;

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_,
                                                                             device_);
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
                auto& blocks = info.batch_kv_cache_resource->mutableBlocks(b, /*group_id=*/0);
                blocks.assign(1, /*value=*/1);
            }
            return {true, 0};
        }))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // incrKVBlock should respect runtime config: reuseCache() && enableDeviceCache().
            EXPECT_TRUE(info.enable_device_cache);
            return {true, 0};
        }));

    ASSERT_TRUE(resource.initKVBlock(/*reserve_step=*/0).ok());
    ASSERT_TRUE(resource.incrKVBlock(/*reserve_step=*/0).ok());
}

TEST_F(StreamCacheResourceTest, testTryReleaseKVBlock_TriggersStoreCacheAsync_WhenFinishedAndReuseCache) {
    // Use incrKVBlock() to avoid loadCacheSync() noise; we only want to validate storeCacheAsync path.
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

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
                                                                             cache_manager_->allocator_,
                                                                             device_);
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

    stream_->setFinishedWithoutLock();
    const int blocks = resource.curBlocksNum();
    ASSERT_EQ(resource.tryReleaseKVBlock(blocks), blocks);

    ASSERT_NE(captured_ctx, nullptr);
    ASSERT_NE(captured_ctx->meta(), nullptr);
    EXPECT_TRUE(captured_ctx->meta()->enableMemoryCache());
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
                                                                             cache_manager_->allocator_,
                                                                             device_);
    cache_manager_->coordinator_ = mock_coord;

    EXPECT_CALL(*mock_coord, asyncWrite(testing::_)).Times(0);

    ASSERT_TRUE(resource.incrKVBlock(/*reserve_step=*/0).ok());
    const int blocks = resource.curBlocksNum();
    ASSERT_GT(blocks, 0);

    // Stream is still running -> should not store to connector.
    ASSERT_EQ(resource.tryReleaseKVBlock(blocks), blocks);
}

}  // namespace rtp_llm
