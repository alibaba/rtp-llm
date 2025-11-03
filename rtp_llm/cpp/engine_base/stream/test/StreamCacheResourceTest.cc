
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

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

    void prepareResource(bool reuse_cache = false) {
        auto cache_config = init_config();
        cache_manager_    = std::make_shared<KVCacheManager>(cache_config, device_);
        ASSERT_TRUE(cache_manager_->init());
        ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        auto                vec               = vector<int>{1, 2, 3, 4, 5, 6};
        std::vector<size_t> shape             = {6};
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

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_ReturnFalse_WhenMemoryBlockCacheDisabled) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();
    // Default: resource_context_.enable_memory_cache == false and stream_->enableMemoryCache() == false
    ASSERT_FALSE(resource.enableMemoryCache());
    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_ReturnTrue_WhenLoadContextAlreadyExists) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    // Enable memory cache gate
    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    ASSERT_TRUE(resource.enableMemoryCache());

    auto existing                = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    resource.load_cache_context_ = existing;

    ASSERT_TRUE(resource.asyncLoadCache());
    ASSERT_EQ(resource.load_cache_context_.get(), existing.get());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_ReturnFalse_WhenCacheManagerReturnsNull) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    ASSERT_TRUE(resource.enableMemoryCache());

    // KVCacheManager has no connector_coordinator_ by default (kv_cache_config.memory_cache_size_mb == 0),
    // so asyncLoadCache returns nullptr.
    resource.load_cache_context_.reset();
    ASSERT_FALSE(resource.asyncLoadCache());
    ASSERT_EQ(resource.load_cache_context_, nullptr);
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_ReturnTrue_WhenCacheManagerReturnsNonNull) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    ASSERT_TRUE(resource.enableMemoryCache());

    // Inject a mock coordinator into KVCacheManager so asyncLoadCache can succeed.
    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_,
                                                                             device_);
    cache_manager_->connector_coordinator_ = mock_coord;

    auto async_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_coord, asyncRead(testing::_, testing::_)).WillOnce(testing::Return(async_ctx));

    ASSERT_TRUE(resource.asyncLoadCache());
    ASSERT_EQ(resource.load_cache_context_.get(), async_ctx.get());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_ReturnTrue_WhenNoContext) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();
    resource.load_cache_context_.reset();
    ASSERT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_ReturnFalse_WhenNotDone) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    auto ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ctx, done()).WillByDefault(testing::Return(false));
    resource.load_cache_context_ = ctx;

    ASSERT_FALSE(resource.loadCacheDone());
    ASSERT_NE(resource.load_cache_context_, nullptr);
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_ReturnTrue_WhenSuccessAndFusedAsyncReadContext) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    // Build a successful FusedAsyncReadContext
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto fused_match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto kv_resource = std::make_shared<KVCacheResource>();
    kv_resource->setReuseBlocksNum(3);

    auto read_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*read_child, success()).WillByDefault(testing::Return(true));
    auto fused_read = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_child});

    auto read_ctx = std::make_shared<FusedAsyncReadContext>(fused_match, kv_resource);
    read_ctx->setFusedReadContext(fused_read);
    resource.load_cache_context_ = read_ctx;

    ASSERT_TRUE(resource.loadCacheDone());
    ASSERT_EQ(resource.load_cache_context_, nullptr);

    const int expected_reuse_len = 3 * resource.seqSizePerBlock();  // seq_size_per_block = 2 in init_config()
    EXPECT_EQ(stream_->initialReuseLength(), expected_reuse_len);
    EXPECT_EQ(stream_->reuseLength(), expected_reuse_len);
    EXPECT_EQ(stream_->localReuseLength(), expected_reuse_len);
    EXPECT_EQ(stream_->getMtpTokenIndex(), static_cast<size_t>(expected_reuse_len));
}

TEST_F(StreamCacheResourceTest, testAsyncStoreCache_ReturnFalse_WhenMemoryBlockCacheDisabled) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();
    ASSERT_FALSE(resource.enableMemoryCache());
    ASSERT_FALSE(resource.asyncStoreCache());
}

TEST_F(StreamCacheResourceTest, testAsyncStoreCache_ReturnTrue_WhenStoreContextAlreadyExists) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    ASSERT_TRUE(resource.enableMemoryCache());

    auto existing                 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    resource.store_cache_context_ = existing;

    ASSERT_TRUE(resource.asyncStoreCache());
    ASSERT_EQ(resource.store_cache_context_.get(), existing.get());
}

TEST_F(StreamCacheResourceTest, testAsyncStoreCache_ReturnTrue_WhenCacheManagerReturnsNonNull) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();

    resource.resource_context_.enable_memory_cache                 = true;
    stream_->generate_input_->generate_config->enable_memory_cache = true;
    ASSERT_TRUE(resource.enableMemoryCache());

    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                             cache_manager_->kv_cache_config_,
                                                                             cache_manager_->runtime_config_,
                                                                             cache_manager_->allocator_,
                                                                             device_);
    cache_manager_->connector_coordinator_ = mock_coord;

    auto async_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_coord, asyncWrite(testing::_, testing::_)).WillOnce(testing::Return(async_ctx));

    ASSERT_TRUE(resource.asyncStoreCache());
    ASSERT_EQ(resource.store_cache_context_.get(), async_ctx.get());
}

}  // namespace rtp_llm
