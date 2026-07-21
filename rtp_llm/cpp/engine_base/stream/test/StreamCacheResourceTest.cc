
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

#include <chrono>
#include <functional>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class ImmediateAllocatorContext: public AsyncContext {
public:
    explicit ImmediateAllocatorContext(bool success, bool done = true): success_(success), done_(done) {}

    void waitDone() override {
        done_ = true;
    }

    bool done() const override {
        return done_;
    }

    bool success() const override {
        return success_;
    }

    void setDone(bool done) {
        done_ = done;
    }

private:
    bool success_{false};
    bool done_{true};
};

class DestructionObserverContext: public AsyncContext {
public:
    explicit DestructionObserverContext(std::function<void()> observer): observer_(std::move(observer)) {}

    ~DestructionObserverContext() override {
        observer_();
    }

    void waitDone() override {}
    bool done() const override {
        return false;
    }
    bool success() const override {
        return false;
    }

private:
    std::function<void()> observer_;
};

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
                                        RoleType                role_type,
                                        const KVCacheConfig&    kv_cache_config      = {},
                                        size_t                  expected_free_blocks = 8) {
        cache_manager_ = std::make_shared<KVCacheManager>(
            cache_config, /*warmup=*/false, /*metrics_reporter=*/nullptr, kv_cache_config);
        ASSERT_TRUE(cache_manager_->init());
        ASSERT_EQ(cache_manager_->freeBlocksNum(), expected_free_blocks);
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
            ASSERT_EQ(batch_resource.blocks(i, 0).size(), inner_size);
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

TEST_F(StreamCacheResourceTest, testWarmUpFakeInitUsesTaggedTopology) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    stream_ =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.kvCache().groupNums(), 1);
    EXPECT_EQ(resource.kvCache().cacheResource().soleGroupTagForLayer(0), "__warmup__");
    EXPECT_EQ(resource.curBlocksNum(), 0);

    stream_->fakeInitKVBlock(2);
    EXPECT_EQ(resource.kvCache().blocks(0, "__warmup__").size(), 2);
}

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
//     ASSERT_TRUE(resource.initKVBlock().ok());
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

//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     stream_->setSeqLength(7);
//     stream_->setIsContextStream(false);
//     ASSERT_TRUE(resource.incrKVBlock().ok());
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
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     // Test with query-level reuse_cache = false
//     stream_->releaseResource();
//     // Re-initialize batch resource after release
//     resource.init(stream_->currentBatchSize());
//     size_t baseline_free_blocks                            = cache_manager_->freeBlocksNum();
//     stream_->generate_input_->generate_config->reuse_cache = false;
//     ASSERT_TRUE(resource.initKVBlock().ok());
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
//     ASSERT_TRUE(resource.initKVBlock().ok());
//     ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
//     ASSERT_EQ(resource.maxBlockSize(), 3);

//     // Test with query-level reuse_cache = false, should also be ignored
//     stream_->releaseResource();
//     // Re-initialize batch resource after release
//     resource.init(stream_->currentBatchSize());
//     stream_->generate_input_->generate_config->reuse_cache = false;
//     ASSERT_TRUE(resource.initKVBlock().ok());
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

TEST_F(StreamCacheResourceTest, testDecodeInitKVBlock_DisablesDeviceCacheOnlyForFirstMalloc) {
    auto cache_config                                     = test::makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                             /*block_num=*/9,
                                                             /*tokens_per_block=*/2,
                                                             rtp_llm::DataType::TYPE_FP16,
                                                             /*group_layer_num=*/2);
    cache_config.disable_decode_first_malloc_device_reuse = true;
    prepareResourceWithCacheConfig(cache_config, {1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, RoleType::DECODE);
    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(cache_manager_->cacheConfig().groupNums(), 1);

    // Enable query-level reuse/device cache, but decode initKVBlock should still force device cache off.
    stream_->generate_input_->generate_config->reuse_cache         = true;
    stream_->generate_input_->generate_config->enable_device_cache = true;
    resource.resource_context_.enable_device_cache                 = true;

    auto allocator             = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;

    testing::InSequence seq;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            EXPECT_FALSE(info.reuse_cache);
            EXPECT_FALSE(info.enable_device_cache);
            return {true, 0};
        }));

    EXPECT_CALL(*allocator, incrMalloc(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // initKVBlock should force-disable cache reuse on the first malloc for decode hybrid.
            EXPECT_FALSE(info.reuse_cache);
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

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.incrKVBlock().ok());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_WithoutAllocatorContext_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator-owned load-back context is in flight.
    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_NoContext_ReturnsTrue) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator load context means the load phase is immediately done.
    ASSERT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_PendingAllocatorLoad_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto load_context                = std::make_shared<ImmediateAllocatorContext>(true, false);
    resource.allocator_load_context_ = load_context;

    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_FALSE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, load_context);
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_CompletedAllocatorLoad_ClearsContext) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(true);
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadContextGatesReadinessUntilDone) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto load_context                = std::make_shared<ImmediateAllocatorContext>(true, false);
    resource.allocator_load_context_ = load_context;

    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_FALSE(resource.loadCacheDone());
    load_context->setDone(true);
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadFailureIsTerminal) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testReleaseResetsAllocatorContextBeforeFreeingRequestBlocks) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_GT(resource.curBlocksNum(), 0);
    ASSERT_LT(cache_manager_->freeBlocksNum(), 8u);

    bool context_destroyed_before_free       = false;
    bool request_blocks_still_present        = false;
    resource.allocator_load_context_         = std::make_shared<DestructionObserverContext>([&] {
        context_destroyed_before_free = true;
        request_blocks_still_present  = resource.curBlocksNum() > 0 && cache_manager_->freeBlocksNum() < 8u;
    });
    std::weak_ptr<AsyncContext> weak_context = resource.allocator_load_context_;

    stream_->releaseResource();

    EXPECT_TRUE(context_destroyed_before_free);
    EXPECT_TRUE(request_blocks_still_present);
    EXPECT_TRUE(weak_context.expired());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
}

}  // namespace rtp_llm
