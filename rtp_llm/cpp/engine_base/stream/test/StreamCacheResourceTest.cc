
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

#include <memory>

using namespace std;

namespace rtp_llm {

class ImmediateTestAsyncContext: public AsyncContext {
public:
    explicit ImmediateTestAsyncContext(bool success, bool done = true): success_(success), done_(done) {}
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

class PendingIncrementContext: public AsyncContext {
public:
    void waitDone() override {
        ++wait_calls_;
    }

    bool done() const override {
        return done_;
    }

    bool success() const override {
        return true;
    }

    void complete() {
        done_ = true;
    }

    size_t waitCalls() const {
        return wait_calls_;
    }

private:
    bool   done_{false};
    size_t wait_calls_{0};
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

TEST_F(StreamCacheResourceTest, testLoadCacheDone_NoContext_ReturnsTrue) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator_load_context_ -> immediately done.
    ASSERT_TRUE(resource.loadCacheDone());
}

// C005-T02: zero-block increment rejects before reserve, keys, allocator, blocks, or contexts mutate.
TEST_F(StreamCacheResourceTest, testZeroBlockIncrementRejectsBeforeManagerMutation) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto       real_allocator                         = cache_manager_->allocator_;
    const auto free_before                            = cache_manager_->freeBlocksNum();
    auto       allocator_context                      = std::make_shared<PendingIncrementContext>();
    resource.allocator_load_context_                  = allocator_context;
    const auto                 reserve_before         = stream_->completeTokenIdsPtr()->getReserveStep();
    const auto                 malloc_failures_before = resource.mallocFailedTimes();
    std::vector<CacheKeysType> keys_before;
    for (int batch_id = 0; batch_id < resource.kvCache().batchSize(); ++batch_id) {
        keys_before.push_back(resource.kvCache().cacheKeys(batch_id));
    }

    auto mock_allocator        = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = mock_allocator;
    EXPECT_CALL(*mock_allocator, initMallocForCommonLen(testing::_)).Times(0);
    EXPECT_CALL(*mock_allocator, incrMalloc(testing::_)).Times(0);

    const auto status = resource.incrKVBlock(/*reserve_step=*/7);
    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(status.message().find("initialized KV cache resource"), std::string::npos);
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(stream_->completeTokenIdsPtr()->getReserveStep(), reserve_before);
    EXPECT_EQ(resource.mallocFailedTimes(), malloc_failures_before);
    EXPECT_EQ(resource.allocator_load_context_, allocator_context);
    for (int batch_id = 0; batch_id < resource.kvCache().batchSize(); ++batch_id) {
        EXPECT_EQ(resource.kvCache().cacheKeys(batch_id), keys_before[static_cast<size_t>(batch_id)]);
    }

    cache_manager_->allocator_ = real_allocator;
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// C005-T02: WAITING after LoadInitiated finishes instead of re-entering RUNNING on zero blocks.
TEST_F(StreamCacheResourceTest, testWaitingAfterLoadInitiatedFinishesOnZeroBlockIncrement) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    auto mock_allocator        = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = mock_allocator;
    EXPECT_CALL(*mock_allocator, initMallocForCommonLen(testing::_)).Times(0);
    EXPECT_CALL(*mock_allocator, incrMalloc(testing::_)).Times(0);

    stream_->setIsContextStream(false);
    stream_->generate_status_->status = StreamState::WAITING;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);

    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_TRUE(resource.isResourceReleased());
    EXPECT_EQ(resource.curBlocksNum(), 0);
}

// C005-T02: RUNNING also terminates before allocator entry when no request blocks exist.
TEST_F(StreamCacheResourceTest, testRunningFinishesAndReleasesOnZeroBlockIncrement) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    auto       real_allocator  = cache_manager_->allocator_;
    const auto free_before     = cache_manager_->freeBlocksNum();
    auto       mock_allocator  = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = mock_allocator;
    EXPECT_CALL(*mock_allocator, initMallocForCommonLen(testing::_)).Times(0);
    EXPECT_CALL(*mock_allocator, incrMalloc(testing::_)).Times(0);
    EXPECT_CALL(*mock_allocator, free(testing::_)).Times(0);

    stream_->setIsContextStream(false);
    stream_->generate_status_->status = StreamState::RUNNING;
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_TRUE(resource.isResourceReleased());
    EXPECT_EQ(resource.curBlocksNum(), 0);

    cache_manager_->allocator_ = real_allocator;
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// C005-T03: a populated non-null increment observer is retained and rejected without waiting or cancellation.
TEST_F(StreamCacheResourceTest, testPopulatedAsyncIncrementRetainsObserverAndFailsClosed) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());

    auto real_allocator = cache_manager_->allocator_;
    auto pool           = real_allocator->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto                         free_before_increment = cache_manager_->freeBlocksNum();
    const auto                         blocks_before         = resource.kvCache().getAllBatchBlocks();
    std::vector<std::vector<uint32_t>> refs_before;
    for (const auto& batch_blocks : blocks_before) {
        std::vector<uint32_t> refs;
        for (const auto block : batch_blocks) {
            refs.push_back(pool->refCount(block));
        }
        refs_before.push_back(std::move(refs));
    }

    auto pending_context       = std::make_shared<PendingIncrementContext>();
    auto mock_allocator        = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = mock_allocator;
    EXPECT_CALL(*mock_allocator, incrMalloc(testing::_))
        .WillOnce(testing::Return(MallocResult{true, 0, 0, pending_context}));

    const auto status = resource.incrKVBlock(/*reserve_step=*/0);
    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(status.message().find("async incremental"), std::string::npos);
    EXPECT_EQ(resource.allocator_load_context_, pending_context);
    EXPECT_EQ(pending_context->waitCalls(), 0u);
    EXPECT_FALSE(pending_context->done());
    EXPECT_EQ(resource.kvCache().getAllBatchBlocks(), blocks_before);
    EXPECT_EQ(real_allocator->freeBlocksNum(), free_before_increment);
    for (size_t batch_id = 0; batch_id < blocks_before.size(); ++batch_id) {
        for (size_t block_id = 0; block_id < blocks_before[batch_id].size(); ++block_id) {
            EXPECT_EQ(pool->refCount(blocks_before[batch_id][block_id]), refs_before[batch_id][block_id]);
        }
    }

    cache_manager_->allocator_ = real_allocator;
    stream_->releaseResource();
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), 8u);
    EXPECT_EQ(pending_context->waitCalls(), 0u);
    EXPECT_FALSE(pending_context->done());
    pending_context->complete();
    EXPECT_TRUE(pending_context->done());
}

// C005-T03: RUNNING takes the existing terminal/error release path for a future async increment.
TEST_F(StreamCacheResourceTest, testRunningFinishesAndReleasesOnAsyncIncrementRejection) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());

    auto real_allocator = cache_manager_->allocator_;
    auto pool           = real_allocator->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto allocated_blocks = resource.kvCache().getAllBatchBlocks();
    auto       pending_context  = std::make_shared<PendingIncrementContext>();
    auto       mock_allocator   = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_  = mock_allocator;

    EXPECT_CALL(*mock_allocator, incrMalloc(testing::_))
        .WillOnce(testing::Return(MallocResult{true, 0, 0, pending_context}));
    EXPECT_CALL(*mock_allocator, insertIntoCache(testing::_)).Times(0);
    EXPECT_CALL(*mock_allocator, free(testing::_)).WillOnce(testing::Invoke([&](const FreeInfo& free_info) {
        real_allocator->free(free_info);
    }));

    stream_->setIsContextStream(false);
    stream_->generate_status_->status = StreamState::RUNNING;
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_TRUE(resource.isResourceReleased());
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(pending_context->waitCalls(), 0u);
    EXPECT_FALSE(pending_context->done());
    EXPECT_EQ(real_allocator->freeBlocksNum(), 8u);
    for (const auto& batch_blocks : allocated_blocks) {
        for (const auto block : batch_blocks) {
            EXPECT_FALSE(pool->isAllocated(block));
        }
    }

    pending_context->complete();
    EXPECT_TRUE(pending_context->done());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadContextGatesExecutionUntilDone) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto load_context                = std::make_shared<ImmediateTestAsyncContext>(true, false);
    resource.allocator_load_context_ = load_context;

    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_FALSE(resource.loadCacheDone());
    load_context->setDone(true);
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadFailureReportsStreamError) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    resource.allocator_load_context_ = std::make_shared<ImmediateTestAsyncContext>(false);
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(stream_->hasError());
}

}  // namespace rtp_llm
