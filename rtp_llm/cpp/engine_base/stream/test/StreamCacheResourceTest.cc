
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <chrono>
#include <condition_variable>
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

std::shared_ptr<LoadAsyncContext> makeAllocatorLoadContext(size_t matched_blocks,
                                                            const std::vector<Tier>& source_tiers,
                                                            bool                     commit = true) {
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors;
    descriptors.reserve(source_tiers.size());
    for (size_t path_index = 0; path_index < source_tiers.size(); ++path_index) {
        descriptors.emplace_back(
            nullptr, /*group_set_id=*/0, path_index, source_tiers[path_index], Tier::DEVICE, BlockIndicesType{1});
    }
    auto context = coordinator->create(std::move(descriptors),
                                       std::vector<bool>(source_tiers.size(), false),
                                       matched_blocks);
    EXPECT_TRUE(coordinator->registerContext(context));
    if (commit) {
        EXPECT_TRUE(context->commit());
    }
    return context;
}

class StreamReadStorageBackend: public StorageBackend {
public:
    ~StreamReadStorageBackend() override {
        shutdown();
    }

    void blockMatches() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_matches_ = false;
    }

    void releaseMatches() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_matches_ = true;
        cv_.notify_all();
    }

    void blockReads() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_reads_ = false;
    }

    void releaseReads() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_reads_ = true;
        cv_.notify_all();
    }

    void waitForMatches(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return match_calls_ >= count; });
    }

    void waitForReads(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return read_calls_ >= count; });
    }

    size_t matchCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return match_calls_;
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++match_calls_;
        cv_.notify_all();
        cv_.wait(lock, [&] { return release_matches_; });
        return {request.handles.size(), nullptr};
    }

    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++read_calls_;
        cv_.notify_all();
        cv_.wait(lock, [&] { return release_reads_; });
    }

    void writeImpl(const StorageRequest&) override {}

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    release_matches_{true};
    bool                    release_reads_{true};
    size_t                  match_calls_{0};
    size_t                  read_calls_{0};
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

    std::shared_ptr<StreamReadStorageBackend> prepareStorageBackendResource(
        bool block_matches, bool seed_host = false, RoleType role_type = RoleType::PDFUSION) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.enable_remote_cache  = true;
        kv_cache_config.enable_host_cache  = seed_host;
        kv_cache_config.host_cache_size_mb = seed_host ? 1 : 0;
        prepareResourceWithCacheConfig(
            init_config(), {1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, role_type, kv_cache_config);

        auto backend = std::make_shared<StreamReadStorageBackend>();
        if (block_matches) {
            backend->blockMatches();
        }
        cache_manager_->allocator_->block_tree_cache_.reset();
        cache_manager_->block_tree_cache_.reset();
        auto cache = createBlockTreeCache(
            cache_manager_->cacheConfig(), kv_cache_config, cache_manager_->allocator_, ParallelismConfig{}, backend);
        EXPECT_NE(cache, nullptr);
        cache_manager_->block_tree_cache_ = cache;
        cache_manager_->allocator_->attachBlockTreeCache(cache);

        if (seed_host) {
            auto& resource = stream_->streamCacheResource();
            initCacheKeys(resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), 2);
            const GroupSetPtr& group = cache->groupSets().front();
            const BlockIdxType block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            RTP_LLM_CHECK(!isNullBlockIdx(block));
            std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
            resources[0][0].host_block = block;
            RTP_LLM_CHECK(cache->tree()
                              ->insertNode({resource.batch_kv_cache_resource_->cacheKeys(0).front()}, resources, false)
                              .accepted_resource_count
                          == 1);
            group->releaseSingleBlock(Tier::HOST, block, BlockTreeRefType::CACHE);
        }

        auto& resource                                                 = stream_->streamCacheResource();
        resource.resource_context_.enable_remote_cache                 = true;
        stream_->generate_input_->generate_config->enable_remote_cache = true;
        resource.kvCacheMutable().setBatchCacheKeys(0, {100, 200, 300});
        return backend;
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

TEST_F(StreamCacheResourceTest, testResourceContextReadsIgnoreRequestCacheSwitchesEnv) {
    {
        autil::EnvGuard env_guard("RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES", "0");
        ResourceContext resource_context;
        KVCacheConfig    kv_cache_config;
        resource_context.initCacheConfig(kv_cache_config);
        EXPECT_FALSE(resource_context.ignore_request_cache_switches);
    }
    {
        autil::EnvGuard env_guard("RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES", "1");
        ResourceContext resource_context;
        KVCacheConfig    kv_cache_config;
        resource_context.initCacheConfig(kv_cache_config);
        EXPECT_TRUE(resource_context.ignore_request_cache_switches);
    }
}

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

TEST_F(StreamCacheResourceTest, testReuseCacheIgnoresPerRequestSwitchWhenConfigured) {
    prepareResource(true);
    auto& resource                                           = stream_->streamCacheResource();
    resource.resource_context_.ignore_request_cache_switches = true;

    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_TRUE(resource.reuseCache());

    resource.resource_context_.reuse_cache = false;
    ASSERT_FALSE(resource.reuseCache());
}

TEST_F(StreamCacheResourceTest, testCacheLookupIgnoresPerRequestTierSwitches) {
    prepareResource(true);
    auto& resource   = stream_->streamCacheResource();
    auto& request    = *stream_->generate_input_->generate_config;
    auto& deployment = resource.resource_context_;

    request.reuse_cache         = true;
    request.enable_device_cache = false;
    request.enable_host_cache   = false;
    request.enable_disk_cache   = false;
    request.enable_remote_cache = false;

    for (const bool device_on : {false, true}) {
        for (const bool host_on : {false, true}) {
            for (const bool disk_on : {false, true}) {
                for (const bool remote_on : {false, true}) {
                    SCOPED_TRACE("L1=" + std::to_string(device_on) + " L2=" + std::to_string(host_on)
                                 + " L3=" + std::to_string(disk_on) + " remote=" + std::to_string(remote_on));
                    deployment.enable_device_cache = device_on;
                    deployment.enable_host_cache   = host_on;
                    deployment.enable_disk_cache   = disk_on;
                    deployment.enable_remote_cache = remote_on;
                    EXPECT_EQ(resource.enableCacheLookup(), device_on || host_on || disk_on || remote_on);
                }
            }
        }
    }

    deployment.enable_device_cache = true;
    request.reuse_cache            = false;
    EXPECT_FALSE(resource.enableCacheLookup());

    deployment.ignore_request_cache_switches = true;
    EXPECT_TRUE(resource.enableCacheLookup());
}

TEST_F(StreamCacheResourceTest, testStoreTargetPicksHighestMutuallyPermittedTier) {
    prepareResource(true);
    auto& resource   = stream_->streamCacheResource();
    auto& request    = *stream_->generate_input_->generate_config;
    auto& deployment = resource.resource_context_;

    request.reuse_cache            = true;
    request.enable_device_cache    = true;
    request.enable_host_cache      = true;
    request.enable_disk_cache      = true;
    deployment.enable_device_cache = true;
    deployment.enable_host_cache   = true;
    deployment.enable_disk_cache   = true;
    EXPECT_EQ(resource.storeTarget(), Tier::DEVICE);

    request.enable_device_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::HOST);

    deployment.enable_host_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::DISK);

    request.enable_disk_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::NONE);

    request.enable_disk_cache      = true;
    deployment.enable_host_cache   = false;
    EXPECT_EQ(resource.storeTarget(), Tier::DISK);

    request.reuse_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::NONE);
}

TEST_F(StreamCacheResourceTest, testStoreTargetIgnoresPerRequestSwitchesWhenConfigured) {
    prepareResource(true);
    auto& resource   = stream_->streamCacheResource();
    auto& request    = *stream_->generate_input_->generate_config;
    auto& deployment = resource.resource_context_;

    deployment.ignore_request_cache_switches = true;
    request.reuse_cache                       = false;
    request.enable_device_cache               = false;
    request.enable_host_cache                 = false;
    request.enable_disk_cache                 = false;
    deployment.enable_device_cache            = true;
    deployment.enable_host_cache              = true;
    deployment.enable_disk_cache              = true;
    EXPECT_EQ(resource.storeTarget(), Tier::DEVICE);

    deployment.enable_device_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::HOST);

    deployment.enable_host_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::DISK);

    deployment.enable_disk_cache = false;
    EXPECT_EQ(resource.storeTarget(), Tier::NONE);
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
            EXPECT_FALSE(info.enable_cache_lookup);
            return {true, 0};
        }));

    EXPECT_CALL(*allocator, incrMalloc(testing::_))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // initKVBlock should force-disable cache reuse on the first malloc for decode hybrid.
            EXPECT_FALSE(info.reuse_cache);
            EXPECT_FALSE(info.enable_cache_lookup);
            // Simulate a successful allocation so subsequent calls go through incrMalloc path.
            for (int b = 0; b < info.batch_kv_cache_resource->batchSize(); ++b) {
                auto& block_ids = info.batch_kv_cache_resource->mutableBlockIds(b, /*group_id=*/0);
                block_ids.assign(BlockIndicesType{/*block=*/1});
            }
            return {true, 0};
        }))
        .WillOnce(testing::Invoke([&](const MallocInfo& info) -> MallocResult {
            // incrKVBlock uses the runtime lookup policy.
            EXPECT_TRUE(info.enable_cache_lookup);
            return {true, 0};
        }));

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.incrKVBlock().ok());
}

TEST_F(StreamCacheResourceTest, testAsyncLoadCache_WithoutAllocatorContext_ReturnsFalse) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator-owned load context is in flight.
    ASSERT_FALSE(resource.asyncLoadCache());
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_NoContext_ReturnsTrue) {
    prepareResource(/*reuse_cache=*/false);
    auto& resource = stream_->streamCacheResource();

    // No allocator load context means the load phase is immediately done.
    ASSERT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testCacheReuseMetricsKeepBlockAlignedInputLength) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource = stream_->streamCacheResource();

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(resource.cache_reuse_metrics_.block_aligned_input_length, 6);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_TRUE(resource.cache_reuse_metrics_.report_match_to_ready_latency);
}

TEST_F(StreamCacheResourceTest, testCacheLoadLatencySegmentsCoverMatchToReady) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource             = stream_->streamCacheResource();
    const int64_t        malloc_begin_time_us = currentTimeUs() - 400;
    MallocResult         result{true, 2};
    result.match_cost_time_us      = 100;
    result.match_end_time_us       = malloc_begin_time_us + 100;
    result.malloc_begin_time_us    = malloc_begin_time_us;
    result.load_prepare_latency_us = 300;
    result.load_attempted          = true;

    resource.cache_reuse_metrics_.load_success                  = false;
    resource.cache_reuse_metrics_.report_load_metrics           = true;
    resource.cache_reuse_metrics_.report_load_wait_latency      = true;
    resource.cache_reuse_metrics_.load_wait_latency_us          = 123;
    resource.cache_reuse_metrics_.report_match_to_ready_latency = true;
    resource.cache_reuse_metrics_.match_to_ready_latency_us     = 456;
    resource.recordCacheReuseMallocResult(result);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_wait_latency);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_match_to_ready_latency);
    EXPECT_EQ(resource.cache_reuse_metrics_.load_wait_latency_us, 0);
    EXPECT_EQ(resource.cache_reuse_metrics_.match_to_ready_latency_us, 0);
    resource.allocator_load_context_ = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::DEVICE});
    ASSERT_TRUE(resource.loadCacheDone());

    const RtpLLMCacheReuseMetricsCollector& metrics = resource.cache_reuse_metrics_;
    EXPECT_TRUE(metrics.report_match_latency);
    EXPECT_TRUE(metrics.report_load_metrics);
    EXPECT_TRUE(metrics.load_success);
    EXPECT_TRUE(metrics.report_load_wait_latency);
    EXPECT_TRUE(metrics.report_match_to_ready_latency);
    EXPECT_EQ(metrics.load_prepare_latency_us, result.load_prepare_latency_us);
    EXPECT_GE(metrics.match_to_ready_latency_us,
              metrics.match_latency_us + metrics.load_prepare_latency_us + metrics.load_wait_latency_us);
}

TEST_F(StreamCacheResourceTest, testCacheLoadPrepareFailureHasNoWaitLatency) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource             = stream_->streamCacheResource();
    const int64_t        malloc_begin_time_us = currentTimeUs() - 400;
    MallocResult         result{false, 0};
    result.match_cost_time_us      = 100;
    result.match_end_time_us       = malloc_begin_time_us + 100;
    result.malloc_begin_time_us    = malloc_begin_time_us;
    result.load_prepare_latency_us = 300;
    result.load_attempted          = true;

    resource.recordCacheReuseMallocResult(result);

    const RtpLLMCacheReuseMetricsCollector& metrics = resource.cache_reuse_metrics_;
    EXPECT_TRUE(metrics.report_load_metrics);
    EXPECT_FALSE(metrics.load_success);
    EXPECT_FALSE(metrics.report_load_wait_latency);
    EXPECT_TRUE(metrics.report_match_to_ready_latency);
    EXPECT_EQ(metrics.load_prepare_latency_us, result.load_prepare_latency_us);
    EXPECT_GE(metrics.match_to_ready_latency_us, metrics.match_latency_us + metrics.load_prepare_latency_us);
}

TEST_F(StreamCacheResourceTest, testCacheLoadFailureKeepsDeviceReuseMetrics) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    StreamCacheResource& resource = stream_->streamCacheResource();
    kmonitor::MetricsTags kmon_tags;
    kmonitor::MetricsReporterPtr reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    stream_->setMetricsReporter(reporter);

    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    stream_->setHostReuseLength(0);
    stream_->setDiskReuseLength(0);
    resource.cache_reuse_metrics_.block_aligned_input_length = 6;
    resource.load_wait_begin_time_us_                         = currentTimeUs();
    resource.malloc_begin_time_us_                            = resource.load_wait_begin_time_us_;

    auto load_context = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::HOST});
    ASSERT_TRUE(load_context->completeOne(false));
    ASSERT_EQ(load_context->mallocStatus(), MallocStatus::NONE);
    resource.allocator_load_context_ = load_context;
    ASSERT_TRUE(resource.loadCacheDone());

    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->initialReuseLength(), 2);
    EXPECT_EQ(stream_->localReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);

    resource.reportCacheReuseMetrics();

    const RtpLLMCacheReuseMetricsCollector& metrics = resource.cache_reuse_metrics_;
    EXPECT_EQ(metrics.block_aligned_input_length, 6);
    EXPECT_EQ(metrics.kv_cache_reuse_length, 2);
    EXPECT_EQ(metrics.device_reuse_length, 2);
    EXPECT_EQ(metrics.host_reuse_length, 0);
    EXPECT_EQ(metrics.disk_reuse_length, 0);
    EXPECT_FLOAT_EQ(metrics.kv_cache_hit_rate, 100.0f / 3.0f);
    EXPECT_FLOAT_EQ(metrics.device_hit_rate, 100.0f / 3.0f);
    EXPECT_FLOAT_EQ(metrics.host_hit_rate, 0.0f);
    EXPECT_FLOAT_EQ(metrics.disk_hit_rate, 0.0f);
    EXPECT_TRUE(metrics.report_reuse_metrics);
    EXPECT_TRUE(metrics.report_load_metrics);
    EXPECT_FALSE(metrics.load_success);
}

TEST_F(StreamCacheResourceTest, testReleasePendingAllocatorLoadDoesNotFinalizeMetrics) {
    prepareResource(/*reuse_cache=*/true);
    StreamCacheResource& resource             = stream_->streamCacheResource();
    const int64_t        malloc_begin_time_us = currentTimeUs();
    MallocResult         result{true, 0};
    result.match_end_time_us    = malloc_begin_time_us;
    result.malloc_begin_time_us = malloc_begin_time_us;
    result.load_attempted       = true;
    resource.recordCacheReuseMallocResult(result);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false, false);

    resource.releaseResource();

    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_metrics);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_load_wait_latency);
    EXPECT_FALSE(resource.cache_reuse_metrics_.report_match_to_ready_latency);
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

TEST_F(StreamCacheResourceTest, StorageMatchDefersReusePublicationUntilReadCompletes) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/false);
    auto& resource = stream_->streamCacheResource();
    backend->blockReads();

    ASSERT_TRUE(resource.initKVBlock().ok());
    backend->waitForReads(1);
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(resource.allocator_load_context_);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), 0u);
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(resource.kvCache().cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).memoryReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).diskReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).storageBackendReuseBlockNum(), 0u);
    EXPECT_FALSE(resource.loadCacheDone());

    backend->releaseReads();
    context->waitDone();
    ASSERT_TRUE(context->success());
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->remoteReuseLength(), 4);
    EXPECT_EQ(resource.kvCache().cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).memoryReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).diskReuseBlockNum(), 0u);
    EXPECT_EQ(resource.kvCache().cacheResource(0).storageBackendReuseBlockNum(), 2u);
}

TEST_F(StreamCacheResourceTest, StorageMatchDoesNotBlockIndependentAllocation) {
    auto  backend        = prepareStorageBackendResource(/*block_matches=*/true);
    auto& first_resource = stream_->streamCacheResource();
    ASSERT_TRUE(first_resource.initKVBlock().ok());
    backend->waitForMatches(1);

    ResourceContext second_context             = stream_->resourceContext();
    second_context.enable_remote_cache         = true;
    auto second_input                          = std::make_shared<GenerateInput>();
    second_input->input_ids                    = torch::tensor(std::vector<int32_t>{7, 8, 9, 10}, torch::kInt32);
    second_input->generate_config              = std::make_shared<GenerateConfig>();
    second_input->generate_config->reuse_cache = true;
    second_input->generate_config->enable_remote_cache = false;
    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = 2;
    model_config.max_seq_len                  = 2048;
    auto second_stream =
        std::make_shared<NormalGenerateStream>(second_input, model_config, RuntimeConfig{}, second_context, nullptr);
    second_stream->generate_status_->status = StreamState::RUNNING;
    second_stream->streamCacheResource().kvCacheMutable().setBatchCacheKeys(0, {400, 500});

    ASSERT_TRUE(second_stream->streamCacheResource().initKVBlock().ok());
    EXPECT_TRUE(second_stream->streamCacheResource().asyncLoadCache());
    // Request-level remote disable is not propagated below lookup admission. The process-level backend still matches.
    EXPECT_EQ(backend->matchCalls(), 1u);

    backend->releaseMatches();
    backend->waitForMatches(2);
    ASSERT_TRUE(first_resource.waitForAllocatorLoad().ok());
    ASSERT_TRUE(second_stream->streamCacheResource().waitForAllocatorLoad().ok());
}

TEST_F(StreamCacheResourceTest, DeferredResultPublishesOnlyDeviceReadyReuseLength) {
    auto  backend  = prepareStorageBackendResource(/*block_matches=*/true, /*seed_host=*/true);
    auto& resource = stream_->streamCacheResource();

    MallocInfo info;
    info.batch_kv_cache_resource      = resource.batch_kv_cache_resource_;
    info.complete_token_ids           = stream_->completeTokenIdsPtr();
    info.reuse_cache                  = true;
    info.enable_cache_lookup          = true;
    MallocResult result               = cache_manager_->malloc(info);
    ASSERT_TRUE(result.success);
    backend->waitForMatches(1);
    ASSERT_NE(result.async_context, nullptr);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.host_reuse_len, 0);
    EXPECT_EQ(result.disk_reuse_len, 0);

    backend->releaseMatches();
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->success());
    cache_manager_->free(FreeInfo{resource.batch_kv_cache_resource_, stream_->completeTokenIdsPtr()});
}

TEST_F(StreamCacheResourceTest, testLoadCacheDone_CompletedAllocatorLoad_ClearsContext) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    resource.allocator_load_context_ = makeAllocatorLoadContext(/*matched_blocks=*/0, {});
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(resource.loadCacheDone());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadContextGatesReadinessUntilDone) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();

    auto load_context                = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::HOST});
    resource.allocator_load_context_ = load_context;

    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_FALSE(resource.loadCacheDone());
    EXPECT_TRUE(load_context->completeOne(true));
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadSuccessCommitsCompleteReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();

    auto load_context =
        makeAllocatorLoadContext(/*matched_blocks=*/3, {Tier::DEVICE, Tier::HOST, Tier::DISK}, /*commit=*/false);
    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/2, 0, load_context, /*host=*/2, /*disk=*/2}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);

    EXPECT_TRUE(load_context->completeOne(true));
    EXPECT_TRUE(load_context->completeOne(true));
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 6);
    EXPECT_EQ(stream_->initialReuseLength(), 6);
    EXPECT_EQ(stream_->localReuseLength(), 6);
    EXPECT_EQ(stream_->hostReuseLength(), 2);
    EXPECT_EQ(stream_->diskReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
}

TEST_F(StreamCacheResourceTest, testInitRejectsSuccessfulNonLoadAllocatorContext) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();

    auto context   = std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/0, 0, context}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));
    EXPECT_CALL(*allocator, free(testing::_)).Times(1);

    EXPECT_FALSE(resource.initKVBlock().ok());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadSuccessUsesCpGroupPolicyReuseUnit) {
    auto cache_config = init_config();
    auto policies     = cache_config.groupPoliciesSnapshot();
    ASSERT_EQ(policies.size(), 1u);
    policies.front().cp_mapping = CpBlockMappingMode::NONE;
    cache_config.setGroupPolicies(policies);
    prepareResourceWithCacheConfig(
        cache_config, {1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();

    cache_manager_->cp_slot_mapper_ = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/2);
    auto load_context =
        makeAllocatorLoadContext(/*matched_blocks=*/2, {Tier::DEVICE, Tier::HOST}, /*commit=*/false);
    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/2, 0, load_context}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_TRUE(load_context->completeOne(true));
    ASSERT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(stream_->reuseLength(), 4);
    EXPECT_EQ(stream_->initialReuseLength(), 4);
    EXPECT_EQ(stream_->localReuseLength(), 4);
    EXPECT_EQ(stream_->hostReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadPendingPublishesZeroDeviceReadyReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(6);
    stream_->setHostReuseLength(2);
    stream_->setDiskReuseLength(2);

    auto load_context = makeAllocatorLoadContext(/*matched_blocks=*/1, {Tier::HOST}, /*commit=*/false);
    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Return(MallocResult{true, /*reuse_len=*/0, 0, load_context, /*host=*/2, /*disk=*/2}));
    EXPECT_CALL(*allocator, incrMalloc(testing::_)).WillOnce(testing::Return(MallocResult{true, 0}));

    ASSERT_TRUE(resource.initKVBlock().ok());
    EXPECT_EQ(resource.allocator_load_context_, load_context);
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(stream_->initialReuseLength(), 0);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->deviceReuseLength(), 0);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);
}

TEST_F(StreamCacheResourceTest, testPrefillAllocatorLoadFailureKeepsDeviceReadyReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);

    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->initialReuseLength(), 2);
    EXPECT_EQ(stream_->localReuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_EQ(stream_->hostReuseLength(), 0);
    EXPECT_EQ(stream_->diskReuseLength(), 0);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testPrefillMaterializationShortfallRearmsAllocatorLoad) {
    auto backend =
        prepareStorageBackendResource(/*block_matches=*/false, /*seed_host=*/false, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.resourceContext().role_type, RoleType::PREFILL);

    size_t commits = 0;
    size_t aborts  = 0;
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>&) {
            ++commits;
            return true;
        },
        [&](LoadAsyncContext&) { ++aborts; });
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1234}),
                           {{{/*group_id=*/0, NULL_BLOCK_IDX}}}};
    auto context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([](LoadAsyncContext&, size_t) {
        return LoadMatchResult{false, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED};
    });

    resource.allocator_load_context_ = context;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    resource.malloc_failed_times_      = 9;
    context->startBackendMatch();
    context->waitDone();
    ASSERT_FALSE(context->success());

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    // Admission ownership belongs to the scheduler: ordinary loading streams
    // clear CanRun when they move back to WAITING, while explicit groups keep
    // their admission and retry in the group lane.
    EXPECT_TRUE(stream_->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(stream_->hasEvent(StreamEvents::LoadInitiated));
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(resource.mallocFailedTimes(), 10);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);

    auto allocator = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
    cache_manager_->allocator_ = allocator;
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_))
        .WillOnce(testing::Invoke([](const MallocInfo& info) -> MallocResult {
            EXPECT_FALSE(info.verbose);
            return {true, 0};
        }));
    EXPECT_CALL(*allocator, incrMalloc(testing::_))
        .WillOnce(testing::Invoke([](const MallocInfo& info) -> MallocResult {
            info.batch_kv_cache_resource->mutableBlockIds(0, /*group_id=*/0).assign({1});
            return {true, 0};
        }));

    EXPECT_EQ(stream_->moveToNext(), StreamState::RUNNING);
    EXPECT_TRUE(stream_->hasEvent(StreamEvents::LoadInitiated));
    EXPECT_EQ(resource.curBlocksNum(), 1);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, testPrefillPermanentMaterializationFailureTerminates) {
    auto backend =
        prepareStorageBackendResource(/*block_matches=*/false, /*seed_host=*/false, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.resourceContext().role_type, RoleType::PREFILL);

    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{5678}),
                           {{{/*group_id=*/0, NULL_BLOCK_IDX}}}};
    auto context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([](LoadAsyncContext&, size_t) {
        return LoadMatchResult{false, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED};
    });

    resource.allocator_load_context_ = context;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    context->startBackendMatch();
    context->waitDone();

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, testPrefillCoordinatorCommitFailureTerminates) {
    auto backend =
        prepareStorageBackendResource(/*block_matches=*/false, /*seed_host=*/false, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.resourceContext().role_type, RoleType::PREFILL);

    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return false; }, [](LoadAsyncContext&) {});
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{9012}),
                           {{{/*group_id=*/0, NULL_BLOCK_IDX}}}};
    auto context = coordinator->create({}, {}, /*matched_blocks=*/0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([](LoadAsyncContext& current, size_t) { return current.commit(); });

    resource.allocator_load_context_ = context;
    stream_->reportEvent(StreamEvents::CanRun);
    stream_->reportEvent(StreamEvents::LoadInitiated);
    stream_->generate_status_->status = StreamState::LOADING_CACHE;
    context->startBackendMatch();
    context->waitDone();
    ASSERT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);

    EXPECT_EQ(stream_->moveToNext(), StreamState::WAITING);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(resource.curBlocksNum(), 0);
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    coordinator->shutdown();
}

TEST_F(StreamCacheResourceTest, testPrefillWaitForAllocatorLoadFailureKeepsDeviceReadyReuse) {
    prepareResource(/*reuse_cache=*/true, RoleType::PREFILL);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);
    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);

    EXPECT_TRUE(resource.waitForAllocatorLoad().ok());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_EQ(stream_->reuseLength(), 2);
    EXPECT_EQ(stream_->deviceReuseLength(), 2);
    EXPECT_FALSE(stream_->hasError());
}

TEST_F(StreamCacheResourceTest, testAllocatorLoadFailureIsTerminal) {
    prepareResource(/*reuse_cache=*/true);
    auto& resource = stream_->streamCacheResource();
    stream_->setReuseLength(2);
    stream_->setMtpTokenIndex(2);
    stream_->setInitialReuseLength(2);
    stream_->setLocalReuseLength(2);

    resource.allocator_load_context_ = std::make_shared<ImmediateAllocatorContext>(false);
    EXPECT_TRUE(resource.asyncLoadCache());
    EXPECT_TRUE(resource.loadCacheDone());
    EXPECT_EQ(resource.allocator_load_context_, nullptr);
    EXPECT_TRUE(stream_->hasError());
    EXPECT_EQ(stream_->reuseLength(), 0);
    EXPECT_EQ(stream_->initialReuseLength(), 0);
    EXPECT_EQ(stream_->localReuseLength(), 0);
    EXPECT_EQ(stream_->deviceReuseLength(), 0);
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
