
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/CacheManager.h"
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
        CacheConfig config(KVCacheParam({3, 9, 1, 1, 2, TYPE_INT8}));
        return config;
    }

    void prepareResource(bool reuse_cache = false) {
        auto cache_config = init_config();
        cache_manager_    = std::make_shared<CacheManager>(cache_config, device_);
        ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
        allocator_ = cache_manager_->kvCacheAllocator();
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
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        stream_             = std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);
        stream_->setRunning();
    }

    void checkBlockFunc(const vector<vector<int>>& block_vec, int outter_size, int inner_size) {
        ASSERT_EQ(block_vec.size(), outter_size);
        ASSERT_EQ(block_vec[0].size(), inner_size);
    };

#define CHECK_BLOCK(block_vec, outter_size, inner_size)                                                                \
    do {                                                                                                               \
        SCOPED_TRACE("checkBlockFunc");                                                                                \
        checkBlockFunc(block_vec, outter_size, inner_size);                                                            \
    } while (0)

protected:
    autil::EnvGuard     perf_scope;
    GenerateStreamPtr   stream_;
    CacheManagerPtr     cache_manager_;
    KVCacheAllocatorPtr allocator_;
};

TEST_F(StreamCacheResourceTest, testAllocateResource) {
    prepareResource();

    auto& resource = stream_->streamCacheResource();

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    CHECK_BLOCK(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    CHECK_BLOCK(blocks.batch_block_id, 2, 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    ASSERT_EQ(blocks.batch_block_id.size(), 0);
}

TEST_F(StreamCacheResourceTest, testReuseCache) {
    prepareResource(true);
    auto& resource = stream_->streamCacheResource();

    ASSERT_TRUE(resource.initKVBlock().ok());
    stream_->setSeqLength(9);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    auto batch_tokens_1                      = stream_->complete_token_ids_->data(0);
    batch_tokens_1[stream_->seqLength() - 3] = 7;
    batch_tokens_1[stream_->seqLength() - 2] = 8;
    auto batch_tokens_2                      = stream_->complete_token_ids_->data(1);
    batch_tokens_2[stream_->seqLength() - 3] = 9;
    batch_tokens_2[stream_->seqLength() - 2] = 10;

    stream_->setFinishedWithoutLock();
    stream_->releaseResource();

    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);
    ASSERT_TRUE(cache_manager_->blockCache().hasKey({1, 2, 3, 4, 5, 6, 7, 8}));
    ASSERT_TRUE(cache_manager_->blockCache().hasKey({1, 2, 3, 4, 5, 6, 9, 10}));
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(6), 1);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(7), 0);

    // test another stream
    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    generate_config->num_return_sequences = 2;
    auto                vec               = vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 11};
    std::vector<size_t> shape             = {9};
    generate_input->input_ids =
        std::make_unique<rtp_llm::Buffer>(rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(vec.data()));
    generate_input->generate_config = generate_config;

    ResourceContext resource_context;
    resource_context.reuse_cache   = true;
    resource_context.cache_manager = cache_manager_;
    ModelConfig model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len = 2048;
    stream_             = std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);
    stream_->setRunning();

    auto& resource2 = stream_->streamCacheResource();
    ASSERT_TRUE(resource2.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    stream_->setIsContextStream(false);
    stream_->setSeqLength(10);
    ASSERT_TRUE(resource2.incrKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    stream_->setSeqLength(11);
    ASSERT_TRUE(resource2.incrKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 0);

    auto tokens_1                      = stream_->complete_token_ids_->data(0);
    tokens_1[stream_->seqLength() - 2] = 12;
    tokens_1[stream_->seqLength() - 1] = 13;
    auto tokens_2                      = stream_->complete_token_ids_->data(1);
    tokens_2[stream_->seqLength() - 2] = 14;
    tokens_2[stream_->seqLength() - 1] = 15;

    stream_->setFinishedWithoutLock();
    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 2);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 3);
}

TEST_F(StreamCacheResourceTest, testTryReleaseKVBlock) {
    prepareResource(false);
    auto& resource = stream_->streamCacheResource();

    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(3), 2);

    stream_->setFinishedWithoutLock();
    resource.tryReleaseKVBlock(1);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);
    ASSERT_EQ(resource.maxBlockSize(), 2);

    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(3), 0);

    stream_->setFinishedWithoutLock();
    resource.tryReleaseKVBlock(2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
    ASSERT_EQ(resource.maxBlockSize(), 0);

    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(1), 0);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(3), 0);

    ASSERT_TRUE(resource.incrKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);

    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(allocator_->blockRefCounter().getRefCounter(6), 0);

    stream_->setFinishedWithoutLock();
    resource.tryReleaseKVBlock(2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 7);
    ASSERT_EQ(resource.maxBlockSize(), 1);

    resource.resource_context_.reuse_cache = true;
    auto tokens_1                          = stream_->complete_token_ids_->data(0);
    tokens_1[0]                            = 1;
    auto tokens_2                          = stream_->complete_token_ids_->data(1);
    tokens_2[0]                            = 2;

    stream_->setFinishedWithoutLock();
    resource.tryReleaseKVBlock(1);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 7);
    ASSERT_EQ(resource.maxBlockSize(), 0);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);

    stream_->setFinishedWithoutLock();
    resource.tryReleaseKVBlock(1);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 7);
    ASSERT_EQ(resource.maxBlockSize(), 0);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);
}

TEST_F(StreamCacheResourceTest, testQueryLevelReuseCacheControl) {
    // Test query-level reuse_cache control when engine-level is enabled
    prepareResource(true);  // Enable engine-level reuse_cache
    auto& resource = stream_->streamCacheResource();

    // Test with query-level reuse_cache = true
    stream_->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);

    // Test with query-level reuse_cache = false
    stream_->releaseResource();
    // Re-initialize batch resource after release
    resource.init(stream_->currentBatchSize());
    size_t baseline_free_blocks                            = cache_manager_->freeBlockNums();
    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(),
              baseline_free_blocks >= 3 ? baseline_free_blocks - 3 : baseline_free_blocks);
    ASSERT_EQ(resource.maxBlockSize(), 3);

    stream_->releaseResource();
}

TEST_F(StreamCacheResourceTest, testQueryLevelReuseCacheMasterSwitch) {
    // Test that query-level reuse_cache is ignored when engine-level is disabled
    prepareResource(false);  // Disable engine-level reuse_cache
    auto& resource = stream_->streamCacheResource();

    // Test with query-level reuse_cache = true, but should be ignored
    stream_->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);

    // Test with query-level reuse_cache = false, should also be ignored
    stream_->releaseResource();
    // Re-initialize batch resource after release
    resource.init(stream_->currentBatchSize());
    stream_->generate_input_->generate_config->reuse_cache = false;
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);

    stream_->releaseResource();
}

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

TEST_F(StreamCacheResourceTest, testStreamCacheResourceEnable3FSMethod) {
    // Start with engine=true to test query toggling
    prepareResource(true);
    auto& resource = stream_->streamCacheResource();

    // engine=true, query=true -> true
    resource.resource_context_.enable_3fs                 = true;
    stream_->generate_input_->generate_config->enable_3fs = true;
    ASSERT_TRUE(resource.enable3FS());

    // engine=true, query=false -> false
    stream_->generate_input_->generate_config->enable_3fs = false;
    ASSERT_FALSE(resource.enable3FS());

    // engine=false, query=true -> false
    resource.resource_context_.enable_3fs                 = false;
    stream_->generate_input_->generate_config->enable_3fs = true;
    ASSERT_FALSE(resource.enable3FS());

    // engine=false, query=false -> false
    stream_->generate_input_->generate_config->enable_3fs = false;
    ASSERT_FALSE(resource.enable3FS());
}

}  // namespace rtp_llm
