
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class StreamCacheResourceTest: public DeviceTestBase {
protected:
    StreamCacheResourceTest(): perf_scope("PERF_TEST", "1") {}

    CacheConfig init_config() {
        CacheConfig config;
        config.layer_num          = 3;
        config.block_num          = 9;
        config.seq_size_per_block = 2;  // tokens_per_block

        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->layer_num          = 3;
        spec->block_nums         = 9;
        spec->local_head_num_kv  = 1;
        spec->size_per_head      = 1;
        spec->seq_size_per_block = 2;
        spec->dtype              = TYPE_INT8;
        spec->type               = KVCacheType::MultiHeadAttention;
        config.cache_specs.push_back(spec);

        std::vector<int> layer_ids(3);
        for (int i = 0; i < 3; ++i) {
            layer_ids[i] = i;
        }
        config.layer_ids.push_back(layer_ids);
        return config;
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
        rtp_llm::GptInitParameter params;
        params.max_seq_len_ = 2048;
        stream_             = std::make_shared<NormalGenerateStream>(generate_input, params, resource_context, nullptr);
        stream_->setRunning();
    }

    void checkBlockFunc(const BatchKVCacheResource& blocks, int outter_size, int inner_size) {
        ASSERT_EQ(blocks.batchSize(), outter_size);
        for (int i = 0; i < outter_size; ++i) {
            ASSERT_EQ(blocks.blocks(i).size(), inner_size);
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

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    CHECK_BLOCK(blocks, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 3);

    CHECK_BLOCK(blocks, 2, 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);

    ASSERT_EQ(blocks.batchSize(), 0);
}

TEST_F(StreamCacheResourceTest, testFallback) {
    prepareResource();
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    CHECK_BLOCK(blocks, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    CHECK_BLOCK(blocks, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 3);

    int old_max_blocks = resource.maxBlockSize();
    stream_->setFallbackPrefixLength(4);
    ASSERT_EQ(stream_->fallbackPrefixLength(), 4);

    int released_blocks = resource.tryReleaseKVBlock(old_max_blocks);
    stream_->setPaused();

    ASSERT_EQ(released_blocks, old_max_blocks);
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 8);
    ASSERT_EQ(stream_->fallbackPrefixLength(), 0);
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
