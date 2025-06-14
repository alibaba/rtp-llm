
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/stream/StreamCacheResource.h"
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
    CacheConfig init_config() {
        CacheConfig config(KVCacheParam({3, 9, 1, 1, 2, TYPE_INT8}));
        return config;
    }

    void prepareResource(bool reuse_cache = false) {
        auto            cache_config = init_config();
        cache_manager_ = std::make_shared<CacheManager>(cache_config, device_);
        ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache = reuse_cache;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_beams = 2;
        auto vec = vector<int>{1, 2, 3, 4, 5, 6};
        std::vector<size_t> shape = {6};
        generate_input->input_ids = std::make_unique<rtp_llm::Buffer>(rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(vec.data()));
        generate_input->generate_config = generate_config;
        rtp_llm::GptInitParameter params;
        params.max_seq_len_ = 2048;
        stream_ = std::make_shared<NormalGenerateStream>(generate_input, params, resource_context, nullptr);
        stream_->setRunning();
    }

    void checkBlockFunc(const vector<vector<int>>& block_vec, int outter_size, int inner_size) {
        ASSERT_EQ(block_vec.size(), outter_size);
        ASSERT_EQ(block_vec[0].size(), inner_size);
    };

protected:
    GenerateStreamPtr stream_;
    CacheManagerPtr cache_manager_;
};

TEST_F(StreamCacheResourceTest, testAllocateResource) {
    prepareResource();

    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    checkBlockFunc(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    checkBlockFunc(blocks.batch_block_id, 2, 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    ASSERT_EQ(blocks.batch_block_id.size(), 0);
}

TEST_F(StreamCacheResourceTest, testFallback) {
    prepareResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    checkBlockFunc(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    resource.tryReleaseKVBlock(resource.maxBlockSize());
    stream_->setPaused();
    checkBlockFunc(blocks.batch_block_id, 2, 0);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
}

TEST_F(StreamCacheResourceTest, testFallbackWithFastGen) {
    prepareResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
    auto& resource = stream_->streamCacheResource();
    stream_->enable_fast_gen_ = true;

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    checkBlockFunc(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    // first fallback
    resource.tryReleaseKVBlock(resource.maxBlockSize());
    stream_->setPaused();
    checkBlockFunc(blocks.batch_block_id, 2, 0);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    // first chunk
    token_capacity = 4;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);

    // second chunk
    token_capacity = 4;
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    // fallback again
    resource.tryReleaseKVBlock(resource.maxBlockSize());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    // first chunk again
    token_capacity = 4;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);
}

TEST_F(StreamCacheResourceTest, testPartialFallback) {
    prepareResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    checkBlockFunc(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    resource.tryReleaseKVBlock(2);
    stream_->setPaused();
    checkBlockFunc(blocks.batch_block_id, 2, 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);

    token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
    ASSERT_EQ(stream_->fallbackPrefixLength(), 4);
    stream_->step();
    ASSERT_EQ(stream_->fallbackPrefixLength(), 0);
}

TEST_F(StreamCacheResourceTest, testPartialFallbackWithFastGen) {
    prepareResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
    stream_->enable_fast_gen_ = true;
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheResource& blocks = resource.kvCache();
    checkBlockFunc(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    // first fallback
    resource.tryReleaseKVBlock(2);
    stream_->setPaused();
    ASSERT_EQ(stream_->fallbackPrefixLength(), 4);
    checkBlockFunc(blocks.batch_block_id, 2, 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);
    stream_->step();
    ASSERT_EQ(stream_->fallbackPrefixLength(), 0);

    // first chunk
    token_capacity = 2;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 3);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);

    // second chunk
    token_capacity = 2;
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    // second fallback
    resource.tryReleaseKVBlock(2);
    checkBlockFunc(blocks.batch_block_id, 2, 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);

    // first chunk again
    token_capacity = 2;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    checkBlockFunc(blocks.batch_block_id, 2, 3);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(stream_->fallbackPrefixLength(), 4);
}

TEST_F(StreamCacheResourceTest, testAllocateResourceWithFastGen) {
    prepareResource();

    stream_->enable_fast_gen_ = true;
    auto& resource = stream_->streamCacheResource();

    // first chunk
    int token_capacity = 4;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);
    ASSERT_EQ(resource.maxBlockSize(), 2);
    const BatchKVCacheResource& blocks = resource.kvCache();
    checkBlockFunc(blocks.batch_block_id, 2, 2);

    // second chunk
    token_capacity = 4;
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    checkBlockFunc(blocks.batch_block_id, 2, 3);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    checkBlockFunc(blocks.batch_block_id, 2, 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    ASSERT_EQ(blocks.batch_block_id.size(), 0);
}

TEST_F(StreamCacheResourceTest, testReuseCache) {
    prepareResource(true);
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    stream_->setSeqLength(9);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    auto batch_tokens_1 = stream_->complete_token_ids_->data(0);
    batch_tokens_1[stream_->seqLength() - 3] = 7;
    batch_tokens_1[stream_->seqLength() - 2] = 8;
    auto batch_tokens_2 = stream_->complete_token_ids_->data(1);
    batch_tokens_2[stream_->seqLength() - 3] = 9;
    batch_tokens_2[stream_->seqLength() - 2] = 10;

    // stream_->reConstructCacheKeys();
    stream_->releaseResource();

    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);
    ASSERT_TRUE(cache_manager_->blockCache().hasKey({1, 2, 3, 4, 5, 6, 7, 8}));
    ASSERT_TRUE(cache_manager_->blockCache().hasKey({1, 2, 3, 4, 5, 6, 9, 10}));
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(6), 1);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(7), 0);

    // test another stream
    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    generate_config->num_beams = 2;
    auto vec = vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 11};
    std::vector<size_t> shape = {9};
    generate_input->input_ids = std::make_unique<rtp_llm::Buffer>(rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(vec.data()));
    generate_input->generate_config = generate_config;

    ResourceContext resource_context;
    resource_context.reuse_cache = true;
    resource_context.cache_manager = cache_manager_;
    rtp_llm::GptInitParameter params;
    params.max_seq_len_ = 2048;
    stream_ = std::make_shared<NormalGenerateStream>(generate_input, params, resource_context, nullptr);
    stream_->setRunning();

    auto& resource2 = stream_->streamCacheResource();
    token_capacity = 1000;
    ASSERT_TRUE(resource2.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    stream_->setIsContextStream(false);
    stream_->setSeqLength(10);
    ASSERT_TRUE(resource2.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    stream_->setSeqLength(11);
    ASSERT_TRUE(resource2.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 0);

    auto tokens_1 = stream_->complete_token_ids_->data(0);
    tokens_1[stream_->seqLength() - 2] = 12;
    tokens_1[stream_->seqLength() - 1] = 13;
    auto tokens_2 = stream_->complete_token_ids_->data(1);
    tokens_2[stream_->seqLength() - 2] = 14;
    tokens_2[stream_->seqLength() - 1] = 15;

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 2);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 3);
}

TEST_F(StreamCacheResourceTest, testReuseCacheWithFastGen) {
    prepareResource(true);
    stream_->enable_fast_gen_ = true;
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);

    ASSERT_EQ(cache_manager_->cacheItemNum(), 1);
    ASSERT_TRUE(cache_manager_->blockCache().hasKey({1, 2, 3, 4, 5, 6}));

    // test another stream
    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    generate_config->num_beams = 2;
    auto vec = vector<int>{1, 2, 30, 40, 50, 60, 90};
    std::vector<size_t> shape = {7};
    generate_input->input_ids = std::make_unique<rtp_llm::Buffer>(rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(vec.data()));
    generate_input->generate_config = generate_config;

    ResourceContext resource_context;
    resource_context.reuse_cache = true;
    resource_context.cache_manager = cache_manager_;
    rtp_llm::GptInitParameter params;
    params.max_seq_len_ = 2048;
    stream_ = std::make_shared<NormalGenerateStream>(generate_input, params, resource_context, nullptr);
    stream_->setRunning();
    stream_->enable_fast_gen_ = true;

    auto& resource2 = stream_->streamCacheResource();
    // first chunk
    token_capacity = 2;
    ASSERT_TRUE(resource2.initKVBlock(token_capacity).ok());
    ASSERT_EQ(stream_->reuseLength(), 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 4);

    // second chunk
    token_capacity = 2;
    ASSERT_TRUE(resource2.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    // third chunk
    token_capacity = 2;
    ASSERT_TRUE(resource2.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    stream_->setIsContextStream(false);
    stream_->setSeqLength(8);
    ASSERT_TRUE(resource2.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(stream_->maxBlockSize(), 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    // partial fallback
    stream_->tryReleaseKVBlock(2);
    stream_->setPaused();
    ASSERT_EQ(stream_->maxBlockSize(), 2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 4);

    // first chunk again
    token_capacity = 2;
    ASSERT_TRUE(resource2.initKVBlock(token_capacity).ok());
    ASSERT_EQ(stream_->reuseLength(), 2);
    ASSERT_EQ(stream_->prefixLength(), 4);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
    ASSERT_EQ(stream_->maxBlockSize(), 3);

    // full fallback
    stream_->tryReleaseKVBlock(stream_->maxBlockSize());
    stream_->setPaused();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);
    ASSERT_EQ(cache_manager_->availableBlockNums(), 8);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);
    ASSERT_EQ(stream_->maxBlockSize(), 0);

    // first chunk again
    token_capacity = 2;
    ASSERT_TRUE(resource2.initKVBlock(token_capacity).ok());
    ASSERT_EQ(stream_->reuseLength(), 6);
    ASSERT_EQ(stream_->prefixLength(), 6);
    ASSERT_EQ(stream_->currentChunkLen(), 8);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);
    ASSERT_EQ(stream_->maxBlockSize(), 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->availableBlockNums(), 8);
}

TEST_F(StreamCacheResourceTest, testTryReleaseKVBlock) {
    prepareResource(false);
    auto& resource = stream_->streamCacheResource();

    int token_capacity = 1000;
    ASSERT_TRUE(resource.initKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(3), 2);

    resource.tryReleaseKVBlock(1);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 6);
    ASSERT_EQ(resource.maxBlockSize(), 2);

    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(3), 0);

    resource.tryReleaseKVBlock(2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
    ASSERT_EQ(resource.maxBlockSize(), 0);

    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(1), 0);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(3), 0);

    ASSERT_TRUE(resource.incrKVBlock(token_capacity).ok());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);

    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager_->blockRefCounter().getRefCounter(6), 0);

    resource.tryReleaseKVBlock(2);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 7);
    ASSERT_EQ(resource.maxBlockSize(), 1);

    resource.resource_context_.reuse_cache = true;
    auto tokens_1 = stream_->complete_token_ids_->data(0);
    tokens_1[0] = 1;
    auto tokens_2 = stream_->complete_token_ids_->data(1);
    tokens_2[0] = 2;

    resource.tryReleaseKVBlock(1);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 7);
    ASSERT_EQ(resource.maxBlockSize(), 0);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);

    resource.tryReleaseKVBlock(1);
    ASSERT_EQ(cache_manager_->freeBlockNums(), 7);
    ASSERT_EQ(resource.maxBlockSize(), 0);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 2);
}

}  // namespace rtp_llm
