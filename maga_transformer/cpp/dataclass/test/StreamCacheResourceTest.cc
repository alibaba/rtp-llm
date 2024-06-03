
#include "gtest/gtest.h"

#define private public
#define protected public
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/testing/TestBase.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

class StreamCacheResourceTest: public DeviceTestBase {
protected:
    CacheConfig init_config() {
        CacheConfig config(3, 9, 1, 1, 2, TYPE_INT8);
        return config;
    }

    void prepareResource() {
        auto            cache_config = init_config();
        ft::DeviceBase* device;
        cache_manager_ = std::make_shared<CacheManager>(cache_config, device_);
        ASSERT_EQ(cache_manager_->freeBlockNums(), 8);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
             
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_beams = 2;
        auto vec = vector<int>{1, 2, 3, 4, 5, 6};
        std::vector<size_t> shape = {6};
        generate_input->input_ids = std::make_unique<ft::Buffer>(ft::MEMORY_CPU, ft::TYPE_INT32, shape, (void*)(vec.data()));
        generate_input->generate_config = generate_config;
        ft::GptInitParameter params;
        params.max_seq_len_ = 2048;
        stream_ = std::make_shared<GenerateStream>(generate_input, params, resource_context, nullptr);
        stream_->setRunning();
    }

protected:
    GenerateStreamPtr stream_;
    CacheManagerPtr cache_manager_;
};

TEST_F(StreamCacheResourceTest, testAllocateResource) {
    prepareResource();

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.needKVCacheBlockNums(), 3);

    ASSERT_TRUE(resource.initKVBlock());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 5);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheBlockAddr& blocks = resource.kvCache();
    auto CHECK_FUNC = [](const auto& block_vec, auto outter_size, auto inner_size) {
        ASSERT_EQ(block_vec.size(), outter_size);
        ASSERT_EQ(block_vec[0][0].size(), inner_size);
    };
    CHECK_FUNC(blocks.k_ptr, 2, 3);
    CHECK_FUNC(blocks.v_ptr, 2, 3);
    CHECK_FUNC(blocks.k_scale_ptr, 2, 3);
    CHECK_FUNC(blocks.v_scale_ptr, 2, 3);

    printf("here1\n");
    fflush(stdout);

    ASSERT_EQ(resource.needKVCacheBlockNums(), 0);

    stream_->setSeqLength(7);
    stream_->setIsContextStream(false);
    ASSERT_EQ(resource.needKVCacheBlockNums(), 2);
    ASSERT_TRUE(resource.incrKVBlock());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 3);

    CHECK_FUNC(blocks.k_ptr, 2, 4);
    CHECK_FUNC(blocks.v_ptr, 2, 4);
    CHECK_FUNC(blocks.k_scale_ptr, 2, 4);
    CHECK_FUNC(blocks.v_scale_ptr, 2, 4);

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 8);

    ASSERT_EQ(blocks.k_ptr.size(), 0);
    ASSERT_EQ(blocks.v_ptr.size(), 0);
    ASSERT_EQ(blocks.k_scale_ptr.size(), 0);
    ASSERT_EQ(blocks.v_scale_ptr.size(), 0);
}

TEST_F(StreamCacheResourceTest, testReuseCache) {
    prepareResource();
    auto& resource = stream_->streamCacheResource();
    resource.resource_context_.reuse_cache = true;
    ASSERT_EQ(resource.needKVCacheBlockNums(), 3);

    ASSERT_TRUE(resource.initKVBlock());
    stream_->setSeqLength(9);
    stream_->setIsContextStream(false);
    ASSERT_TRUE(resource.incrKVBlock());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 1);

    auto batch_tokens_1 = stream_->complete_token_ids_->view(0, 1).data<int32_t>();
    batch_tokens_1[stream_->seqLength() - 3] = 7;
    batch_tokens_1[stream_->seqLength() - 2] = 8;
    auto batch_tokens_2 = stream_->complete_token_ids_->view(1, 1).data<int32_t>();
    batch_tokens_2[stream_->seqLength() - 3] = 9;
    batch_tokens_2[stream_->seqLength() - 2] = 10;

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
    auto vec = vector<int>{1, 2, 3, 4, 5, 6, 9, 10, 11};
    std::vector<size_t> shape = {9};
    generate_input->input_ids = std::make_unique<ft::Buffer>(ft::MEMORY_CPU, ft::TYPE_INT32, shape, (void*)(vec.data()));
    generate_input->generate_config = generate_config;
    int max_seq_len = 2048;

    ResourceContext resource_context;
    resource_context.reuse_cache = true;
    resource_context.cache_manager = cache_manager_;
    ft::GptInitParameter params;
    params.max_seq_len_ = 2048;
    stream_ = std::make_shared<GenerateStream>(generate_input, params, resource_context, nullptr);
    stream_->setRunning();

    auto& resource2 = stream_->streamCacheResource();
    ASSERT_TRUE(resource2.initKVBlock());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 2);

    stream_->setIsContextStream(false);
    stream_->setSeqLength(10);
    ASSERT_TRUE(resource2.incrKVBlock());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 2);

    stream_->setSeqLength(11);
    ASSERT_TRUE(resource2.incrKVBlock());
    ASSERT_EQ(cache_manager_->freeBlockNums(), 0);

    auto tokens_1 = stream_->complete_token_ids_->view(0, 1).data<int32_t>();
    tokens_1[stream_->seqLength() - 2] = 12;
    tokens_1[stream_->seqLength() - 1] = 13;
    auto tokens_2 = stream_->complete_token_ids_->view(1, 1).data<int32_t>();
    tokens_2[stream_->seqLength() - 2] = 14;
    tokens_2[stream_->seqLength() - 1] = 15;

    stream_->releaseResource();
    ASSERT_EQ(cache_manager_->freeBlockNums(), 2);
    ASSERT_EQ(cache_manager_->cacheItemNum(), 4);
}

TEST_F(StreamCacheResourceTest, testError) {}

}  // namespace rtp_llm
