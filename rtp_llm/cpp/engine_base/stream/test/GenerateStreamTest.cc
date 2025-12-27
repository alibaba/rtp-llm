
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder(): device_(rtp_llm::DeviceFactory::getDefaultDevice()) {
        model_config_.max_seq_len = 2048;
    }

    CacheConfig init_config() {
        CacheConfig config;
        config.layer_num          = 3;
        config.block_num          = 9;
        config.seq_size_per_block = 2;  // tokens_per_block

        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->layer_num          = 3;
        spec->local_head_num_kv  = 1;
        spec->size_per_head      = 1;
        spec->seq_size_per_block = 2;
        spec->dtype              = rtp_llm::DataType::TYPE_INT8;
        spec->type               = KVCacheType::MultiHeadAttention;
        config.cache_specs.push_back(spec);

        std::vector<int> layer_ids(3);
        for (int i = 0; i < 3; ++i) {
            layer_ids[i] = i;
        }
        config.layer_ids.push_back(layer_ids);
        return config;
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids       = rtp_llm::vector2Buffer(input_ids);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids             = rtp_llm::vector2Buffer(input_ids);
        generate_input->generate_config       = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        auto stream              = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids       = rtp_llm::vector2Buffer(input_ids);
        auto stream_ptr                 = std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto new_tokens_ptr = rtp_llm::vector2Buffer(new_token_ids);
        device_->copy(
            {*(stream_ptr->completeTokenIds()->index(0)->slice(stream_ptr->seqLength(), new_token_ids.size())),
             *new_tokens_ptr});
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    ModelConfig          model_config_;
    RuntimeConfig        runtime_config_;
    rtp_llm::DeviceBase* device_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testGenerateStreamReuseCacheMethod) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // default true
    ASSERT_TRUE(stream->reuseCache());

    // flip to false and verify
    stream->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(stream->reuseCache());

    // flip back to true and verify
    stream->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(stream->reuseCache());
}

TEST_F(GenerateStreamTest, testGenerateStreamEnable3FSMethod) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // default true
    ASSERT_TRUE(stream->enable3FS());

    // flip to false and verify
    stream->generate_input_->generate_config->enable_3fs = false;
    ASSERT_FALSE(stream->enable3FS());

    // flip back to true and verify
    stream->generate_input_->generate_config->enable_3fs = true;
    ASSERT_TRUE(stream->enable3FS());
}

TEST_F(GenerateStreamTest, testAsyncLoadCache_ReturnFalse_WhenStopped) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});

    stream->generate_status_->status = StreamState::STOPPED;
    ASSERT_FALSE(stream->asyncLoadCache());
    ASSERT_FALSE(stream->loadingCache());
}

TEST_F(GenerateStreamTest, testAsyncLoadCache_ReturnFalse_WhenStreamCacheResourceAsyncLoadCacheFalse) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});

    // Default memory block cache gate is disabled, so StreamCacheResource::asyncLoadCache() returns false.
    ASSERT_FALSE(stream->asyncLoadCache());
    ASSERT_FALSE(stream->loadingCache());
}

TEST_F(GenerateStreamTest, testAsyncLoadCache_ReturnTrue_AndSetsLoadingCache_WhenUnderlyingAsyncLoadCacheTrue) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createComplexContextStream({1, 2, 3, 4});

    // Enable StreamCacheResource::enableMemoryBlockCache() gate
    stream->stream_cache_resource_->resource_context_.enable_memory_block_cache = true;
    stream->generate_input_->generate_config->enable_memory_block_cache         = true;

    // Inject a mock coordinator into KVCacheManager so StreamCacheResource::asyncLoadCache can succeed.
    auto cache_manager = stream->stream_cache_resource_->resource_context_.cache_manager;
    ASSERT_NE(cache_manager, nullptr);
    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager->config_,
                                                                             cache_manager->kv_cache_config_,
                                                                             cache_manager->runtime_config_,
                                                                             cache_manager->allocator_,
                                                                             stream->device_);
    cache_manager->connector_coordinator_ = mock_coord;

    auto async_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_coord, asyncRead(testing::_, testing::_)).WillOnce(testing::Return(async_ctx));

    ASSERT_TRUE(stream->asyncLoadCache());
    ASSERT_TRUE(stream->loadingCache());
}

TEST_F(GenerateStreamTest, testLoadCacheDone_ReturnFalse_WhenLoadCacheContextNotDone) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});

    auto not_done = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*not_done, done()).WillByDefault(testing::Return(false));
    stream->stream_cache_resource_->load_cache_context_ = not_done;

    ASSERT_FALSE(stream->loadCacheDone());
}

TEST_F(GenerateStreamTest, testLoadCacheDone_ReturnTrue_StreamCacheResourceReturnTrue) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});

    // Case 1: no context => true
    stream->stream_cache_resource_->load_cache_context_.reset();
    ASSERT_TRUE(stream->loadCacheDone());

    // Case 2: context done (success=false) => returns true and clears context
    auto done_fail = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*done_fail, done()).WillByDefault(testing::Return(true));
    ON_CALL(*done_fail, success()).WillByDefault(testing::Return(false));
    stream->stream_cache_resource_->load_cache_context_ = done_fail;

    ASSERT_TRUE(stream->loadCacheDone());
    ASSERT_EQ(stream->stream_cache_resource_->load_cache_context_, nullptr);
}

TEST_F(GenerateStreamTest, testAsyncStoreCache_ReturnFalse_StreamCacheResourceReturnFalse) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});

    // Default memory block cache gate is disabled, so StreamCacheResource::asyncStoreCache() returns false.
    stream->stream_cache_resource_->store_cache_context_.reset();
    ASSERT_FALSE(stream->asyncStoreCache());
    ASSERT_EQ(stream->stream_cache_resource_->store_cache_context_, nullptr);
}

TEST_F(GenerateStreamTest, testAsyncStoreCache_ReturnTrue_WhenUnderlyingAsyncStoreCacheTrue) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createComplexContextStream({1, 2, 3, 4});

    // Enable StreamCacheResource::enableMemoryBlockCache() gate
    stream->stream_cache_resource_->resource_context_.enable_memory_block_cache = true;
    stream->generate_input_->generate_config->enable_memory_block_cache         = true;

    auto cache_manager = stream->stream_cache_resource_->resource_context_.cache_manager;
    ASSERT_NE(cache_manager, nullptr);
    auto mock_coord =
        std::make_shared<testing::NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager->config_,
                                                                             cache_manager->kv_cache_config_,
                                                                             cache_manager->runtime_config_,
                                                                             cache_manager->allocator_,
                                                                             stream->device_);
    cache_manager->connector_coordinator_ = mock_coord;

    auto async_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    EXPECT_CALL(*mock_coord, asyncWrite(testing::_, testing::_)).WillOnce(testing::Return(async_ctx));

    ASSERT_TRUE(stream->asyncStoreCache());
    ASSERT_EQ(stream->stream_cache_resource_->store_cache_context_.get(), async_ctx.get());
}

}  // namespace rtp_llm
