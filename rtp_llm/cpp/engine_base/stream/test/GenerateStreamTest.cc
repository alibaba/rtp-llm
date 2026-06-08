
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
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
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto complete_ids = stream_ptr->completeTokenIds();
        std::memcpy(complete_ids.data_ptr<int32_t>() + stream_ptr->seqLength(),
                    new_token_ids.data(),
                    new_token_ids.size() * sizeof(int));
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testBatchSizeWithNumReturnSequences) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input                   = std::make_shared<GenerateInput>();
    auto generate_config                  = std::make_shared<GenerateConfig>();
    generate_config->num_return_sequences = 3;
    generate_input->generate_config       = generate_config;
    generate_input->input_ids             = torch::tensor({1, 2, 3}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    // prefill: batch=1 even though n=3
    EXPECT_EQ(1, stream->batchSize(0));
    // decode: batch=n=3
    EXPECT_EQ(3, stream->batchSize(1));
    EXPECT_EQ(3, stream->batchSize(5));
    EXPECT_EQ(3, stream->maxBatchSize());
}

TEST_F(GenerateStreamTest, testBatchSizeWithBeamSearch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input             = std::make_shared<GenerateInput>();
    auto generate_config            = std::make_shared<GenerateConfig>();
    generate_config->num_beams      = 4;
    generate_input->generate_config = generate_config;
    generate_input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    // beam search: first step 1 beam, subsequent steps 4 beams
    EXPECT_EQ(1, stream->batchSize(0));
    EXPECT_EQ(4, stream->batchSize(1));
    EXPECT_EQ(4, stream->maxBatchSize());
}

TEST_F(GenerateStreamTest, testNeedTilingForSampling) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    // Case 1: context stream with num_return_sequences > 1, no beam search -> needs tiling
    {
        auto input                   = std::make_shared<GenerateInput>();
        auto config                  = std::make_shared<GenerateConfig>();
        config->num_return_sequences = 2;
        input->generate_config       = config;
        input->input_ids             = torch::tensor({1, 2}, torch::kInt32);
        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        // context stream: currentBatchSize()=1, nextBatchSize()=2, no beams
        EXPECT_TRUE(stream->isContextStream());
        EXPECT_TRUE(stream->needTilingForSampling());
    }

    // Case 2: context stream with num_beams > 1 -> no tiling (beam search handles expansion)
    {
        auto input             = std::make_shared<GenerateInput>();
        auto config            = std::make_shared<GenerateConfig>();
        config->num_beams      = 2;
        input->generate_config = config;
        input->input_ids       = torch::tensor({1, 2}, torch::kInt32);
        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        EXPECT_TRUE(stream->isContextStream());
        EXPECT_FALSE(stream->needTilingForSampling());
    }

    // Case 3: decoder stream with num_return_sequences > 1 -> no tiling (not context stream)
    {
        auto input                   = std::make_shared<GenerateInput>();
        auto config                  = std::make_shared<GenerateConfig>();
        config->num_return_sequences = 2;
        input->generate_config       = config;
        input->input_ids             = torch::tensor({1, 2}, torch::kInt32);
        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        EXPECT_FALSE(stream->needTilingForSampling());
    }
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

}  // namespace rtp_llm
