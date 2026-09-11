
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder(rtp_llm::GptInitParameter params):
        params_(params), device_(rtp_llm::DeviceFactory::getDefaultDevice()) {
        params_.max_seq_len_ = 2048;
    }

    CacheConfig init_config() {
        CacheConfig config(KVCacheParam{KVCacheParam{3, 9, 1, 1, 2, rtp_llm::DataType::TYPE_INT8}});
        return config;
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids       = rtp_llm::vector2Buffer(input_ids);
        return std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto            cache_config  = init_config();
        auto            cache_manager = std::make_shared<CacheManager>(cache_config, device_);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids             = rtp_llm::vector2Buffer(input_ids);
        generate_input->generate_config       = generate_config;
        rtp_llm::GptInitParameter params;
        params.max_seq_len_ = 2048;
        auto stream         = std::make_shared<NormalGenerateStream>(generate_input, params, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids       = rtp_llm::vector2Buffer(input_ids);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto new_tokens_ptr = rtp_llm::vector2Buffer(new_token_ids);
        device_->copy(
            {*(stream_ptr->completeTokenIds()->index(0)->slice(stream_ptr->seqLength(), new_token_ids.size())),
             *new_tokens_ptr});
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    rtp_llm::GptInitParameter params_;
    rtp_llm::DeviceBase*      device_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

TEST_F(GenerateStreamTest, boundedHistoryAcceptsPaddedSamplerBatch) {
    auto input                             = std::make_shared<GenerateInput>();
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 2;
    input->input_ids                       = createBuffer<int32_t>({2}, {11, 12}, AllocationType::HOST);
    CompleteTokenIds history(device_, 1, 4, 8192, 16);
    history.init(input, 3);
    EXPECT_EQ(history.completeTokenIds()->shape()[1], 7);
    std::vector<int32_t> padded(4 * 20, 9);
    for (size_t i = 0; i < 4; ++i) {
        padded[i * 20]     = 11;
        padded[i * 20 + 1] = 12;
        padded[i * 20 + 2] = 20 + i;
    }
    auto tokens = createBuffer<int32_t>({4, 20}, padded, AllocationType::HOST);
    int  error  = 0;
    ASSERT_TRUE(history.update(tokens, 0, 1, 2, 4, 128, true, 42, error));
    EXPECT_EQ(history.batchSize(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(history.completeTokenIdsVec(i), (std::vector<int>{11, 12, 20 + i}));
    }
}

class RejectingLogitsProcessor: public BaseLogitsProcessor {
public:
    explicit RejectingLogitsProcessor(DeviceBase* device): BaseLogitsProcessor(device) {}
    void process(const SamplerInputs&, size_t, size_t) override {}
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const BufferPtr&, int32_t) override {
        throw std::runtime_error("rejected constrained token");
    }
};

TEST_F(GenerateStreamTest, testInvalidGeneratedTokenNeverReachesOutputQueueOrCrashesWorker) {
    for (bool final_step : {false, true}) {
        GptInitParameter params;
        params.max_seq_len_                    = 128;
        params.vocab_size_                     = 64;
        params.special_tokens_.eos_token_id_   = 63;
        auto input                             = std::make_shared<GenerateInput>();
        input->need_release_resource           = false;
        input->input_ids                       = vector2Buffer(std::vector<int>{1, 2});
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = final_step ? 1 : 10;
        auto stream = std::make_shared<NormalGenerateStream>(input, params, ResourceContext{}, nullptr);
        stream->logits_processor_list_.push_back(std::make_shared<RejectingLogitsProcessor>(device_));
        const std::vector<int32_t> token  = {final_step ? 63 : 3};
        auto                       output = createHostBuffer<int32_t>({1, 1}, token.data());
        EXPECT_NO_THROW(stream->update({output, 1}));
        EXPECT_TRUE(stream->stopped());
        EXPECT_FALSE(stream->hasOutput());
    }
}

TEST_F(GenerateStreamTest, testConstruct) {
    rtp_llm::GptInitParameter params;
    auto                      builder = GenerateStreamBuilder(params);
    auto                      stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto                      stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testConstructCacheKey) {
    rtp_llm::GptInitParameter params;
    auto                      builder    = GenerateStreamBuilder(params);
    auto                      stream1    = builder.createComplexContextStream({{1, 2, 3, 4, 5}, {}});
    auto&                     cache_key1 = stream1->cacheKeys(0);
    auto&                     cache_key2 = stream1->cacheKeys(1);
    ASSERT_EQ(cache_key1.size(), 3);
    ASSERT_EQ(cache_key2.size(), 3);
    ASSERT_EQ(cache_key1[0], cache_key2[0]);
    ASSERT_EQ(cache_key1[1], cache_key2[1]);

    stream1->stream_cache_resource_->reConstructCacheKeys();
    ASSERT_EQ(cache_key1.size(), 2);
    ASSERT_EQ(cache_key2.size(), 2);

    stream1->setSeqLength(6);
    auto batch_tokens_1                      = stream1->complete_token_ids_->data(0);
    batch_tokens_1[stream1->seqLength() - 1] = 8;
    auto batch_tokens_2                      = stream1->complete_token_ids_->data(0);
    batch_tokens_2[stream1->seqLength() - 1] = 9;
    stream1->stream_cache_resource_->reConstructCacheKeys();
    ASSERT_EQ(cache_key1.size(), 3);
    ASSERT_EQ(cache_key2.size(), 3);
    ASSERT_NE(cache_key1[2], cache_key2[2]);

    stream1->setSeqLength(7);
    stream1->stream_cache_resource_->reConstructCacheKeys();
    ASSERT_EQ(cache_key1.size(), 3);
    ASSERT_EQ(cache_key2.size(), 3);
    ASSERT_NE(cache_key1[2], cache_key2[2]);
}

TEST_F(GenerateStreamTest, testGenerateStreamReuseCacheMethod) {
    rtp_llm::GptInitParameter params;
    auto                      builder = GenerateStreamBuilder(params);
    auto                      stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

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
    rtp_llm::GptInitParameter params;
    auto                      builder = GenerateStreamBuilder(params);
    auto                      stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // default true
    ASSERT_TRUE(stream->enable3FS());

    // flip to false and verify
    stream->generate_input_->generate_config->enable_3fs = false;
    ASSERT_FALSE(stream->enable3FS());

    // flip back to true and verify
    stream->generate_input_->generate_config->enable_3fs = true;
    ASSERT_TRUE(stream->enable3FS());
}

}  // namespace rtp_llm
