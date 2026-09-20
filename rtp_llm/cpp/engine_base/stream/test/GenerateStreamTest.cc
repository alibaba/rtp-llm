
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

    GenerateStreamPtr createContextStream(std::vector<int>                        input_ids,
                                          std::shared_ptr<const V41RequestInputs> v41_inputs = nullptr) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->v41_inputs      = std::move(v41_inputs);
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

TEST_F(GenerateStreamTest, testImageReuseBoundariesFollowTokenView) {
    auto          images = std::make_shared<V41RequestInputs>();
    V41ImageInput first;
    first.start = 3;
    first.types = torch::zeros({4}, torch::kInt32);
    V41ImageInput second;
    second.start   = 7;
    second.types   = torch::zeros({3}, torch::kInt32);
    images->images = {first, second};
    auto builder   = GenerateStreamBuilder();
    auto stream    = builder.createContextStream(std::vector<int>(12, 1), images);
    auto tokens    = stream->completeTokenIdsPtr();
    for (int boundary : {0, 3, 7, 10, 12}) {
        EXPECT_TRUE(tokens->isValidReuseLength(boundary)) << boundary;
    }
    for (int boundary : {4, 6, 8, 9}) {
        EXPECT_FALSE(tokens->isValidReuseLength(boundary)) << boundary;
    }
    CompleteTokenIds shifted(*tokens, /*share=*/true, /*shift_token_num=*/2);
    CompleteTokenIds copied(shifted);
    for (const auto* view : {&shifted, &copied}) {
        EXPECT_TRUE(view->isValidReuseLength(1));
        EXPECT_FALSE(view->isValidReuseLength(2));
        EXPECT_TRUE(view->isValidReuseLength(5));
        EXPECT_FALSE(view->isValidReuseLength(6));
        EXPECT_TRUE(view->isValidReuseLength(8));
    }
    auto text_stream = builder.createContextStream(std::vector<int>(12, 1));
    EXPECT_TRUE(text_stream->completeTokenIdsPtr()->isValidReuseLength(4));
}

TEST_F(GenerateStreamTest, testSetReuseLengthDoesNotRewindRestoredCache) {
    auto          images = std::make_shared<V41RequestInputs>();
    V41ImageInput image;
    image.start = 730;
    image.types = torch::zeros({994}, torch::kInt32);
    images->images.push_back(image);
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream(std::vector<int>(2048, 1), images);
    // Boundary legality belongs to snapshot selection. The stream must preserve
    // the supplied execution position and keep all reuse counters consistent.
    for (int boundary : {512, 1536, 1792}) {
        stream->setReuseLength(boundary);
        stream->setInitialReuseLength(boundary);
        stream->setLocalReuseLength(boundary);
        EXPECT_EQ(stream->reuseLength(), boundary);
        EXPECT_EQ(stream->initialReuseLength(), boundary);
        EXPECT_EQ(stream->localReuseLength(), boundary);
        EXPECT_EQ(stream->deviceReuseLength(), boundary);
    }
}

TEST_F(GenerateStreamTest, testInitialReuseLengthMustBeLessThanSeqLength) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    stream->setInitialReuseLength(stream->seqLength() - 1);
    ASSERT_EQ(stream->initialReuseLength(), stream->seqLength() - 1);

    EXPECT_THROW(stream->setInitialReuseLength(stream->seqLength()), RTPException);
}

TEST_F(GenerateStreamTest, testSyncSpeculativeMaxLengthDoesNotCountAnchorAsNewToken) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    auto            builder = GenerateStreamBuilder();
    auto            stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    auto sp_output_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer->propose_step = 3;
    stream->setSPOutputBuffer(sp_output_buffer);
    // Scheduler/cache reservation includes the target-verify anchor, but the
    // output-length limit must reserve only the three newly proposed tokens.
    stream->setReserveStep(4);

    EXPECT_EQ(stream->maxTokenNum(), 2045);
}

// clearMtpAsyncDeviceState rejects stale epochs. A worker that
// captured epoch N must not clear state that step N+1 already published
// under epoch N+1.
TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateStaleEpochReject) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // Start: epoch counter is 0, state is default-constructed.
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, 0u);
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 1: publish state, capture epoch_1.
    GenerateStream::MtpAsyncDeviceState s1;
    s1.accept_len_gpu      = torch::ones({1}, torch::kInt32);
    s1.engram_token_window_gpu = torch::tensor({{6, 5, 4, 3}}, torch::kInt32);
    const uint64_t epoch_1 = stream->setMtpAsyncDeviceState(std::move(s1));
    ASSERT_EQ(epoch_1, 1u);
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 2: another publish before the worker for epoch_1 ran. Counter
    // bumps; old epoch should now be stale.
    GenerateStream::MtpAsyncDeviceState s2;
    s2.accept_len_gpu      = torch::ones({1}, torch::kInt32) * 2;
    s2.engram_token_window_gpu = torch::tensor({{8, 7, 6, 5}}, torch::kInt32);
    const uint64_t epoch_2 = stream->setMtpAsyncDeviceState(std::move(s2));
    ASSERT_EQ(epoch_2, 2u);
    ASSERT_NE(epoch_1, epoch_2);

    // Stale worker for epoch_1 attempts to clear: must be rejected, state
    // for epoch_2 must remain intact.
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_1));
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, epoch_2);
    EXPECT_TRUE(torch::equal(stream->getEngramTokenWindowGpu(), torch::tensor({{8, 7, 6, 5}}, torch::kInt32)));

    // Worker for epoch_2 clears successfully.
    ASSERT_TRUE(stream->clearMtpAsyncDeviceState(epoch_2));
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_FALSE(stream->getEngramTokenWindowGpu().defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, 0u);

    // Repeated stale clear after the live state is gone is also a no-op
    // (epoch 0 != epoch_2 since state was reset to default).
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_2));
}

TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateTracksRealAndUpperBoundSeqLen) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    GenerateStream::MtpAsyncDeviceState state;
    state.last_real_seq_len = stream->seqLength();
    state.next_real_seq_len = state.last_real_seq_len + 2;
    stream->setMtpAsyncDeviceState(std::move(state));

    ASSERT_EQ(stream->getMtpAsyncDeviceState().last_real_seq_len, stream->seqLength());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().next_real_seq_len, stream->seqLength() + 2);
}

// setSpecDecodeDeviceState / clearSpecDecodeDeviceState
// continue to work as wrappers around the new struct API.
TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateBackCompatWrappers) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    auto accept_len     = torch::ones({1}, torch::kInt32);
    auto accept_tokens  = torch::ones({1, 2}, torch::kInt32);
    auto next_seq_len   = torch::ones({1}, torch::kInt32) * 7;
    auto propose_tokens = torch::ones({1, 4}, torch::kInt32);

    stream->setSpecDecodeDeviceState(accept_len, accept_tokens, next_seq_len, propose_tokens);
    ASSERT_TRUE(stream->getAcceptLenGpu().defined());
    ASSERT_TRUE(stream->getAcceptTokensGpu().defined());
    ASSERT_TRUE(stream->getNextSeqLenGpu().defined());
    ASSERT_TRUE(stream->getProposeTokensGpu().defined());

    stream->clearSpecDecodeDeviceState();
    ASSERT_FALSE(stream->getAcceptLenGpu().defined());
    ASSERT_FALSE(stream->getAcceptTokensGpu().defined());
    ASSERT_FALSE(stream->getNextSeqLenGpu().defined());
    ASSERT_FALSE(stream->getProposeTokensGpu().defined());
}

}  // namespace rtp_llm
