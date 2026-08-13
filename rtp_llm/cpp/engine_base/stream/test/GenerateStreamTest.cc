
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <future>

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

TEST_F(GenerateStreamTest, testNextOutputConsumesCancellationWhileWaitingForFirstToken) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});

    std::atomic<bool> cancelled{false};
    std::promise<void> predicate_seen;
    auto               predicate_seen_future = predicate_seen.get_future();
    std::promise<void> release_predicate;
    auto               release_predicate_future = release_predicate.get_future().share();
    std::once_flag     predicate_seen_once;
    auto               wait_result = std::async(std::launch::async, [&]() {
        return stream->nextOutput([&]() {
            const bool observed_cancel = cancelled.load(std::memory_order_acquire);
            std::call_once(predicate_seen_once, [&]() { predicate_seen.set_value(); });
            release_predicate_future.wait();
            return observed_cancel;
        });
    });

    const bool entered_wait = predicate_seen_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    cancelled.store(true, std::memory_order_release);
    release_predicate.set_value();
    if (!entered_wait) {
        stream->reportError(ErrorCode::CANCELLED, "test cleanup before wait");
    }

    const auto ready = wait_result.wait_for(std::chrono::milliseconds(200));
    if (ready != std::future_status::ready) {
        stream->reportError(ErrorCode::CANCELLED, "test cleanup");
        EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(1500)), std::future_status::ready);
    }
    EXPECT_TRUE(entered_wait);
    ASSERT_EQ(ready, std::future_status::ready);

    const auto result = wait_result.get();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamTest, testNextOutputDoesNotLoseOutputNotificationBeforeWait) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3, 4}));

    std::promise<void> predicate_seen;
    auto               predicate_seen_future = predicate_seen.get_future();
    std::promise<void> release_predicate;
    auto               release_predicate_future = release_predicate.get_future().share();
    std::once_flag     predicate_seen_once;
    auto               wait_result = std::async(std::launch::async, [&]() {
        return stream->nextOutput([&]() {
            std::call_once(predicate_seen_once, [&]() { predicate_seen.set_value(); });
            release_predicate_future.wait();
            return false;
        });
    });

    const bool before_wait = predicate_seen_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    GenerateOutputs output;
    output.request_id = 17;
    stream->generate_outputs_queue_.push(output);
    release_predicate.set_value();

    const auto ready = wait_result.wait_for(std::chrono::milliseconds(200));
    if (ready != std::future_status::ready) {
        stream->reportError(ErrorCode::CANCELLED, "test cleanup");
        EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(1500)), std::future_status::ready);
    }
    EXPECT_TRUE(before_wait);
    ASSERT_EQ(ready, std::future_status::ready);
    ASSERT_TRUE(wait_result.get().ok());
}

TEST_F(GenerateStreamTest, testNextOutputPreservesExistingErrorOverCancellation) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4});
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "request timed out");

    const auto result = stream->nextOutput([]() { return true; });

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
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

TEST_F(GenerateStreamTest, testSpecBurstDoesNotMatchEosBeforeMinNewTokens) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9}, {0, 2, 3});
    stream->generate_input_->generate_config->min_new_tokens = 3;

    ASSERT_FALSE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 4);
}

TEST_F(GenerateStreamTest, testSpecBurstMatchesEosAtMinNewTokens) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9}, {1, 2, 0});
    stream->generate_input_->generate_config->min_new_tokens = 3;

    ASSERT_TRUE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 4);
}

TEST_F(GenerateStreamTest, testSpecBurstStopWordUsesEndPositionForMinNewTokens) {
    auto builder = GenerateStreamBuilder();

    auto early_stop = builder.createDecoderStream({9}, {1, 2, 3});
    early_stop->generate_input_->generate_config->min_new_tokens = 3;
    early_stop->generate_input_->generate_config->stop_words_list = {{1}};
    ASSERT_FALSE(early_stop->needFinish());
    ASSERT_EQ(early_stop->seqLength(), 4);

    auto boundary_stop = builder.createDecoderStream({9}, {1, 2, 3});
    boundary_stop->generate_input_->generate_config->min_new_tokens = 3;
    boundary_stop->generate_input_->generate_config->stop_words_list = {{2, 3}};
    ASSERT_TRUE(boundary_stop->needFinish());
    ASSERT_EQ(boundary_stop->seqLength(), 4);
}

TEST_F(GenerateStreamTest, testEmptyAndOverlongStopWordsDoNotMatch) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9, 10}, {11});
    stream->generate_input_->generate_config->stop_words_list = {{}, {1, 2, 3, 4}};

    ASSERT_FALSE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 3);
}

TEST_F(GenerateStreamTest, testSpecBurstScansEosBeforeMaxLengthFinish) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9}, {1, 0, 2});
    stream->generate_input_->generate_config->min_new_tokens = 2;
    stream->generate_input_->generate_config->max_new_tokens = 3;

    ASSERT_TRUE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 3);
}

TEST_F(GenerateStreamTest, testSpecBurstScansStopWordBeforeMaxLengthFinish) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9}, {1, 2, 3});
    stream->generate_input_->generate_config->max_new_tokens  = 3;
    stream->generate_input_->generate_config->stop_words_list = {{2}};

    ASSERT_TRUE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 3);
}

TEST_F(GenerateStreamTest, testSpecBurstChoosesEarliestStopAcrossPatterns) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9}, {1, 2, 3});
    stream->generate_input_->generate_config->stop_words_list = {{3}, {1}};

    ASSERT_TRUE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 2);
}

TEST_F(GenerateStreamTest, testIgnoreEosSkipsEosAndSingletonEosStop) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createDecoderStream({9}, {0});
    stream->generate_input_->generate_config->ignore_eos      = true;
    stream->generate_input_->generate_config->stop_words_list = {{0}};

    ASSERT_FALSE(stream->needFinish());
    ASSERT_EQ(stream->seqLength(), 2);
}

}  // namespace rtp_llm
