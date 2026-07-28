
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

#include <chrono>
#include <future>
#include <mutex>
#include <vector>

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
        model_config_.vocab_size  = 1024;
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
        generate_input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
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
        generate_input->begin_time_us         = autil::TimeUtility::currentTimeInMicroSeconds();
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        model_config.vocab_size  = 1024;
        auto stream              = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
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

template<typename T>
void waitForConsumer(std::future<T>& future, const std::shared_ptr<NormalGenerateStream>& stream) {
    const auto status = future.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test consumer timed out");
    }
    EXPECT_EQ(status, std::future_status::ready);
    future.wait();
}

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

TEST_F(GenerateStreamTest, zeroWaitTimeoutBlocksUntilOutputIsPublished) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    std::promise<void> consumer_started;
    auto               consumer_ready = consumer_started.get_future();
    auto               consumer       = std::async(std::launch::async, [stream, &consumer_started] {
        consumer_started.set_value();
        return stream->nextOutput(0);
    });
    consumer_ready.get();

    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        GenerateOutputs             outputs;
        outputs.request_id = 123;
        stream->enqueueGenerateOutput(std::move(outputs));
    }

    waitForConsumer(consumer, stream);
    auto result = consumer.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().request_id, 123);
}

TEST_F(GenerateStreamTest, outputPublishedBeforeConsumerWaitIsObserved) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    GenerateOutputs outputs;
    outputs.request_id = 321;
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(std::move(outputs));
    }

    auto result = stream->nextOutput(1);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().request_id, 321);
}

TEST_F(GenerateStreamTest, pendingCompletionIsConsumerVisibleBeforeSchedulerCommit) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createComplexContextStream({1, 2, 3}));
    stream->setNeedReleaseResource(true);
    stream->generate_status_->status.store(StreamState::RUNNING);

    std::promise<void> consumer_started;
    auto               consumer_ready = consumer_started.get_future();
    auto               consumer       = std::async(std::launch::async, [stream, &consumer_started] {
        consumer_started.set_value();
        return stream->nextOutput();
    });
    consumer_ready.get();

    stream->reportEvent(StreamEvents::GenerateDone);
    waitForConsumer(consumer, stream);
    auto finished_result = consumer.get();
    ASSERT_FALSE(finished_result.ok());
    EXPECT_EQ(finished_result.status().code(), ErrorCode::FINISHED);

    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());

    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
}

TEST_F(GenerateStreamTest, nextOutputDrainsFinalOutputBeforeCompletion) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));
    stream->generate_status_->status.store(StreamState::RUNNING);

    GenerateOutputs outputs;
    outputs.request_id = 456;
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(std::move(outputs));
        stream->reportEventWithoutLock(StreamEvents::GenerateDone);
    }

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    EXPECT_EQ(output_result.value().request_id, 456);

    auto finished_result = stream->nextOutput();
    ASSERT_FALSE(finished_result.ok());
    EXPECT_EQ(finished_result.status().code(), ErrorCode::FINISHED);

    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
}

TEST_F(GenerateStreamTest, consumerWaitWakesOnError) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    std::promise<void> consumer_started;
    auto               consumer_ready = consumer_started.get_future();
    auto               consumer       = std::async(std::launch::async, [stream, &consumer_started] {
        consumer_started.set_value();
        return stream->nextOutput();
    });
    consumer_ready.get();
    stream->reportError(ErrorCode::CANCELLED, "cancelled");

    waitForConsumer(consumer, stream);
    auto result = consumer.get();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamTest, errorTakesPrecedenceOverQueuedOutput) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    GenerateOutputs outputs;
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(std::move(outputs));
        stream->reportEventWithoutLock(StreamEvents::Error, ErrorCode::CANCELLED, "cancelled");
    }

    auto result = stream->nextOutput();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(stream->hasOutput());
}

TEST_F(GenerateStreamTest, outputQueueCapacityReportsFullWithoutDeadlock) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        for (size_t output = 0; output < stream->kOutputCapacity; ++output) {
            GenerateOutputs generate_outputs;
            generate_outputs.request_id = output;
            stream->enqueueGenerateOutput(std::move(generate_outputs));
        }
        EXPECT_EQ(stream->generate_outputs_.size(), stream->kOutputCapacity);

        GenerateOutputs overflow_output;
        overflow_output.request_id = stream->kOutputCapacity;
        stream->enqueueGenerateOutput(std::move(overflow_output));
        EXPECT_EQ(stream->generate_outputs_.size(), stream->kOutputCapacity);
    }

    const auto result = stream->nextOutput();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::OUTPUT_QUEUE_FULL);
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
}

TEST_F(GenerateStreamTest, consumerWaitWakesOnNeedRemoteGenerate) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    std::promise<void> consumer_started;
    auto               consumer_ready = consumer_started.get_future();
    auto               consumer       = std::async(std::launch::async, [stream, &consumer_started] {
        consumer_started.set_value();
        return stream->nextOutput();
    });
    consumer_ready.get();
    stream->reportEvent(StreamEvents::NeedRemoteGenerate);

    waitForConsumer(consumer, stream);
    auto result = consumer.get();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::FINISHED);
    EXPECT_NE(stream->getStatus(), StreamState::FINISHED);
}

TEST_F(GenerateStreamTest, pdUpdatePublishesOutputBeforeRemoteHandoffCompletion) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createComplexContextStream({1, 2, 3}));
    stream->generateConfig()->num_return_sequences = 1;
    stream->generateConfig()->pd_separation        = true;
    stream->generate_status_->status.store(StreamState::RUNNING);

    const auto new_tokens = torch::tensor({{42}}, torch::kInt32);
    stream->update({new_tokens,
                    1,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false});

    ASSERT_TRUE(stream->hasEvent(StreamEvents::NeedRemoteGenerate));
    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    ASSERT_EQ(output_result.value().generate_outputs.size(), 1);
    EXPECT_EQ(output_result.value().generate_outputs[0].output_ids.item<int>(), 42);

    auto finished_result = stream->nextOutput();
    ASSERT_FALSE(finished_result.ok());
    EXPECT_EQ(finished_result.status().code(), ErrorCode::FINISHED);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
}

TEST_F(GenerateStreamTest, expiredConsumerDeadlineReportsTimeout) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));
    stream->generateConfig()->timeout_ms = 50;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 100 * 1000);

    auto result = stream->nextOutput();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST_F(GenerateStreamTest, positiveWaitTimeoutReturnsNoUpdateWithoutChangingStreamState) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    auto result = stream->nextOutput(1);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::OUTPUT_QUEUE_NO_UPDATE);
    EXPECT_TRUE(stream->statusInfo().ok());
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
}

TEST_F(GenerateStreamTest, queuedOutputWinsOverExpiredDeadline) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));
    stream->generateConfig()->timeout_ms = 20;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 40 * 1000);

    GenerateOutputs outputs;
    outputs.request_id = 789;
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(std::move(outputs));
    }

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    EXPECT_EQ(output_result.value().request_id, 789);

    const auto timeout_result = stream->nextOutput();
    ASSERT_FALSE(timeout_result.ok());
    EXPECT_EQ(timeout_result.status().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST_F(GenerateStreamTest, schedulerTimeoutDoesNotOverrideQueuedOutput) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));
    stream->generateConfig()->timeout_ms = 10;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 20 * 1000);

    GenerateOutputs outputs;
    outputs.request_id = 901;
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(std::move(outputs));
    }
    EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
    EXPECT_TRUE(stream->statusInfo().ok());

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    EXPECT_EQ(output_result.value().request_id, 901);
}

TEST_F(GenerateStreamTest, schedulerTimeoutDoesNotOverridePendingCompletion) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createComplexContextStream({1, 2, 3}));
    stream->setNeedReleaseResource(true);
    stream->generate_status_->status.store(StreamState::RUNNING);
    stream->generateConfig()->timeout_ms = 10;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 20 * 1000);

    stream->reportEvent(StreamEvents::GenerateDone);

    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(stream->statusInfo().ok());

    auto finished_result = stream->nextOutput();
    ASSERT_FALSE(finished_result.ok());
    EXPECT_EQ(finished_result.status().code(), ErrorCode::FINISHED);
}

TEST_F(GenerateStreamTest, schedulerTimeoutDoesNotOverrideRemoteHandoff) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));
    stream->generateConfig()->timeout_ms = 10;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 20 * 1000);

    stream->reportEvent(StreamEvents::NeedRemoteGenerate);

    EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
    EXPECT_TRUE(stream->statusInfo().ok());

    auto finished_result = stream->nextOutput();
    ASSERT_FALSE(finished_result.ok());
    EXPECT_EQ(finished_result.status().code(), ErrorCode::FINISHED);
}

TEST_F(GenerateStreamTest, singleProducerConsumerPreservesOutputOrder) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    constexpr size_t output_count = 200;
    auto             consumer     = std::async(std::launch::async, [stream] {
        std::vector<int64_t> request_ids;
        request_ids.reserve(output_count);
        for (size_t output = 0; output < output_count; ++output) {
            auto result = stream->nextOutput();
            if (!result.ok()) {
                return std::vector<int64_t>{};
            }
            request_ids.push_back(result.value().request_id);
        }
        return request_ids;
    });

    for (size_t output = 0; output < output_count; ++output) {
        GenerateOutputs generate_outputs;
        generate_outputs.request_id = output;
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(std::move(generate_outputs));
    }

    waitForConsumer(consumer, stream);
    const auto request_ids = consumer.get();
    ASSERT_EQ(request_ids.size(), output_count);
    for (size_t output = 0; output < output_count; ++output) {
        EXPECT_EQ(request_ids[output], output);
    }
    EXPECT_TRUE(stream->statusInfo().ok());
}

TEST_F(GenerateStreamTest, publicReadinessReaderIsSafeDuringPublication) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createContextStream({1, 2, 3}));

    std::promise<void> start;
    auto               start_signal = start.get_future().share();
    auto               reader       = std::async(std::launch::async, [stream, start_signal] {
        start_signal.wait();
        for (size_t iteration = 0; iteration < 2000; ++iteration) {
            static_cast<void>(stream->hasError());
            static_cast<void>(stream->isActive());
            static_cast<void>(stream->hasEvent(StreamEvents::GenerateDone));
            static_cast<void>(stream->hasOutput());
        }
    });

    auto publisher = std::async(std::launch::async, [stream, start_signal] {
        start_signal.wait();
        for (size_t output = 0; output < 200; ++output) {
            GenerateOutputs generate_outputs;
            generate_outputs.request_id = output;
            std::lock_guard<std::mutex> lock(*stream->mutex_);
            stream->enqueueGenerateOutput(std::move(generate_outputs));
        }
        stream->reportEvent(StreamEvents::GenerateDone);
        stream->reportError(ErrorCode::CANCELLED, "cancelled");
    });

    start.set_value();
    waitForConsumer(reader, stream);
    waitForConsumer(publisher, stream);
    reader.get();
    publisher.get();

    EXPECT_TRUE(stream->hasOutput());
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

}  // namespace rtp_llm
