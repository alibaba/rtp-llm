
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

TEST_F(GenerateStreamTest, testBatchSizeWithNumReturnSequences) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input                                   = std::make_shared<GenerateInput>();
    generate_input->generate_config                       = std::make_shared<GenerateConfig>();
    generate_input->generate_config->num_return_sequences = 3;
    generate_input->input_ids                             = torch::tensor({1, 2, 3}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    EXPECT_EQ(1, stream->batchSize(0));
    EXPECT_EQ(3, stream->batchSize(1));
    EXPECT_EQ(3, stream->batchSize(5));
    EXPECT_EQ(3, stream->maxBatchSize());
    EXPECT_TRUE(stream->needTilingForSampling());
}

TEST_F(GenerateStreamTest, testBatchSizeWithBeamSearch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    RuntimeConfig runtime_config;

    auto generate_input                        = std::make_shared<GenerateInput>();
    generate_input->generate_config            = std::make_shared<GenerateConfig>();
    generate_input->generate_config->num_beams = 4;
    generate_input->input_ids                  = torch::tensor({1, 2, 3}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    EXPECT_EQ(1, stream->batchSize(0));
    EXPECT_EQ(4, stream->batchSize(1));
    EXPECT_EQ(4, stream->maxBatchSize());
    EXPECT_FALSE(stream->needTilingForSampling());
}

TEST_F(GenerateStreamTest, testCompleteTokenIdsUsesRequestBoundAndInitializesAllRows) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    RuntimeConfig runtime_config;

    auto generate_input                             = std::make_shared<GenerateInput>();
    generate_input->generate_config                 = std::make_shared<GenerateConfig>();
    generate_input->generate_config->num_beams      = 2;
    generate_input->generate_config->max_new_tokens = 4;
    generate_input->input_ids                       = torch::tensor({7, 8, 9}, torch::kInt32);

    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);

    auto token_ids = stream->completeTokenIds();
    ASSERT_EQ(2, token_ids.size(0));
    ASSERT_EQ(7, token_ids.size(1));
    EXPECT_TRUE(torch::equal(token_ids[0].narrow(0, 0, 3), generate_input->input_ids));
    EXPECT_TRUE(torch::equal(token_ids[1].narrow(0, 0, 3), generate_input->input_ids));
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
    const uint64_t epoch_1 = stream->setMtpAsyncDeviceState(std::move(s1));
    ASSERT_EQ(epoch_1, 1u);
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 2: another publish before the worker for epoch_1 ran. Counter
    // bumps; old epoch should now be stale.
    GenerateStream::MtpAsyncDeviceState s2;
    s2.accept_len_gpu      = torch::ones({1}, torch::kInt32) * 2;
    const uint64_t epoch_2 = stream->setMtpAsyncDeviceState(std::move(s2));
    ASSERT_EQ(epoch_2, 2u);
    ASSERT_NE(epoch_1, epoch_2);

    // Stale worker for epoch_1 attempts to clear: must be rejected, state
    // for epoch_2 must remain intact.
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_1));
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, epoch_2);

    // Worker for epoch_2 clears successfully.
    ASSERT_TRUE(stream->clearMtpAsyncDeviceState(epoch_2));
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
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
TEST_F(GenerateStreamTest, testDynamicBeamLayoutDependsOnCurrentTransition) {
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;

    auto input                                 = std::make_shared<GenerateInput>();
    input->input_ids                           = torch::tensor({2}, torch::kInt32);
    input->generate_config                     = std::make_shared<GenerateConfig>();
    input->generate_config->variable_num_beams = {1, 2, 1};
    auto stream =
        std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);

    EXPECT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());
    stream->setSeqLength(2);
    EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
    stream->setSeqLength(3);
    EXPECT_TRUE(stream->usesBeamSearchTokenLayoutForCurrentStep());
    stream->setSeqLength(4);
    EXPECT_FALSE(stream->usesBeamSearchTokenLayoutForCurrentStep());
}

TEST_F(GenerateStreamTest, testDynamicBeamOutputUsesUpdatedCurrentBatchSize) {
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;

    auto input                                 = std::make_shared<GenerateInput>();
    input->input_ids                           = torch::tensor({2}, torch::kInt32);
    input->generate_config                     = std::make_shared<GenerateConfig>();
    input->generate_config->variable_num_beams = {2, 3};
    auto stream =
        std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);

    auto beam_tokens    = torch::tensor({2, 4, 2, 7}, torch::kInt32).reshape({2, 2});
    int  error_token_id = 0;
    ASSERT_TRUE(
        stream->complete_token_ids_->update(beam_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->resizeSubGenerateStatus(2);
    ASSERT_EQ(stream->currentBatchSize(), 2);
    ASSERT_EQ(stream->nextBatchSize(), 3);

    auto outputs = stream->prepareGenerateOutput({beam_tokens,
                                                  1,
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Tensor()});
    ASSERT_EQ(outputs.generate_outputs.size(), 2);
    EXPECT_EQ(outputs.generate_outputs[0].output_ids[0][0].item<int32_t>(), 4);
    EXPECT_EQ(outputs.generate_outputs[1].output_ids[0][0].item<int32_t>(), 7);
}

TEST_F(GenerateStreamTest, testDynamicBeamSoftmaxHistoryFollowsParentRows) {
    ModelConfig model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;

    auto input                                   = std::make_shared<GenerateInput>();
    input->input_ids                             = torch::tensor({2}, torch::kInt32);
    input->generate_config                       = std::make_shared<GenerateConfig>();
    input->generate_config->variable_num_beams   = {2, 3};
    input->generate_config->max_new_tokens       = 3;
    input->generate_config->return_softmax_probs = true;
    auto stream =
        std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);

    int  error_token_id = 0;
    auto first_tokens   = torch::tensor({2, 4, 2, 7}, torch::kInt32).reshape({2, 2});
    ASSERT_TRUE(
        stream->complete_token_ids_->update(first_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->setSoftmaxProbs(torch::tensor({0.1f, 0.2f}).reshape({2, 1}), 1, torch::tensor({0, 0}, torch::kInt32));

    auto second_tokens = torch::tensor({2, 7, 1, 2, 7, 3, 2, 4, 5}, torch::kInt32).reshape({3, 3});
    ASSERT_TRUE(
        stream->complete_token_ids_->update(second_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->setSoftmaxProbs(
        torch::tensor({0.3f, 0.4f, 0.5f}).reshape({3, 1}), 2, torch::tensor({1, 1, 0}, torch::kInt32));

    auto probabilities = stream->getSoftmaxProbs();
    EXPECT_FLOAT_EQ(probabilities[0][1].item<float>(), 0.2f);
    EXPECT_FLOAT_EQ(probabilities[1][1].item<float>(), 0.2f);
    EXPECT_FLOAT_EQ(probabilities[2][1].item<float>(), 0.1f);
    EXPECT_FLOAT_EQ(probabilities[0][2].item<float>(), 0.3f);
    EXPECT_FLOAT_EQ(probabilities[1][2].item<float>(), 0.4f);
    EXPECT_FLOAT_EQ(probabilities[2][2].item<float>(), 0.5f);

    stream->setSoftmaxProbs(torch::tensor({0.6f, 0.7f, 0.8f}).reshape({3, 1}), 3, torch::Tensor());
    EXPECT_EQ(probabilities.size(0), 3);
    EXPECT_EQ(probabilities.size(1), 4);
    EXPECT_FLOAT_EQ(probabilities[0][3].item<float>(), 0.6f);
    EXPECT_FLOAT_EQ(probabilities[1][3].item<float>(), 0.7f);
    EXPECT_FLOAT_EQ(probabilities[2][3].item<float>(), 0.8f);
}

TEST_F(GenerateStreamTest, testNonStreamingFinalOutputReturnsCachedAllHiddenStates) {
    auto builder                                                       = GenerateStreamBuilder();
    auto stream                                                        = builder.createContextStream({1, 2});
    stream->generate_input_->generate_config->max_new_tokens           = 2;
    stream->generate_input_->generate_config->return_all_hidden_states = true;
    stream->generate_input_->generate_config->return_incremental       = true;
    stream->generate_input_->generate_config->is_streaming             = false;

    auto first_all_hidden_states = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({2, 2});
    stream->step();
    stream->update(StreamUpdateInfo{torch::tensor({10}, torch::kInt32).reshape({1, 1}),
                                    1,
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    first_all_hidden_states,
                                    false});
    ASSERT_FALSE(stream->hasOutput());

    stream->step();
    stream->update(StreamUpdateInfo{torch::tensor({11}, torch::kInt32).reshape({1, 1}),
                                    1,
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    false});

    ASSERT_TRUE(stream->hasOutput());
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    const auto& generate_output = output.value().generate_outputs[0];
    ASSERT_TRUE(generate_output.finished);
    ASSERT_TRUE(generate_output.all_hidden_states.has_value());
    ASSERT_TRUE(torch::equal(generate_output.all_hidden_states.value(), first_all_hidden_states));
}

TEST_F(GenerateStreamTest, testAllHiddenStatesCopiedToCpuOnceForMultipleOutputs) {
    auto builder = GenerateStreamBuilder();
    auto stream  = std::dynamic_pointer_cast<NormalGenerateStream>(builder.createComplexContextStream({1, 2}));
    ASSERT_NE(stream, nullptr);
    stream->generate_input_->generate_config->return_all_hidden_states = true;
    stream->iter_count_                                                = 1;

    auto all_hidden_states =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});
    StreamUpdateInfo update_info{torch::Tensor(),
                                 0,
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 all_hidden_states,
                                 false};

    auto outputs = stream->prepareGenerateOutput(update_info);

    ASSERT_EQ(outputs.generate_outputs.size(), 2);
    const auto& first  = outputs.generate_outputs[0].all_hidden_states;
    const auto& second = outputs.generate_outputs[1].all_hidden_states;
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_FALSE(first->is_cuda());
    ASSERT_EQ(first->data_ptr(), second->data_ptr());
    ASSERT_TRUE(torch::equal(first.value(), all_hidden_states.cpu()));
}

TEST_F(GenerateStreamTest, testInputEmbeddingsDisableTokenOnlyReuseCache) {
    auto builder                                   = GenerateStreamBuilder();
    auto stream                                    = builder.createContextStream({1, 2, 3, 4, 5, 6});
    stream->generate_input_->input_embeddings      = std::vector<torch::Tensor>{torch::rand({1, 8}, torch::kFloat32)};
    stream->generate_input_->input_embeddings_locs = std::vector<int32_t>{2};

    ASSERT_TRUE(stream->hasInputEmbeddings());
    ASSERT_FALSE(stream->reuseCache());
    ASSERT_FALSE(stream->enableDeviceCache());
    ASSERT_FALSE(stream->enableMemoryCache());
    ASSERT_FALSE(stream->enableRemoteCache());
}

}  // namespace rtp_llm
