#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "autil/TimeUtility.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;
using OutputResult = ErrorResult<GenerateOutputs>;

// Use the existing virtual hasError() as a test-only observation seam. The
// third predicate evaluation is wait_for's initial predicate, after two empty
// observations. nextOutput, enqueue, state transition and CV stay production.
// Compile this translation unit with the repository's -fno-access-control.
struct WaitFence {
    std::promise<void>          reached;
    std::promise<void>          release;
    std::shared_future<void>    released = release.get_future().share();
    mutable std::atomic<size_t> calls{0};
};

class ObservedNormalStream final: public NormalGenerateStream {
public:
    using NormalGenerateStream::NormalGenerateStream;
    std::shared_ptr<WaitFence> fence;

    bool hasError() const override {
        if (fence && ++fence->calls == 3) {
            fence->reached.set_value();
            fence->released.wait();
        }
        return GenerateStream::hasError();
    }
};

class NormalGenerateStreamOutputWaitTest: public ::testing::Test {
protected:
    std::shared_ptr<ObservedNormalStream> makeStream(int64_t timeout_ms = 0) {
        auto input                             = std::make_shared<GenerateInput>();
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 32;
        input->generate_config->timeout_ms     = timeout_ms;
        input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
        input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
        input->need_release_resource           = false;
        ModelConfig model;
        model.max_seq_len                  = 128;
        model.vocab_size                   = 128;
        model.attn_config.tokens_per_block = 2;
        auto stream = std::make_shared<ObservedNormalStream>(input, model, RuntimeConfig{}, ResourceContext{}, nullptr);
        stream->generate_status_->status.store(StreamState::RUNNING);
        return stream;
    }

    static GenerateOutputs output(int64_t id, bool final = false) {
        GenerateOutputs outputs;
        outputs.request_id = id;
        GenerateOutput item;
        item.output_ids = torch::tensor({{static_cast<int32_t>(id)}}, torch::kInt32);
        item.finished   = final;
        outputs.generate_outputs.push_back(std::move(item));
        return outputs;
    }

    static void enqueue(const std::shared_ptr<ObservedNormalStream>& stream, int64_t id, bool final = false) {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->enqueueGenerateOutput(output(id, final));
    }

    static std::future<OutputResult> read(const std::shared_ptr<ObservedNormalStream>& stream) {
        return std::async(std::launch::async, [stream] { return stream->nextOutput(); });
    }

    static void expectPrompt(std::future<OutputResult>& result) {
        EXPECT_EQ(result.wait_for(100ms), std::future_status::ready)
            << "notification must finish before autil's 1s periodic poll";
    }

    // The worker pauses under mutex_ at wait_for's predicate. After release,
    // a producer acquiring mutex_ can publish only after the atomic CV release.
    static std::future<OutputResult> startFenced(const std::shared_ptr<ObservedNormalStream>& stream) {
        stream->fence = std::make_shared<WaitFence>();
        auto reached  = stream->fence->reached.get_future();
        auto result   = read(stream);
        EXPECT_EQ(reached.wait_for(100ms), std::future_status::ready);
        stream->fence->release.set_value();
        return result;
    }

    static void expectOutput(OutputResult result, int64_t id, bool final = false) {
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        EXPECT_EQ(result.value().request_id, id);
        ASSERT_EQ(result.value().generate_outputs.size(), 1);
        EXPECT_EQ(result.value().generate_outputs[0].finished, final);
    }
};

TEST_F(NormalGenerateStreamOutputWaitTest, EnqueueBetweenEmptyPredicateAndAtomicWait) {
    auto stream   = makeStream();
    stream->fence = std::make_shared<WaitFence>();
    auto reached  = stream->fence->reached.get_future();
    auto result   = read(stream);
    EXPECT_EQ(reached.wait_for(100ms), std::future_status::ready);
    // Pre-arm the producer while the consumer still holds mutex_. Publication
    // cannot cross the empty-to-wait gap without taking that same mutex.
    std::promise<void> producer_started;
    auto               started  = producer_started.get_future();
    auto               producer = std::async(std::launch::async, [&] {
        producer_started.set_value();
        enqueue(stream, 7);
    });
    EXPECT_EQ(started.wait_for(100ms), std::future_status::ready);
    stream->fence->release.set_value();
    producer.get();
    expectPrompt(result);
    expectOutput(result.get(), 7);
    EXPECT_FALSE(stream->hasOutput());
}

TEST_F(NormalGenerateStreamOutputWaitTest, FinalOutputDequeuedWhileRunningThenSchedulerFinishes) {
    auto stream = makeStream();
    enqueue(stream, 9, true);
    expectOutput(stream->nextOutput(), 9, true);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    auto result = startFenced(stream);
    stream->reportEvent(StreamEvents::GenerateDone);
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    expectPrompt(result);
    EXPECT_EQ(result.get().status().code(), ErrorCode::FINISHED);
}

TEST_F(NormalGenerateStreamOutputWaitTest, FinishedDrainsQueuedOutputsFifoThenReturnsFinished) {
    auto stream = makeStream();
    enqueue(stream, 1);
    enqueue(stream, 2, true);
    stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
    expectOutput(stream->nextOutput(), 1);
    expectOutput(stream->nextOutput(), 2, true);
    EXPECT_EQ(stream->nextOutput().status().code(), ErrorCode::FINISHED);
}

TEST_F(NormalGenerateStreamOutputWaitTest, ZeroOutputFinishedBeforeReader) {
    auto stream = makeStream();
    stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
    auto result = read(stream);
    expectPrompt(result);
    EXPECT_EQ(result.get().status().code(), ErrorCode::FINISHED);
}

TEST_F(NormalGenerateStreamOutputWaitTest, ZeroOutputFinishedWakesReader) {
    auto stream = makeStream();
    auto result = startFenced(stream);
    stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
    expectPrompt(result);
    EXPECT_EQ(result.get().status().code(), ErrorCode::FINISHED);
}

TEST_F(NormalGenerateStreamOutputWaitTest, ErrorAndCancelBeforeReaderPreserveOriginalErrorAndQueuedFinal) {
    for (auto code : {ErrorCode::UNKNOWN_ERROR, ErrorCode::CANCELLED}) {
        auto stream = makeStream();
        enqueue(stream, 11, true);
        stream->reportError(code, "original");
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "later");
        auto result = read(stream);
        expectPrompt(result);
        auto value = result.get();
        EXPECT_EQ(value.status().code(), code);
        EXPECT_EQ(value.status().ToString(), "original");
        EXPECT_EQ(stream->generate_outputs_queue_.getSize(), 1);
    }
}

TEST_F(NormalGenerateStreamOutputWaitTest, EveryPublicErrorPublisherWakesWaitingReader) {
    for (int publisher = 0; publisher < 3; ++publisher) {
        auto stream = makeStream();
        auto result = startFenced(stream);
        if (publisher == 0) {
            stream->reportEvent(StreamEvents::Error, ErrorCode::CANCELLED, "cancel");
        } else if (publisher == 1) {
            std::lock_guard<std::mutex> lock(*stream->mutex_);
            stream->reportEventWithoutLock(StreamEvents::Error, ErrorCode::CANCELLED, "cancel");
        } else {
            stream->reportError(ErrorCode::CANCELLED, "cancel");
        }
        expectPrompt(result);
        auto value = result.get();
        EXPECT_EQ(value.status().code(), ErrorCode::CANCELLED);
        EXPECT_EQ(value.status().ToString(), "cancel");
    }
}

TEST_F(NormalGenerateStreamOutputWaitTest, TimeoutDetectedInInitialCheckDoesNotWaitAgainOrDeadlock) {
    auto stream = makeStream(1);
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 1000000);
    auto result = read(stream);
    expectPrompt(result);
    EXPECT_EQ(result.get().status().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST_F(NormalGenerateStreamOutputWaitTest, QueuedOutputAndAlreadyFinishedBypassFreshTimeoutCheck) {
    auto stream = makeStream(1);
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 1000000);
    enqueue(stream, 13);
    expectOutput(stream->nextOutput(), 13);
    EXPECT_FALSE(stream->GenerateStream::hasError());
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->generate_status_->status.store(StreamState::FINISHED);
    }
    EXPECT_EQ(stream->nextOutput().status().code(), ErrorCode::FINISHED);
    EXPECT_FALSE(stream->GenerateStream::hasError());
}

TEST_F(NormalGenerateStreamOutputWaitTest, PeriodicWaitStillChecksTimeoutUnlocked) {
    auto stream = makeStream(500);
    auto result = startFenced(stream);
    // No notifications: the existing 1s periodic timeout must still be retained.
    ASSERT_EQ(result.wait_for(1500ms), std::future_status::ready);
    EXPECT_EQ(result.get().status().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST_F(NormalGenerateStreamOutputWaitTest, QueueFullIsNonblockingAndPreservesAllOriginalEntries) {
    auto stream = makeStream();
    for (int id = 0; id < 1000; ++id) {
        enqueue(stream, id);
    }
    auto producer = std::async(std::launch::async, [stream] { enqueue(stream, 1000); });
    EXPECT_EQ(producer.wait_for(100ms), std::future_status::ready);
    producer.get();
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::OUTPUT_QUEUE_FULL);
    EXPECT_EQ(stream->generate_outputs_queue_.getSize(), 1000);
    EXPECT_EQ(stream->nextOutput().status().code(), ErrorCode::OUTPUT_QUEUE_FULL);
    // Error policy leaves the queue untouched; inspect remaining contents only
    // after the producer has joined and no output consumer is running.
    for (int id = 0; id < 1000; ++id) {
        EXPECT_EQ(stream->generate_outputs_queue_.getAndPopFront().request_id, id);
    }
}

TEST_F(NormalGenerateStreamOutputWaitTest, OneNotificationWakesCopiedStreamsSeparateQueuesAndRemotePredicate) {
    auto stream = makeStream();
    auto copy   = std::make_shared<ObservedNormalStream>(static_cast<const GenerateStream&>(*stream));
    ASSERT_EQ(stream->mutex_, copy->mutex_);
    ASSERT_EQ(stream->cv_, copy->cv_);
    ASSERT_EQ(stream->generate_status_, copy->generate_status_);
    auto               first  = startFenced(stream);
    auto               second = startFenced(copy);
    std::promise<void> remote_started;
    auto               started = remote_started.get_future();
    auto               remote  = std::async(std::launch::async, [&] {
        remote_started.set_value();
        return copy->waitForRemoteGenerate();
    });
    EXPECT_EQ(started.wait_for(100ms), std::future_status::ready);
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        // Direct queue publication deliberately sends no output notification.
        // Only the actual NeedRemoteGenerate publisher may wake all consumers.
        stream->generate_outputs_queue_.push(output(21));
        copy->generate_outputs_queue_.push(output(22));
        stream->reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
    }
    expectPrompt(first);
    expectPrompt(second);
    EXPECT_EQ(remote.wait_for(100ms), std::future_status::ready);
    expectOutput(first.get(), 21);
    expectOutput(second.get(), 22);
    EXPECT_TRUE(remote.get());
}

TEST_F(NormalGenerateStreamOutputWaitTest, NeedRemoteGenerateWithoutOutputWakesRealRemoteConsumer) {
    auto               stream = makeStream();
    std::promise<void> started_promise;
    auto               started = started_promise.get_future();
    auto               remote  = std::async(std::launch::async, [&] {
        started_promise.set_value();
        return stream->waitForRemoteGenerate();
    });
    EXPECT_EQ(started.wait_for(100ms), std::future_status::ready);
    stream->reportEvent(StreamEvents::NeedRemoteGenerate);
    EXPECT_EQ(remote.wait_for(100ms), std::future_status::ready);
    EXPECT_TRUE(remote.get());
    EXPECT_FALSE(stream->hasOutput());
}

TEST_F(NormalGenerateStreamOutputWaitTest, SchedulerFinishedWakesEveryCopiedOutputConsumer) {
    auto stream = makeStream();
    auto copy   = std::make_shared<ObservedNormalStream>(static_cast<const GenerateStream&>(*stream));
    auto first  = startFenced(stream);
    auto second = startFenced(copy);
    stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
    expectPrompt(first);
    expectPrompt(second);
    EXPECT_EQ(first.get().status().code(), ErrorCode::FINISHED);
    EXPECT_EQ(second.get().status().code(), ErrorCode::FINISHED);
}
}  // namespace
}  // namespace rtp_llm
