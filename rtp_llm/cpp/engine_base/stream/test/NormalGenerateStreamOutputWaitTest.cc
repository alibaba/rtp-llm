// Exercise publication, bounded cancellation and terminal precedence on the real stream.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

constexpr int64_t kSentinel      = 999;
constexpr auto    kBoundedReturn = std::chrono::seconds(20);

int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::shared_ptr<NormalGenerateStream> makeStream(int timeout_ms = -1) {
    auto generate_input             = std::make_shared<GenerateInput>();
    auto generate_config            = std::make_shared<GenerateConfig>();
    generate_config->timeout_ms     = timeout_ms;
    generate_input->generate_config = generate_config;
    generate_input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    model_config.max_seq_len = 2048;
    return std::make_shared<NormalGenerateStream>(
        generate_input, model_config, runtime_config, resource_context, nullptr);
}
void publish(const std::shared_ptr<NormalGenerateStream>& stream, int64_t marker) {
    GenerateOutputs outputs;
    outputs.request_id = marker;
    std::lock_guard<std::mutex> lock(*stream->mutex_);
    stream->enqueueGenerateOutput(std::move(outputs));
}
TEST(NormalGenerateStreamOutputWait, C1PublishIsSerializedAgainstWaitRegistration) {
    auto                         stream = makeStream();
    std::unique_lock<std::mutex> registration_window(*stream->mutex_);

    std::mutex              done_mu;
    std::condition_variable done_cv;
    bool                    publish_returned = false;
    std::atomic<bool>       at_publish{false};

    std::thread producer([&] {
        at_publish.store(true);
        publish(stream, 1);
        {
            std::lock_guard<std::mutex> lk(done_mu);
            publish_returned = true;
        }
        done_cv.notify_all();
    });
    while (!at_publish.load()) {
        std::this_thread::yield();
    }
    bool completed_inside_window;
    {
        std::unique_lock<std::mutex> done_lock(done_mu);
        completed_inside_window =
            done_cv.wait_for(done_lock, std::chrono::milliseconds(500), [&] { return publish_returned; });
    }
    EXPECT_FALSE(completed_inside_window)
        << "publish completed while a waiter owned the predicate-to-park window, so a notify could land "
           "after the predicate returned false and before the wait was registered";
    ErrorResult<GenerateOutputs> result(ErrorCode::UNKNOWN_ERROR, "unset");
    std::thread                  consumer([&] { result = stream->nextOutput(); });

    registration_window.unlock();
    producer.join();
    consumer.join();

    {
        std::lock_guard<std::mutex> lk(done_mu);
        EXPECT_TRUE(publish_returned);
    }
    ASSERT_TRUE(result.ok()) << "consumer was not served after the registration window closed";
    EXPECT_EQ(1, result.value().request_id);
    EXPECT_TRUE(stream->generate_outputs_.empty());
}
TEST(NormalGenerateStreamOutputWait, C2AlreadyQueuedOutputNeverRegistersAWait) {
    auto stream = makeStream();
    publish(stream, 7);
    auto result = stream->nextOutput();

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(7, result.value().request_id);
}

TEST(NormalGenerateStreamOutputWait, C2OutputPublishedDuringWaitIsDeliveredExactlyOnceInOrder) {
    auto stream = makeStream();

    for (int64_t marker = 1; marker <= 3; ++marker) {
        ErrorResult<GenerateOutputs> result(ErrorCode::UNKNOWN_ERROR, "unset");
        std::thread                  consumer([&] { result = stream->nextOutput(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        publish(stream, marker);
        consumer.join();

        ASSERT_TRUE(result.ok()) << "round " << marker;
        EXPECT_EQ(marker, result.value().request_id) << "delivery was not ordered";
    }
    EXPECT_TRUE(stream->generate_outputs_.empty());
    stream->generate_status_->status = StreamState::FINISHED;
    auto extra                       = stream->nextOutput();
    EXPECT_FALSE(extra.ok());
    EXPECT_EQ(ErrorCode::FINISHED, extra.status().code());
}
TEST(NormalGenerateStreamOutputWait, C3CancelledBeforeWaitingReturnsImmediately) {
    auto stream = makeStream();
    auto result = stream->nextOutput([] { return true; });

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(ErrorCode::CANCELLED, result.status().code());
}

TEST(NormalGenerateStreamOutputWait, C3CancelledWhileParkedWithNoOutputIsBounded) {
    auto stream = makeStream();

    std::atomic<bool> cancel{false};
    auto              began = std::chrono::steady_clock::now();

    ErrorResult<GenerateOutputs> result(ErrorCode::UNKNOWN_ERROR, "unset");
    std::thread                  consumer([&] { result = stream->nextOutput([&] { return cancel.load(); }); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    cancel.store(true);
    consumer.join();
    auto elapsed = std::chrono::steady_clock::now() - began;

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(ErrorCode::CANCELLED, result.status().code());
    EXPECT_LT(elapsed, kBoundedReturn) << "cancellation was not bounded by the wait slice";
}
TEST(NormalGenerateStreamOutputWait, C4TerminalCausePrecedenceAndFirstErrorWins) {
    {
        auto stream = makeStream();
        stream->reportError(ErrorCode::MALLOC_FAILED, "oom");
        auto result = stream->nextOutput([] { return true; });
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::MALLOC_FAILED, result.status().code());
    }
    {
        auto stream = makeStream();
        publish(stream, 5);
        auto result = stream->nextOutput([] { return true; });
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(5, result.value().request_id);
    }
    {
        auto stream                      = makeStream();
        stream->generate_status_->status = StreamState::FINISHED;
        auto result                      = stream->nextOutput();
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::FINISHED, result.status().code());
    }
    {
        auto stream            = makeStream(/*timeout_ms=*/1);
        stream->begin_time_us_ = nowUs() - 60LL * 1000 * 1000;  // 60 s ago: already past the deadline
        auto result            = stream->nextOutput();
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::GENERATE_TIMEOUT, result.status().code());
    }
    {
        auto stream            = makeStream(/*timeout_ms=*/1);
        stream->begin_time_us_ = nowUs() - 60LL * 1000 * 1000;
        stream->reportError(ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
        stream->checkTimeoutWithoutLock();  // already past the deadline: fires, but must not overwrite the cause
        auto result = stream->nextOutput();
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::OUTPUT_QUEUE_FULL, result.status().code());
    }
}
TEST(NormalGenerateStreamOutputWait, C5FullQueueIsATerminalErrorWithoutBlockingTheProducer) {
    auto stream = makeStream();
    for (size_t i = 0; i < stream->kOutputCapacity - 2; ++i)
        publish(stream, 0);

    publish(stream, 1);
    publish(stream, 2);
    publish(stream, 3);
    ASSERT_TRUE(stream->hasError());
    auto first  = stream->nextOutput();
    auto second = stream->nextOutput();
    ASSERT_FALSE(first.ok());
    EXPECT_EQ(ErrorCode::OUTPUT_QUEUE_FULL, first.status().code());
    ASSERT_FALSE(second.ok());
    EXPECT_EQ(ErrorCode::OUTPUT_QUEUE_FULL, second.status().code());
}
class NoCancelSupportStream: public GenerateStream {
public:
    using GenerateStream::GenerateStream;

    ErrorResult<GenerateOutputs> nextOutput(int64_t wait_timeout_ms = 0) override {
        GenerateOutputs outputs;
        outputs.request_id = kSentinel;
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }
    void updateOutput(const StreamUpdateInfo&) override {}
};

TEST(NormalGenerateStreamOutputWait, C6NoArgumentOverloadStillDelivers) {
    auto stream = makeStream();
    publish(stream, 11);

    auto result = stream->nextOutput();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(11, result.value().request_id);
}

TEST(NormalGenerateStreamOutputWait, C6BaseDefaultIgnoresCancellationForSubclassesWithoutSupport) {
    auto generate_input             = std::make_shared<GenerateInput>();
    auto generate_config            = std::make_shared<GenerateConfig>();
    generate_input->generate_config = generate_config;
    generate_input->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    model_config.max_seq_len = 2048;
    GenerateStreamPtr stream = std::make_shared<NoCancelSupportStream>(
        generate_input, model_config, runtime_config, resource_context, nullptr);

    auto result = stream->nextOutput([] { return true; });
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(kSentinel, result.value().request_id);
}

TEST(NormalGenerateStreamOutputWait, C6CoordinatorOutlivesStreamDestruction) {
    auto stream      = makeStream();
    auto coordinator = stream->consumer_cv_;
    auto mutex       = stream->mutex_;
    stream.reset();
    ASSERT_NE(nullptr, coordinator);
    ASSERT_NE(nullptr, mutex);
    std::unique_lock<std::mutex> lock(*mutex);
    coordinator->notify_all();
}

}  // namespace
}  // namespace rtp_llm
