#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingShutdownSteps.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

EmbeddingStreamPtr makeStream(std::chrono::milliseconds timeout = 1s) {
    auto input =
        std::make_shared<EmbeddingInput>(std::vector<int32_t>{1}, std::vector<int32_t>{0}, std::vector<int32_t>{1}, 42);
    return std::make_shared<EmbeddingStream>(input, EmbeddingProfileConfig{}, timeout);
}

TEST(EmbeddingStreamShutdownTest, SuccessfulTensorResultRemainsUnchanged) {
    auto stream = makeStream();
    auto output = torch::ones({1});
    ASSERT_TRUE(stream->setStart());
    stream->updateTensorOutput(output);
    stream->setError("late error");
    stream->updateTensorOutput(torch::zeros({1}));
    EXPECT_FALSE(stream->setStart());
    EXPECT_NO_THROW(stream->waitFinish());
    EXPECT_NO_THROW(stream->waitFinish([] { return true; }));
    ASSERT_TRUE(stream->embeddingOutput()->output.t.has_value());
    EXPECT_TRUE(torch::equal(output, stream->embeddingOutput()->output.t.value()));
}

TEST(EmbeddingStreamShutdownTest, SuccessfulMapResultRemainsUnchanged) {
    auto                                              stream = makeStream();
    std::vector<std::map<std::string, torch::Tensor>> output{{{"value", torch::ones({1})}}};
    stream->updateMapOutput(output);
    EXPECT_NO_THROW(stream->waitFinish());
    ASSERT_TRUE(stream->embeddingOutput()->output.map.has_value());
    EXPECT_EQ(stream->embeddingOutput()->output.map->size(), 1);
    EXPECT_FALSE(stream->embeddingOutput()->output.isTensor);
}

TEST(EmbeddingStreamShutdownTest, WaitingAndExecutingRequestsHaveBoundedWaits) {
    for (bool running : {false, true}) {
        auto stream = makeStream(20ms);
        if (running) {
            ASSERT_TRUE(stream->setStart());
        }
        auto started = std::chrono::steady_clock::now();
        EXPECT_THROW(stream->waitFinish(), EmbeddingStreamTimeout);
        EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
        auto result = stream->embeddingOutput();
        EXPECT_EQ(result->error_info.code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_NE(result->error_info.ToString().find("embedding request timeout"), std::string::npos);
        EXPECT_NE(result->error_info.ToString().find("8204"), std::string::npos);

        stream->updateTensorOutput(torch::ones({1}));
        std::vector<std::map<std::string, torch::Tensor>> late{{{"value", torch::ones({1})}}};
        stream->updateMapOutput(late);
        stream->setError("late failure");
        EXPECT_EQ(late.size(), 1);
        EXPECT_FALSE(stream->setStart());
        EXPECT_FALSE(result->output.t.has_value());
        EXPECT_FALSE(result->output.map.has_value());
        EXPECT_EQ(result->error_info.code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_THROW(stream->waitFinish(), EmbeddingStreamTimeout);
    }
}

TEST(EmbeddingStreamShutdownTest, DeadlineStartsAtConstructionAndCannotBeDisabled) {
    auto stream = makeStream(1ms);
    std::this_thread::sleep_for(5ms);
    EXPECT_THROW(stream->waitFinish(), EmbeddingStreamTimeout);
    EXPECT_THROW(makeStream(0ms), std::invalid_argument);
    EXPECT_THROW(makeStream(-1ms), std::invalid_argument);
    EXPECT_THROW(makeStream(std::chrono::milliseconds::max()), std::invalid_argument);
}

TEST(EmbeddingStreamShutdownTest, CancellationWakesWaitersAndDropsLateResults) {
    for (bool running : {false, true}) {
        auto stream = makeStream(2s);
        if (running) {
            ASSERT_TRUE(stream->setStart());
        }
        std::atomic<bool>  cancelled{false};
        std::promise<void> waiting;
        auto               waiter = std::async(std::launch::async, [&] {
            bool first_check = true;
            EXPECT_THROW(stream->waitFinish([&] {
                if (first_check) {
                    first_check = false;
                    waiting.set_value();
                    return false;
                }
                return cancelled.load();
            }),
                         EmbeddingStreamCancelled);
        });
        auto               observer =
            std::async(std::launch::async, [stream] { EXPECT_THROW(stream->waitFinish(), EmbeddingStreamCancelled); });
        EXPECT_EQ(waiting.get_future().wait_for(1s), std::future_status::ready);
        cancelled = true;
        EXPECT_EQ(waiter.wait_for(500ms), std::future_status::ready);
        EXPECT_EQ(observer.wait_for(500ms), std::future_status::ready);
        waiter.get();
        observer.get();

        stream->updateTensorOutput(torch::ones({1}));
        std::vector<std::map<std::string, torch::Tensor>> late{{{"value", torch::ones({1})}}};
        stream->updateMapOutput(late);
        stream->setError("late failure");
        EXPECT_EQ(late.size(), 1);
        EXPECT_FALSE(stream->setStart());
        auto result = stream->embeddingOutput();
        EXPECT_FALSE(result->output.t.has_value());
        EXPECT_FALSE(result->output.map.has_value());
        EXPECT_EQ(result->error_info.code(), ErrorCode::CANCELLED);
        EXPECT_THROW(stream->waitFinish(), EmbeddingStreamCancelled);
    }
}

TEST(EmbeddingStreamShutdownTest, CancellationPollingPreservesTimeout) {
    auto stream = makeStream(20ms);
    EXPECT_THROW(stream->waitFinish([] { return false; }), EmbeddingStreamTimeout);
    EXPECT_EQ(stream->embeddingOutput()->error_info.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_THROW(stream->waitFinish([] { return true; }), EmbeddingStreamTimeout);
}

TEST(EmbeddingStreamShutdownTest, StopWakesWaitersAndRejectsEnqueue) {
    EmbeddingScheduler scheduler(ModelConfig{}, ConcurrencyConfig{}, RuntimeConfig{});
    auto               stream = makeStream();
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto wait = [stream] {
        try {
            stream->waitFinish();
            return false;
        } catch (const EmbeddingStreamUnavailable&) {
            return stream->embeddingOutput()->error_info.code() == ErrorCode::CANCELLED;
        }
    };
    auto first  = std::async(std::launch::async, wait);
    auto second = std::async(std::launch::async, wait);
    auto stop   = std::async(std::launch::async, [&scheduler] { return scheduler.stop(); });
    EXPECT_TRUE(scheduler.stop().ok());
    EXPECT_TRUE(stop.get().ok());
    EXPECT_EQ(first.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(second.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(first.get());
    EXPECT_TRUE(second.get());
    EXPECT_TRUE(scheduler.stop().ok());
    EXPECT_THROW(stream->waitFinish(), EmbeddingStreamUnavailable);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
    EXPECT_TRUE(scheduler.scheduleNew().value().empty());
    auto rejected = makeStream();
    EXPECT_EQ(scheduler.enqueue(rejected).code(), absl::StatusCode::kUnavailable);
    EXPECT_THROW(rejected->waitFinish(), EmbeddingStreamUnavailable);
    EXPECT_EQ(rejected->embeddingOutput()->error_info.code(), ErrorCode::CANCELLED);
}

TEST(EmbeddingStreamShutdownTest, EnqueueRejectionThrowsTypedErrorPromptly) {
    EmbeddingScheduler scheduler(ModelConfig{}, ConcurrencyConfig{}, RuntimeConfig{});
    EXPECT_TRUE(scheduler.stop().ok());
    auto rejected = makeStream(60s);
    auto status   = scheduler.enqueue(rejected);
    ASSERT_TRUE(absl::IsUnavailable(status));
    rejected->setEnqueueError(status);
    auto started = std::chrono::steady_clock::now();
    EXPECT_THROW(rejected->waitFinish([] { return false; }), EmbeddingStreamUnavailable);
    EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);

    // A rejection that did not finish the stream still fails promptly, as a generic error.
    auto failed = makeStream(60s);
    failed->setEnqueueError(absl::InternalError("queue broken"));
    started = std::chrono::steady_clock::now();
    try {
        failed->waitFinish();
        ADD_FAILURE() << "waitFinish should throw";
    } catch (const EmbeddingStreamUnavailable&) {
        ADD_FAILURE() << "unexpected unavailable";
    } catch (const EmbeddingStreamCancelled&) {
        ADD_FAILURE() << "unexpected cancelled";
    } catch (const EmbeddingStreamTimeout&) {
        ADD_FAILURE() << "unexpected timeout";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("queue broken"), std::string::npos);
    }
    EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
    EXPECT_EQ(failed->embeddingOutput()->error_info.code(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_FALSE(failed->setStart());
}

TEST(EmbeddingStreamShutdownTest, ClientDeadlineShortensRequestTimeout) {
    const auto now = std::chrono::system_clock::now();
    EXPECT_EQ(EmbeddingStream::timeoutForDeadline(std::chrono::system_clock::time_point::max(), now),
              EmbeddingStream::kRequestTimeout);
    EXPECT_EQ(EmbeddingStream::timeoutForDeadline(now + 120s, now), EmbeddingStream::kRequestTimeout);
    EXPECT_EQ(EmbeddingStream::timeoutForDeadline(now + 5s, now), 5000ms);
    EXPECT_EQ(EmbeddingStream::timeoutForDeadline(now + 500us, now), 1ms);
    EXPECT_EQ(EmbeddingStream::timeoutForDeadline(now - 1s, now), 1ms);

    auto stream  = makeStream(EmbeddingStream::timeoutForDeadline(now + 30ms, now));
    auto started = std::chrono::steady_clock::now();
    EXPECT_THROW(stream->waitFinish([] { return false; }), EmbeddingStreamTimeout);
    EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
    EXPECT_EQ(stream->embeddingOutput()->error_info.code(), ErrorCode::DEADLINE_EXCEEDED);
}

TEST(EmbeddingStreamShutdownTest, ShutdownStepsContinueAfterFailureAndResume) {
    bool first_done   = false;
    bool second_done  = false;
    int  first_calls  = 0;
    int  second_calls = 0;
    bool fail_first   = true;
    auto stop         = [&] {
        EmbeddingShutdownSteps steps;
        steps.run(first_done, "first", [&] {
            ++first_calls;
            if (fail_first) {
                throw std::runtime_error("injected failure");
            }
        });
        steps.run(second_done, "second", [&] { ++second_calls; });
        steps.rethrowFirstError();
    };
    EXPECT_THROW(stop(), std::runtime_error);
    EXPECT_FALSE(first_done);
    EXPECT_TRUE(second_done);
    EXPECT_EQ(second_calls, 1);
    fail_first = false;
    EXPECT_NO_THROW(stop());
    EXPECT_TRUE(first_done);
    EXPECT_EQ(first_calls, 2);
    EXPECT_EQ(second_calls, 1);
    EXPECT_NO_THROW(stop());
    EXPECT_EQ(first_calls, 2);
}

TEST(EmbeddingStreamShutdownTest, TimedOutQueuedRequestIsNotScheduled) {
    ModelConfig model;
    model.max_seq_len = 8;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.max_context_batch_size = 1;
    EmbeddingScheduler scheduler(model, ConcurrencyConfig{}, runtime);
    auto               expired = makeStream(1ms);
    ASSERT_TRUE(scheduler.enqueue(expired).ok());
    EXPECT_THROW(expired->waitFinish(), EmbeddingStreamTimeout);
    auto live = makeStream();
    ASSERT_TRUE(scheduler.enqueue(live).ok());
    auto scheduled = scheduler.scheduleNew();
    ASSERT_TRUE(scheduled.ok());
    ASSERT_EQ(scheduled->size(), 1);
    EXPECT_EQ(scheduled->front(), live);
}

TEST(EmbeddingStreamShutdownTest, CancelledQueuedRequestIsNotScheduled) {
    ModelConfig model;
    model.max_seq_len = 8;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.max_context_batch_size = 1;
    EmbeddingScheduler scheduler(model, ConcurrencyConfig{}, runtime);
    auto               cancelled = makeStream();
    ASSERT_TRUE(scheduler.enqueue(cancelled).ok());
    EXPECT_THROW(cancelled->waitFinish([] { return true; }), EmbeddingStreamCancelled);
    auto live = makeStream();
    ASSERT_TRUE(scheduler.enqueue(live).ok());
    auto scheduled = scheduler.scheduleNew();
    ASSERT_TRUE(scheduled.ok());
    ASSERT_EQ(scheduled->size(), 1);
    EXPECT_EQ(scheduled->front(), live);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
    EXPECT_EQ(cancelled->embeddingOutput()->error_info.code(), ErrorCode::CANCELLED);
}

TEST(EmbeddingStreamShutdownTest, ExecutorOwnershipSurvivesCallerTimeout) {
    auto                           caller   = makeStream(1ms);
    std::weak_ptr<EmbeddingStream> lifetime = caller;
    std::promise<void>             complete;
    auto worker = std::async(std::launch::async, [stream = caller, ready = complete.get_future()]() mutable {
        ready.wait();
        stream->updateTensorOutput(torch::ones({1}));
        return stream->embeddingOutput()->error_info.code();
    });
    EXPECT_THROW(caller->waitFinish(), EmbeddingStreamTimeout);
    caller.reset();
    EXPECT_FALSE(lifetime.expired());
    complete.set_value();
    EXPECT_EQ(worker.get(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(lifetime.expired());
}

}  // namespace
}  // namespace rtp_llm
