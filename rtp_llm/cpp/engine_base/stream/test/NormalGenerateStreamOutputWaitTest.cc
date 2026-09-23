// Regression suite for the nextOutput() wait/publish protocol.
//
// The defect these tests pin. The consumer used to clear its wake flag and test the queue OUTSIDE the
// coordinator mutex, then lock and park, while the producer stored the flag and notified WITHOUT ever
// taking that mutex. A publish landing after the consumer's predicate had returned false but before the
// wait was registered therefore notified nobody, and delivery slipped to the end of the wait slice.
// That is a lost-NOTIFICATION (ordering) defect, not an atomic-visibility defect, so shortening the
// slice would only have hidden it. It is repaired by publishing the flag under the coordinator mutex and
// by clearing it, re-testing the queue and registering the wait inside one critical section.
//
// Lock order asserted throughout: stream mutex_ -> output_wait_->mu -> the output queue's own lock.
//
// These tests drive the REAL NormalGenerateStream, the REAL private enqueueGenerateOutput() publish path
// and the REAL nextOutput() wait. The target compiles with -fno-access-control like its siblings, which
// is what makes the coordinator mutex reachable as a deterministic test seam -- so the ordering property
// is proved by lock serialization rather than by a wall-clock threshold, and no hook is added to
// production code. Where a bound is asserted at all it is deliberately generous: it can only trip on an
// indefinite block, never on the slice under test.

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

constexpr int64_t kSentinel = 999;

// Generous on purpose. The wait slice under test is 100 ms; a bound this loose can only be exceeded by
// an indefinitely blocked consumer, which is the failure these tests exist to catch. It is never the
// evidence for the ordering property -- C1 proves that by serialization instead.
constexpr auto kBoundedReturn = std::chrono::seconds(20);

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
    ModelConfig                     model_config;
    RuntimeConfig                   runtime_config;
    ResourceContext                 resource_context;
    model_config.max_seq_len = 2048;
    return std::make_shared<NormalGenerateStream>(
        generate_input, model_config, runtime_config, resource_context, nullptr);
}

// The production publish path: capacity check, queue push, then the wake publication under test.
void publish(const std::shared_ptr<NormalGenerateStream>& stream, int64_t marker) {
    GenerateOutputs outputs;
    outputs.request_id = marker;
    stream->enqueueGenerateOutput(std::move(outputs));
}

// ---------------------------------------------------------------------------
// C1 -- publication cannot slip between the predicate check and wait registration
// ---------------------------------------------------------------------------
TEST(NormalGenerateStreamOutputWait, C1PublishIsSerializedAgainstWaitRegistration) {
    auto stream = makeStream();

    // Open the registration window FIRST, then start the producer. The window is exactly the span in
    // which a consumer owns the coordinator mutex -- from clearing its wake flag, through evaluating
    // the predicate, until cv.wait_for atomically releases the mutex and registers. Owning it here
    // reproduces that whole span without needing a hook inside the production wait.
    std::unique_lock<std::mutex> registration_window(stream->output_wait_->mu);

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

    // Guaranteed to be reached: the producer does not need the coordinator mutex to get here, so this
    // spin cannot deadlock and it removes thread-startup jitter from the measurement below.
    while (!at_publish.load()) {
        std::this_thread::yield();
    }

    // The discriminating assertion. Publication is serialized on the coordinator mutex, so while this
    // window is open the producer CANNOT finish: the wait below must time out. Under the defect the
    // producer never touches that mutex and completes in microseconds, so the wait returns immediately.
    // Both directions are therefore determined by the protocol, not by how generous the bound is -- the
    // bound only has to exceed a few uncontended queue operations, which 500 ms does by orders of
    // magnitude, while a fixed publication could never satisfy it at all.
    bool completed_inside_window;
    {
        std::unique_lock<std::mutex> done_lock(done_mu);
        completed_inside_window =
            done_cv.wait_for(done_lock, std::chrono::milliseconds(500), [&] { return publish_returned; });
    }
    EXPECT_FALSE(completed_inside_window)
        << "publish completed while a waiter owned the predicate-to-park window, so a notify could land "
           "after the predicate returned false and before the wait was registered";

    // A consumer that arrives inside the window must still be served once it closes, whichever thread
    // takes the mutex first: if the producer wins, the consumer's predicate observes the queued output;
    // if the consumer wins, it parks and the producer's notify reaches it.
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
    // Exactly once: the published output was consumed, not duplicated or left behind.
    EXPECT_TRUE(stream->generate_outputs_queue_.isEmpty());
}

// ---------------------------------------------------------------------------
// C2 -- exactly-once ordered delivery, and an arriving output needs no slice
// ---------------------------------------------------------------------------
TEST(NormalGenerateStreamOutputWait, C2AlreadyQueuedOutputNeverRegistersAWait) {
    auto stream = makeStream();
    publish(stream, 7);

    // Hold the coordinator mutex for the whole call. If nextOutput() tried to register a wait for an
    // output that is already queued it would block here and the test would time out, so a pass proves
    // the already-queued path does not touch the wait at all -- no slice of delay is even possible.
    std::lock_guard<std::mutex> hold(stream->output_wait_->mu);
    auto                        result = stream->nextOutput();

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(7, result.value().request_id);
}

TEST(NormalGenerateStreamOutputWait, C2OutputPublishedDuringWaitIsDeliveredExactlyOnceInOrder) {
    auto stream = makeStream();

    for (int64_t marker = 1; marker <= 3; ++marker) {
        ErrorResult<GenerateOutputs> result(ErrorCode::UNKNOWN_ERROR, "unset");
        std::thread                  consumer([&] { result = stream->nextOutput(); });

        // Let the consumer reach its wait, then publish through the real path.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        publish(stream, marker);
        consumer.join();

        ASSERT_TRUE(result.ok()) << "round " << marker;
        EXPECT_EQ(marker, result.value().request_id) << "delivery was not ordered";
    }

    // Nothing may be delivered twice. Finish the stream and drain it: the caller must see the terminal
    // condition rather than a repeat of any earlier marker. (OUTPUT_QUEUE_IS_EMPTY is not reachable for
    // a single consumer -- the wait loop only exits on an error, on FINISHED, or on a non-empty queue --
    // so FINISHED is the terminal result this path can produce.)
    EXPECT_TRUE(stream->generate_outputs_queue_.isEmpty());
    stream->generate_status_->status = StreamState::FINISHED;
    auto extra                       = stream->nextOutput();
    EXPECT_FALSE(extra.ok());
    EXPECT_EQ(ErrorCode::FINISHED, extra.status().code());
}

// ---------------------------------------------------------------------------
// C3 -- cancellation of a request that produces no output is bounded
// ---------------------------------------------------------------------------
TEST(NormalGenerateStreamOutputWait, C3CancelledBeforeWaitingReturnsImmediately) {
    auto stream = makeStream();

    // The predicate is evaluated before any wait is registered, so an already-cancelled request must
    // not park at all. Holding the mutex makes that deterministic: a park would block and time out.
    std::lock_guard<std::mutex> hold(stream->output_wait_->mu);
    auto                        result = stream->nextOutput([] { return true; });

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(ErrorCode::CANCELLED, result.status().code());
}

TEST(NormalGenerateStreamOutputWait, C3CancelledWhileParkedWithNoOutputIsBounded) {
    auto stream = makeStream();

    std::atomic<bool> cancel{false};
    auto              began = std::chrono::steady_clock::now();

    ErrorResult<GenerateOutputs> result(ErrorCode::UNKNOWN_ERROR, "unset");
    std::thread                  consumer([&] { result = stream->nextOutput([&] { return cancel.load(); }); });

    // This is the shape that used to wedge: a queued/no-output request (a non-streaming decode that
    // only emits once it finishes, or a request still waiting for admission) whose client is gone.
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    cancel.store(true);
    consumer.join();
    auto elapsed = std::chrono::steady_clock::now() - began;

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(ErrorCode::CANCELLED, result.status().code());
    EXPECT_LT(elapsed, kBoundedReturn) << "cancellation was not bounded by the wait slice";
}

// ---------------------------------------------------------------------------
// C4 -- one terminal result, original cause preserved, no double release
// ---------------------------------------------------------------------------
TEST(NormalGenerateStreamOutputWait, C4TerminalCausePrecedenceAndFirstErrorWins) {
    // An error already recorded beats a concurrent cancellation: the loop tests hasError() first, so the
    // caller sees the real cause rather than a cancellation that raced with it.
    {
        auto stream = makeStream();
        stream->reportError(ErrorCode::MALLOC_FAILED, "oom");
        auto result = stream->nextOutput([] { return true; });
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::MALLOC_FAILED, result.status().code());
    }

    // A queued output beats a concurrent cancellation: real work already produced is not discarded.
    {
        auto stream = makeStream();
        publish(stream, 5);
        auto result = stream->nextOutput([] { return true; });
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(5, result.value().request_id);
    }

    // A finished stream with nothing queued reports FINISHED, not an empty-queue error.
    {
        auto stream = makeStream();
        stream->generate_status_->status = StreamState::FINISHED;
        auto result                      = stream->nextOutput();
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::FINISHED, result.status().code());
    }

    // A request deadline expires while parked with no output.
    {
        auto stream          = makeStream(/*timeout_ms=*/1);
        stream->begin_time_us_ = nowUs() - 60LL * 1000 * 1000;  // 60 s ago: already past the deadline
        auto result          = stream->nextOutput();
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::GENERATE_TIMEOUT, result.status().code());
    }

    // The FIRST terminal cause is preserved: a deadline that expires afterwards must not replace the
    // earlier error, so a caller never sees its real failure reported as a timeout.
    {
        auto stream            = makeStream(/*timeout_ms=*/1);
        stream->begin_time_us_ = nowUs() - 60LL * 1000 * 1000;
        stream->reportError(ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
        stream->checkTimeout();  // already past the deadline: fires, but must not overwrite the cause
        auto result = stream->nextOutput();
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(ErrorCode::OUTPUT_QUEUE_FULL, result.status().code());
    }
}

// ---------------------------------------------------------------------------
// C5 -- a full queue is a bounded terminal error; the producer must not block
// ---------------------------------------------------------------------------
TEST(NormalGenerateStreamOutputWait, C5FullQueueIsATerminalErrorWithoutBlockingTheProducer) {
    auto stream = makeStream();
    stream->generate_outputs_queue_.setCapacity(2);

    publish(stream, 1);
    publish(stream, 2);

    // The third publish must take the error branch and RETURN. A plain queue push would block inside the
    // queue's own condition wait while holding the queue lock -- which is precisely the lock inversion
    // the documented order forbids -- so this call returning at all is the evidence.
    publish(stream, 3);
    ASSERT_TRUE(stream->hasError());

    // The terminal error supersedes the buffered outputs, and repeated calls keep reporting the same
    // cause: no double delivery and no second, different terminal result.
    auto first  = stream->nextOutput();
    auto second = stream->nextOutput();
    ASSERT_FALSE(first.ok());
    EXPECT_EQ(ErrorCode::OUTPUT_QUEUE_FULL, first.status().code());
    ASSERT_FALSE(second.ok());
    EXPECT_EQ(ErrorCode::OUTPUT_QUEUE_FULL, second.status().code());
}

// ---------------------------------------------------------------------------
// C6 -- overloads, subclasses and coordinator lifetime
// ---------------------------------------------------------------------------

// A subclass that implements only the no-argument overload, i.e. one that does NOT support cancellation.
// The base default deliberately ignores the predicate, so this documents that the overload's existence
// is not a guarantee that every stream subclass honours cancellation.
//
// API trap worth knowing: overriding nextOutput() here HIDES the base's cancellation-aware overload, so
// the predicate form is unreachable through this static type (a subclass would need
// `using GenerateStream::nextOutput;` to re-expose it). Production callers hold a GenerateStreamPtr, so
// the test below dispatches through the base type exactly as LocalRpcServer::pollStreamOutput does.
class NoCancelSupportStream: public GenerateStream {
public:
    using GenerateStream::GenerateStream;

    ErrorResult<GenerateOutputs> nextOutput() override {
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
    ModelConfig                     model_config;
    RuntimeConfig                   runtime_config;
    ResourceContext                 resource_context;
    model_config.max_seq_len = 2048;
    // Held as the BASE type on purpose: that is how the RPC layer calls it, and it is the only static
    // type through which the cancellation-aware overload is visible for this subclass.
    GenerateStreamPtr stream = std::make_shared<NoCancelSupportStream>(
        generate_input, model_config, runtime_config, resource_context, nullptr);

    auto result = stream->nextOutput([] { return true; });
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(kSentinel, result.value().request_id);
}

TEST(NormalGenerateStreamOutputWait, C6CoordinatorOutlivesStreamDestruction) {
    auto stream     = makeStream();
    auto coordinator = stream->output_wait_;  // shared ownership, as for the stream's own cv_

    stream.reset();  // runs ~NormalGenerateStream, which publishes a courtesy wake

    // The coordinator is still usable after the stream is gone, so a wake published during destruction
    // cannot touch freed memory. This is the ownership half of the contract; the destructor notify is a
    // courtesy only and is not a substitute for callers holding a stream reference across nextOutput().
    ASSERT_NE(nullptr, coordinator);
    {
        std::lock_guard<std::mutex> lock(coordinator->mu);
        coordinator->wake.store(true);
    }
    coordinator->cv.notify_all();
}

}  // namespace
}  // namespace rtp_llm
