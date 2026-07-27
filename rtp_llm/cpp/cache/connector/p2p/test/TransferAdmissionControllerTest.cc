#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/TransferAdmissionController.h"

namespace rtp_llm {
namespace {

struct FakeWorker {
    std::atomic<bool> transport_stopped{false};
};

TEST(TransferAdmissionControllerTest, TeardownWaitsForDelayedPhysicalCompletion) {
    auto                                    worker       = std::make_shared<FakeWorker>();
    std::weak_ptr<FakeWorker>               worker_owner = worker;
    TransferAdmissionController<FakeWorker> admission(worker);

    auto lease = admission.tryAcquire();
    ASSERT_NE(lease, nullptr);
    ASSERT_EQ(lease->worker().get(), worker.get());

    std::mutex              callback_mutex;
    std::condition_variable callback_cv;
    bool                    callback_started = false;
    bool                    allow_completion = false;
    std::thread callback([lease, &callback_mutex, &callback_cv, &callback_started, &allow_completion]() mutable {
        {
            std::lock_guard<std::mutex> lock(callback_mutex);
            callback_started = true;
            callback_cv.notify_all();
        }
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_cv.wait(lock, [&allow_completion]() { return allow_completion; });
        lease.reset();
    });
    lease.reset();

    {
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_cv.wait(lock, [&callback_started]() { return callback_started; });
    }
    ASSERT_TRUE(admission.close());
    EXPECT_EQ(admission.inflightCount(), 1);
    EXPECT_FALSE(admission.teardown([](FakeWorker& value) {
        value.transport_stopped.store(true, std::memory_order_release);
        return true;
    }));
    EXPECT_FALSE(worker->transport_stopped.load(std::memory_order_acquire));

    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        allow_completion = true;
        callback_cv.notify_all();
    }
    callback.join();

    ASSERT_EQ(admission.inflightCount(), 0);
    ASSERT_TRUE(admission.teardown([](FakeWorker& value) {
        value.transport_stopped.store(true, std::memory_order_release);
        return true;
    }));
    EXPECT_TRUE(worker->transport_stopped.load(std::memory_order_acquire));
    EXPECT_FALSE(worker_owner.expired());
    EXPECT_EQ(admission.tryAcquire(), nullptr);
}

TEST(TransferAdmissionControllerTest, SnapshotAndLeaseAreAtomicWithGateClose) {
    auto                                    worker = std::make_shared<FakeWorker>();
    TransferAdmissionController<FakeWorker> admission(worker);

    auto first = admission.tryAcquire();
    ASSERT_NE(first, nullptr);
    ASSERT_TRUE(admission.close());
    EXPECT_EQ(admission.tryAcquire(), nullptr);
    EXPECT_EQ(first->worker().get(), worker.get());
    first.reset();

    ASSERT_TRUE(admission.teardown([](FakeWorker&) { return true; }));
    ASSERT_TRUE(admission.rebuild([](FakeWorker&) { return true; }));
    ASSERT_TRUE(admission.resume());
    EXPECT_NE(admission.tryAcquire(), nullptr);
}

TEST(TransferAdmissionControllerTest, ReopeningGateDoesNotResumeUntouchedTransport) {
    auto                                    worker = std::make_shared<FakeWorker>();
    TransferAdmissionController<FakeWorker> admission(worker);
    int                                     transport_resume_calls = 0;

    ASSERT_TRUE(admission.close());
    ASSERT_TRUE(admission.resume([&transport_resume_calls](FakeWorker&) {
        ++transport_resume_calls;
        return true;
    }));

    EXPECT_EQ(transport_resume_calls, 0);
    EXPECT_NE(admission.tryAcquire(), nullptr);
}

TEST(TransferAdmissionControllerTest, RebuiltTransportResumesBeforeGateReopens) {
    auto                                    worker = std::make_shared<FakeWorker>();
    TransferAdmissionController<FakeWorker> admission(worker);
    int                                     transport_resume_calls = 0;

    ASSERT_TRUE(admission.close());
    ASSERT_TRUE(admission.teardown([](FakeWorker&) { return true; }));
    ASSERT_TRUE(admission.rebuild([](FakeWorker&) { return true; }));
    ASSERT_TRUE(admission.resume([&transport_resume_calls](FakeWorker&) {
        ++transport_resume_calls;
        return true;
    }));

    EXPECT_EQ(transport_resume_calls, 1);
    EXPECT_NE(admission.tryAcquire(), nullptr);
}

TEST(TransferAdmissionControllerTest, FailedTransportResumeKeepsGateClosed) {
    auto                                    worker = std::make_shared<FakeWorker>();
    TransferAdmissionController<FakeWorker> admission(worker);

    ASSERT_TRUE(admission.close());
    ASSERT_TRUE(admission.teardown([](FakeWorker&) { return true; }));
    ASSERT_TRUE(admission.rebuild([](FakeWorker&) { return true; }));
    EXPECT_FALSE(admission.resume([](FakeWorker&) { return false; }));
    EXPECT_EQ(admission.tryAcquire(), nullptr);

    EXPECT_TRUE(admission.resume([](FakeWorker&) { return true; }));
    EXPECT_NE(admission.tryAcquire(), nullptr);
}

}  // namespace
}  // namespace rtp_llm
