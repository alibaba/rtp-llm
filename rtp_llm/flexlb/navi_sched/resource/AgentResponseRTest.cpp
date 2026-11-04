#include "rtp_llm/flexlb/navi_sched/data/AgentResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentResponseR.h"
#include "unittest/unittest.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace navi_sched;

namespace {

class CapturePipe : public navi::AsyncPipe {
public:
    CapturePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        values.emplace_back(
            std::dynamic_pointer_cast<AgentResponseToClientData>(data));
        return navi::EC_NONE;
    }

    navi::ErrorCode setDataOverride(const navi::DataPtr &data, bool,
                                    bool = false) override
    {
        return setData(data);
    }

    navi::ErrorCode setEof() override
    {
        ++eofCount;
        return navi::EC_NONE;
    }

    std::vector<AgentResponseToClientDataPtr> values;
    size_t eofCount = 0;
};

class RejectPipe : public CapturePipe {
public:
    navi::ErrorCode setData(const navi::DataPtr &) override
    {
        return navi::EC_ABORT;
    }
};

class BlockingEofPipe : public CapturePipe {
public:
    navi::ErrorCode setEof() override
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _eofEntered = true;
        _condition.notify_all();
        _condition.wait(lock, [this]() { return _releaseEof; });
        return CapturePipe::setEof();
    }

    void waitForEof()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this]() { return _eofEntered; });
    }

    void releaseEof()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _releaseEof = true;
        _condition.notify_all();
    }

private:
    std::mutex _mutex;
    std::condition_variable _condition;
    bool _eofEntered = false;
    bool _releaseEof = false;
};

} // namespace

class AgentResponseRTest : public TESTBASE {
};

TEST_F(AgentResponseRTest, testPublishesMultipleOutputsBeforeTerminal)
{
    AgentResponseR resource;
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(resource.registerRequest(7, pipe));

    NaviSchedRequestTiming timing;
    timing.queueWaitMs = 2.5;
    ASSERT_TRUE(resource.acknowledge(7, timing));
    ASSERT_EQ(0, pipe->eofCount);
    ASSERT_EQ(1, resource.requestCount());
    ASSERT_TRUE(resource.publish(7, {"batch-1-a", "batch-1-b"}, false));
    ASSERT_TRUE(resource.publish(7, {"batch-2"}, true));

    ASSERT_EQ(3, pipe->values.size());
    ASSERT_EQ(AgentResponseKind::ACK, pipe->values[0]->kind);
    ASSERT_DOUBLE_EQ(2.5, pipe->values[0]->requestTiming.queueWaitMs);
    ASSERT_EQ(AgentResponseKind::OUTPUT, pipe->values[1]->kind);
    ASSERT_EQ(std::vector<std::string>({"batch-1-a", "batch-1-b"}),
              pipe->values[1]->frames);
    ASSERT_FALSE(pipe->values[1]->terminal);
    ASSERT_EQ(std::vector<std::string>({"batch-2"}), pipe->values[2]->frames);
    ASSERT_TRUE(pipe->values[2]->terminal);
    ASSERT_EQ(1, pipe->eofCount);
    ASSERT_EQ(0, resource.requestCount());
    ASSERT_FALSE(resource.publish(7, {"late"}, false));
}

TEST_F(AgentResponseRTest, testBuffersOutputUntilAcknowledgement)
{
    AgentResponseR resource;
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(resource.registerRequest(8, pipe));

    ASSERT_TRUE(resource.publish(8, {"first-token"}, false));
    ASSERT_TRUE(resource.publish(8, {"last-token"}, true));
    ASSERT_TRUE(pipe->values.empty());

    ASSERT_TRUE(resource.acknowledge(8, NaviSchedRequestTiming()));
    ASSERT_EQ(3, pipe->values.size());
    EXPECT_EQ(AgentResponseKind::ACK, pipe->values[0]->kind);
    EXPECT_EQ(std::vector<std::string>({"first-token"}), pipe->values[1]->frames);
    EXPECT_EQ(std::vector<std::string>({"last-token"}), pipe->values[2]->frames);
    EXPECT_TRUE(pipe->values[2]->terminal);
    EXPECT_EQ(1, pipe->eofCount);
    EXPECT_EQ(0, resource.requestCount());
}

TEST_F(AgentResponseRTest, testTerminalAcknowledgementClosesWithoutCancellation)
{
    AgentResponseR resource;
    auto pipe = std::make_shared<CapturePipe>();
    std::vector<uint64_t> cancelled;
    resource.setCancelHandler(
        [&cancelled](uint64_t requestId) { cancelled.emplace_back(requestId); });
    ASSERT_TRUE(resource.registerRequest(9, pipe));

    NaviSchedRequestTiming timing;
    timing.executionLatencyMs = 3.5;
    ASSERT_TRUE(resource.acknowledge(9, timing, true));

    ASSERT_EQ(1, pipe->values.size());
    EXPECT_EQ(AgentResponseKind::ACK, pipe->values[0]->kind);
    EXPECT_TRUE(pipe->values[0]->terminal);
    EXPECT_DOUBLE_EQ(3.5, pipe->values[0]->requestTiming.executionLatencyMs);
    EXPECT_EQ(1, pipe->eofCount);
    EXPECT_EQ(0, resource.requestCount());
    EXPECT_TRUE(cancelled.empty());
}

TEST_F(AgentResponseRTest, testTerminalAcknowledgementCancelAndFinishAllAreRaceSafe)
{
    for (uint64_t requestId = 1; requestId <= 256; ++requestId) {
        AgentResponseR resource;
        auto pipe = std::make_shared<CapturePipe>();
        std::atomic<size_t> cancelledCount{0};
        resource.setCancelHandler([&](uint64_t cancelledRequestId) {
            EXPECT_EQ(requestId, cancelledRequestId);
            cancelledCount.fetch_add(1, std::memory_order_relaxed);
        });
        ASSERT_TRUE(resource.registerRequest(requestId, pipe));

        std::atomic<size_t> ready{0};
        std::atomic<bool> start{false};
        const auto waitForStart = [&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        };
        std::thread acknowledgeThread([&]() {
            waitForStart();
            resource.acknowledge(requestId, NaviSchedRequestTiming(), true);
        });
        std::thread cancelThread([&]() {
            waitForStart();
            resource.cancel(requestId, pipe);
        });
        std::thread finishThread([&]() {
            waitForStart();
            resource.finishAll("agent stopped");
        });
        while (ready.load(std::memory_order_acquire) != 3) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        acknowledgeThread.join();
        cancelThread.join();
        finishThread.join();

        ASSERT_EQ(0, resource.requestCount());
        ASSERT_EQ(1, pipe->eofCount);
        ASSERT_LE(cancelledCount.load(std::memory_order_relaxed), 1);
        ASSERT_LE(pipe->values.size(), 1);
        if (!pipe->values.empty()) {
            EXPECT_TRUE(pipe->values.front()->terminal);
            EXPECT_TRUE(AgentResponseKind::ACK == pipe->values.front()->kind ||
                        AgentResponseKind::ERROR == pipe->values.front()->kind);
        }
    }
}

TEST_F(AgentResponseRTest, testRejectedPipeDoesNotAffectAnotherRequest)
{
    AgentResponseR resource;
    auto rejected = std::make_shared<RejectPipe>();
    auto healthy = std::make_shared<CapturePipe>();
    std::vector<uint64_t> cancelled;
    resource.setCancelHandler(
        [&cancelled](uint64_t requestId) { cancelled.emplace_back(requestId); });
    ASSERT_TRUE(resource.registerRequest(1, rejected));
    ASSERT_TRUE(resource.registerRequest(2, healthy));

    EXPECT_FALSE(resource.acknowledge(1, NaviSchedRequestTiming()));
    EXPECT_TRUE(resource.acknowledge(2, NaviSchedRequestTiming()));
    EXPECT_TRUE(resource.publish(2, {"output"}, true));

    EXPECT_EQ(std::vector<uint64_t>({1}), cancelled);
    ASSERT_EQ(2, healthy->values.size());
    EXPECT_EQ(AgentResponseKind::ACK, healthy->values[0]->kind);
    EXPECT_EQ(std::vector<std::string>({"output"}), healthy->values[1]->frames);
    EXPECT_EQ(0, resource.requestCount());
}

TEST_F(AgentResponseRTest, testCancelRequiresMatchingPipeAndNotifiesOnce)
{
    AgentResponseR resource;
    auto pipe = std::make_shared<CapturePipe>();
    auto otherPipe = std::make_shared<CapturePipe>();
    std::vector<uint64_t> cancelled;
    resource.setCancelHandler(
        [&cancelled](uint64_t requestId) { cancelled.emplace_back(requestId); });
    ASSERT_TRUE(resource.registerRequest(11, pipe));

    ASSERT_FALSE(resource.cancel(11, otherPipe));
    ASSERT_EQ(1, resource.requestCount());
    ASSERT_TRUE(resource.cancel(11, pipe));
    ASSERT_EQ(std::vector<uint64_t>({11}), cancelled);
    ASSERT_EQ(1, pipe->eofCount);
    ASSERT_FALSE(resource.cancel(11, pipe));
    ASSERT_EQ(std::vector<uint64_t>({11}), cancelled);
}

TEST_F(AgentResponseRTest, testCancelNotifiesWhenFinishAllRemovesEntry)
{
    AgentResponseR resource;
    auto pipe = std::make_shared<BlockingEofPipe>();
    std::vector<uint64_t> cancelled;
    resource.setCancelHandler(
        [&cancelled](uint64_t requestId) { cancelled.emplace_back(requestId); });
    ASSERT_TRUE(resource.registerRequest(12, pipe));

    bool cancelResult = false;
    std::thread cancelThread(
        [&]() { cancelResult = resource.cancel(12, pipe); });
    pipe->waitForEof();

    std::thread finishThread(
        [&]() { resource.finishAll("agent stopped"); });
    while (resource.requestCount() != 0) {
        std::this_thread::yield();
    }
    pipe->releaseEof();

    cancelThread.join();
    finishThread.join();
    EXPECT_TRUE(cancelResult);
    EXPECT_EQ(std::vector<uint64_t>({12}), cancelled);
    EXPECT_TRUE(pipe->values.empty());
    EXPECT_EQ(1, pipe->eofCount);
    EXPECT_EQ(0, resource.requestCount());
}

TEST_F(AgentResponseRTest, testFinishAllReturnsTerminalErrors)
{
    AgentResponseR resource;
    auto first = std::make_shared<CapturePipe>();
    auto second = std::make_shared<CapturePipe>();
    ASSERT_TRUE(resource.registerRequest(1, first));
    ASSERT_TRUE(resource.registerRequest(2, second));

    resource.finishAll("agent stopped");

    ASSERT_EQ(0, resource.requestCount());
    for (const auto &pipe : {first, second}) {
        ASSERT_EQ(1, pipe->values.size());
        ASSERT_EQ(AgentResponseKind::ERROR, pipe->values[0]->kind);
        ASSERT_TRUE(pipe->values[0]->terminal);
        ASSERT_EQ("agent stopped", pipe->values[0]->errorMessage);
        ASSERT_EQ(1, pipe->eofCount);
    }
    ASSERT_FALSE(resource.registerRequest(3, std::make_shared<CapturePipe>()));
}

TEST_F(AgentResponseRTest, testFailureBeforeAcknowledgementIsTerminal)
{
    AgentResponseR resource;
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(resource.registerRequest(9, pipe));

    ASSERT_TRUE(resource.fail(9, "request rejected"));

    ASSERT_EQ(1, pipe->values.size());
    EXPECT_EQ(AgentResponseKind::ERROR, pipe->values[0]->kind);
    EXPECT_EQ("request rejected", pipe->values[0]->errorMessage);
    EXPECT_TRUE(pipe->values[0]->terminal);
    EXPECT_EQ(1, pipe->eofCount);
    EXPECT_EQ(0, resource.requestCount());
    EXPECT_FALSE(resource.acknowledge(9, NaviSchedRequestTiming()));
}

TEST_F(AgentResponseRTest, testTerminalCancelAndFinishAllAreRaceSafe)
{
    for (uint64_t requestId = 1; requestId <= 256; ++requestId) {
        AgentResponseR resource;
        auto pipe = std::make_shared<CapturePipe>();
        std::atomic<size_t> cancelledCount{0};
        resource.setCancelHandler([&](uint64_t cancelledRequestId) {
            EXPECT_EQ(requestId, cancelledRequestId);
            cancelledCount.fetch_add(1, std::memory_order_relaxed);
        });
        ASSERT_TRUE(resource.registerRequest(requestId, pipe));
        ASSERT_TRUE(resource.acknowledge(requestId, NaviSchedRequestTiming()));

        std::atomic<size_t> ready{0};
        std::atomic<bool> start{false};
        const auto waitForStart = [&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        };
        std::thread publishThread([&]() {
            waitForStart();
            resource.publish(requestId, {"terminal"}, true);
        });
        std::thread cancelThread([&]() {
            waitForStart();
            resource.cancel(requestId, pipe);
        });
        std::thread finishThread([&]() {
            waitForStart();
            resource.finishAll("agent stopped");
        });
        while (ready.load(std::memory_order_acquire) != 3) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        publishThread.join();
        cancelThread.join();
        finishThread.join();

        ASSERT_EQ(0, resource.requestCount());
        ASSERT_EQ(1, pipe->eofCount);
        ASSERT_LE(cancelledCount.load(std::memory_order_relaxed), 1);
        ASSERT_GE(pipe->values.size(), 1);
        ASSERT_LE(pipe->values.size(), 2);
        ASSERT_EQ(AgentResponseKind::ACK, pipe->values.front()->kind);
        if (pipe->values.size() == 2) {
            ASSERT_TRUE(pipe->values.back()->terminal);
        }
    }
}

TEST_F(AgentResponseRTest, testBufferedTerminalAcknowledgeAndCancelAreRaceSafe)
{
    for (uint64_t requestId = 1; requestId <= 256; ++requestId) {
        AgentResponseR resource;
        auto pipe = std::make_shared<CapturePipe>();
        std::atomic<size_t> cancelledCount{0};
        resource.setCancelHandler([&](uint64_t cancelledRequestId) {
            EXPECT_EQ(requestId, cancelledRequestId);
            cancelledCount.fetch_add(1, std::memory_order_relaxed);
        });
        ASSERT_TRUE(resource.registerRequest(requestId, pipe));

        std::atomic<size_t> ready{0};
        std::atomic<bool> start{false};
        const auto waitForStart = [&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        };
        std::thread publishThread([&]() {
            waitForStart();
            resource.publish(requestId, {"terminal"}, true);
        });
        std::thread acknowledgeThread([&]() {
            waitForStart();
            resource.acknowledge(requestId, NaviSchedRequestTiming());
        });
        std::thread cancelThread([&]() {
            waitForStart();
            resource.cancel(requestId, pipe);
        });
        while (ready.load(std::memory_order_acquire) != 3) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        publishThread.join();
        acknowledgeThread.join();
        cancelThread.join();
        resource.finishAll("test cleanup");

        ASSERT_EQ(0, resource.requestCount());
        ASSERT_EQ(1, pipe->eofCount);
        ASSERT_LE(cancelledCount.load(std::memory_order_relaxed), 1);
        ASSERT_LE(pipe->values.size(), 2);
        if (!pipe->values.empty()) {
            ASSERT_EQ(AgentResponseKind::ACK, pipe->values.front()->kind);
        }
        if (pipe->values.size() == 2) {
            ASSERT_TRUE(pipe->values.back()->terminal);
        }
    }
}
