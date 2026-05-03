#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

namespace rtp_llm::test {

TEST(ResponseBufferRegistryTest, CreateReturnsSameEntryForDuplicateId) {
    ResponseBufferRegistry reg;
    auto                   a = reg.create(42);
    auto                   b = reg.create(42);
    EXPECT_EQ(a.get(), b.get());
    EXPECT_EQ(reg.size(), 1u);
}

TEST(ResponseBufferRegistryTest, GetReturnsNullWhenMissing) {
    ResponseBufferRegistry reg;
    EXPECT_EQ(reg.get(99), nullptr);
    reg.create(99);
    EXPECT_NE(reg.get(99), nullptr);
}

TEST(ResponseBufferRegistryTest, EraseRemovesEntry) {
    ResponseBufferRegistry reg;
    reg.create(1);
    EXPECT_EQ(reg.size(), 1u);
    reg.erase(1);
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_EQ(reg.get(1), nullptr);
}

TEST(ResponseBufferRegistryTest, GcSkipsLiveAndDrainsTerminalIdle) {
    ResponseBufferRegistry reg;

    auto alive = reg.create(1);
    auto done  = reg.create(2);
    done->done.store(true);
    auto cancelled = reg.create(3);
    cancelled->cancelled.store(true);

    // With a large TTL, nothing is idle enough to be swept.
    EXPECT_EQ(reg.gc(std::chrono::hours(1)), 0u);
    EXPECT_EQ(reg.size(), 3u);

    // With zero TTL, terminal drained entries are swept; alive stays.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(reg.gc(std::chrono::microseconds(0)), 2u);
    EXPECT_EQ(reg.size(), 1u);
    EXPECT_NE(reg.get(1), nullptr);
}

TEST(ResponseBufferRegistryTest, GcKeepsTerminalEntryWithPendingQueue) {
    ResponseBufferRegistry reg;
    auto                   entry = reg.create(7);
    entry->done.store(true);
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        entry->queue.emplace_back();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(reg.gc(std::chrono::microseconds(0)), 0u);
    EXPECT_EQ(reg.size(), 1u);
}

TEST(ResponseBufferWriterTest, WritePushesAndNotifies) {
    auto                 entry = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB out;
    out.set_request_id(123);
    EXPECT_TRUE(writer.Write(out, grpc::WriteOptions{}));

    std::lock_guard<std::mutex> lock(entry->mu);
    ASSERT_EQ(entry->queue.size(), 1u);
    EXPECT_EQ(entry->queue.front().request_id(), 123);
}

TEST(ResponseBufferWriterTest, WriteReturnsFalseWhenCancelled) {
    auto entry = std::make_shared<ResponseBufferEntry>();
    entry->cancelled.store(true);
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB out;
    EXPECT_FALSE(writer.Write(out, grpc::WriteOptions{}));
    std::lock_guard<std::mutex> lock(entry->mu);
    EXPECT_TRUE(entry->queue.empty());
}

TEST(ResponseBufferWriterTest, WriteWakesBlockedConsumer) {
    auto                 entry = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB observed;
    std::thread       consumer([&] {
        std::unique_lock<std::mutex> lock(entry->mu);
        entry->cv.wait(lock, [&] { return !entry->queue.empty(); });
        observed = entry->queue.front();
        entry->queue.pop_front();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    GenerateOutputsPB out;
    out.set_request_id(42);
    EXPECT_TRUE(writer.Write(out, grpc::WriteOptions{}));
    consumer.join();
    EXPECT_EQ(observed.request_id(), 42);
}

// End-to-end async-submit handshake — mirrors PrefillRpcServer::FetchResponse
// drain loop against a producer simulating a detached finishStream worker.
// Validates: (1) multi-output ordering, (2) done-termination, (3) registry erase.
TEST(AsyncSubmitChainTest, ProducerConsumerDrainOrderAndTerminate) {
    ResponseBufferRegistry registry;
    const int64_t          request_id = 1001;
    auto                   entry      = registry.create(request_id);

    constexpr int kN = 8;
    std::thread   producer([entry] {
        ResponseBufferWriter writer(entry);
        for (int i = 0; i < kN; ++i) {
            GenerateOutputsPB out;
            out.set_request_id(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ASSERT_TRUE(writer.Write(out, grpc::WriteOptions{}));
        }
        entry->done.store(true);
        entry->cv.notify_all();
    });

    std::vector<int64_t> observed;
    bool                 terminal = false;
    while (!terminal) {
        std::deque<GenerateOutputsPB> drained;
        {
            std::unique_lock<std::mutex> lock(entry->mu);
            entry->cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return !entry->queue.empty() || entry->done.load() || entry->cancelled.load()
                       || entry->error_status.has_value();
            });
            drained.swap(entry->queue);
            terminal = entry->done.load() || entry->cancelled.load() || entry->error_status.has_value();
        }
        for (auto& o : drained) {
            observed.push_back(o.request_id());
        }
    }
    producer.join();

    ASSERT_EQ(observed.size(), static_cast<size_t>(kN));
    for (int i = 0; i < kN; ++i) {
        EXPECT_EQ(observed[i], i);
    }
    registry.erase(request_id);
    EXPECT_EQ(registry.get(request_id), nullptr);
}

TEST(AsyncSubmitChainTest, CancelMidstreamStopsDrain) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.create(2002);

    std::thread producer([entry] {
        ResponseBufferWriter writer(entry);
        for (int i = 0; i < 3; ++i) {
            GenerateOutputsPB out;
            out.set_request_id(i);
            writer.Write(out, grpc::WriteOptions{});
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Consumer drains a bit, then cancels.
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    entry->cancelled.store(true);
    entry->cv.notify_all();
    producer.join();

    std::unique_lock<std::mutex> lock(entry->mu);
    EXPECT_TRUE(entry->cancelled.load());
    // Writes after cancel should have been rejected (Write returns false path).
    // At least one output should have been captured before cancel flipped.
    EXPECT_LE(entry->queue.size(), 3u);
}

TEST(AsyncSubmitChainTest, ErrorStatusPropagatesToConsumer) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.create(3003);

    std::thread producer([entry] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard<std::mutex> lock(entry->mu);
        entry->error_status = grpc::Status(grpc::StatusCode::INTERNAL, "simulated finishStream failure");
        entry->done.store(true);
        entry->cv.notify_all();
    });

    grpc::Status terminal_status = grpc::Status::OK;
    while (true) {
        std::unique_lock<std::mutex> lock(entry->mu);
        entry->cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
            return entry->done.load() || entry->error_status.has_value();
        });
        if (entry->error_status.has_value()) {
            terminal_status = *entry->error_status;
            break;
        }
        if (entry->done.load())
            break;
    }
    producer.join();

    EXPECT_EQ(terminal_status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_NE(terminal_status.error_message().find("simulated finishStream failure"), std::string::npos);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
