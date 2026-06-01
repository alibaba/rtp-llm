#include <chrono>
#include <functional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

namespace rtp_llm::test {

TEST(ResponseBufferRegistryTest, CreateReturnsSameEntryForDuplicateId) {
    ResponseBufferRegistry registry;
    auto                   first  = registry.createOrGet(42);
    auto                   second = registry.createOrGet(42);
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(registry.size(), 1u);
}

TEST(ResponseBufferRegistryTest, ReserveReturnsNullForDuplicateId) {
    ResponseBufferRegistry registry;
    auto                   first  = registry.reserve(42);
    auto                   second = registry.reserve(42);
    EXPECT_NE(first, nullptr);
    EXPECT_EQ(second, nullptr);
    EXPECT_EQ(registry.size(), 1u);
}

TEST(ResponseBufferRegistryTest, GetReturnsNullWhenMissing) {
    ResponseBufferRegistry registry;
    EXPECT_EQ(registry.get(99), nullptr);
    registry.createOrGet(99);
    EXPECT_NE(registry.get(99), nullptr);
}

TEST(ResponseBufferRegistryTest, EraseRemovesEntry) {
    ResponseBufferRegistry registry;
    registry.createOrGet(1);
    EXPECT_EQ(registry.size(), 1u);
    registry.erase(1);
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_EQ(registry.get(1), nullptr);
}

TEST(ResponseBufferRegistryTest, GcSkipsLiveAndDrainsTerminalIdle) {
    ResponseBufferRegistry registry;

    auto alive = registry.createOrGet(1);
    (void)alive;
    auto done = registry.createOrGet(2);
    done->done.store(true);
    auto cancelled = registry.createOrGet(3);
    cancelled->cancelled.store(true);

    EXPECT_EQ(registry.gc(std::chrono::hours(1)), 0u);
    EXPECT_EQ(registry.size(), 3u);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(registry.gc(std::chrono::microseconds(0)), 2u);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_NE(registry.get(1), nullptr);
}

TEST(ResponseBufferRegistryTest, GcSweepsTerminalEntryWithPendingQueueAfterTtl) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.createOrGet(7);
    entry->done.store(true);
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        entry->queue.emplace_back();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(registry.gc(std::chrono::microseconds(0)), 1u);
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_EQ(registry.get(7), nullptr);
}

TEST(ResponseBufferRegistryTest, CancelAllMarksEntriesAndInvokesProducers) {
    ResponseBufferRegistry registry;
    auto                   first  = registry.createOrGet(1);
    auto                   second = registry.createOrGet(2);
    int                    first_cancel_count = 0;
    int                    second_cancel_count = 0;

    {
        std::lock_guard<std::mutex> lock(first->mu);
        first->cancel_producer = [&] { ++first_cancel_count; };
    }
    {
        std::lock_guard<std::mutex> lock(second->mu);
        second->cancel_producer = [&] { ++second_cancel_count; };
    }

    registry.cancelAll();
    registry.cancelAll();

    EXPECT_TRUE(first->cancelled.load());
    EXPECT_TRUE(second->cancelled.load());
    EXPECT_EQ(first_cancel_count, 1);
    EXPECT_EQ(second_cancel_count, 1);
}

TEST(ResponseBufferWriterTest, WritePushesAndNotifies) {
    auto                 entry = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB output;
    output.set_request_id(123);
    EXPECT_TRUE(writer.Write(output, grpc::WriteOptions{}));

    std::lock_guard<std::mutex> lock(entry->mu);
    ASSERT_EQ(entry->queue.size(), 1u);
    EXPECT_EQ(entry->queue.front().request_id(), 123);
}

TEST(ResponseBufferWriterTest, WriteReturnsFalseWhenCancelled) {
    auto entry = std::make_shared<ResponseBufferEntry>();
    entry->cancelled.store(true);
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB output;
    EXPECT_FALSE(writer.Write(output, grpc::WriteOptions{}));
    std::lock_guard<std::mutex> lock(entry->mu);
    EXPECT_TRUE(entry->queue.empty());
}

TEST(ResponseBufferEntryTest, CancelProducerCanBeInvokedByOwner) {
    auto entry    = std::make_shared<ResponseBufferEntry>();
    bool observed = false;
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        entry->cancel_producer = [&] { observed = true; };
    }

    std::function<void()> cancel_producer;
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        cancel_producer = entry->cancel_producer;
    }
    ASSERT_TRUE(static_cast<bool>(cancel_producer));
    cancel_producer();
    EXPECT_TRUE(observed);
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
    GenerateOutputsPB output;
    output.set_request_id(42);
    EXPECT_TRUE(writer.Write(output, grpc::WriteOptions{}));
    consumer.join();
    EXPECT_EQ(observed.request_id(), 42);
}

TEST(AsyncSubmitChainTest, ProducerConsumerDrainOrderAndTerminate) {
    ResponseBufferRegistry registry;
    const int64_t          request_id = 1001;
    auto                   entry      = registry.createOrGet(request_id);

    constexpr int kOutputCount = 8;
    std::thread   producer([entry] {
        ResponseBufferWriter writer(entry);
        for (int i = 0; i < kOutputCount; ++i) {
            GenerateOutputsPB output;
            output.set_request_id(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ASSERT_TRUE(writer.Write(output, grpc::WriteOptions{}));
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
        for (auto& output : drained) {
            observed.push_back(output.request_id());
        }
    }
    producer.join();

    ASSERT_EQ(observed.size(), static_cast<size_t>(kOutputCount));
    for (int i = 0; i < kOutputCount; ++i) {
        EXPECT_EQ(observed[i], i);
    }
    registry.erase(request_id);
    EXPECT_EQ(registry.get(request_id), nullptr);
}

TEST(AsyncSubmitChainTest, ErrorStatusPropagatesToConsumer) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.createOrGet(3003);

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
        if (entry->done.load()) {
            break;
        }
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
