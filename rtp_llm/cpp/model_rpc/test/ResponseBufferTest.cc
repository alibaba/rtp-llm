#include <chrono>
#include <functional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

namespace rtp_llm::test {

namespace {

std::shared_ptr<ResponseBufferEntry> reserveAndPublish(ResponseBufferRegistry& registry, int64_t request_id) {
    auto entry = registry.reserve(request_id);
    EXPECT_NE(entry, nullptr);
    registry.publish(request_id, entry);
    return entry;
}

}  // namespace

TEST(ResponseBufferRegistryTest, ReserveRejectsDuplicateId) {
    ResponseBufferRegistry registry;
    auto                   first  = registry.reserve(42);
    auto                   second = registry.reserve(42);
    EXPECT_NE(first, nullptr);
    EXPECT_EQ(second, nullptr);
    EXPECT_EQ(registry.size(), 1u);
}

TEST(ResponseBufferRegistryTest, PendingEntryIsInvisibleUntilPublished) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.reserve(99);

    auto pending_claim = registry.claim(99);
    EXPECT_EQ(pending_claim.status, ResponseBufferRegistry::ClaimStatus::NOT_FOUND);
    EXPECT_EQ(pending_claim.entry, nullptr);

    registry.publish(99, entry);
    auto ready_claim = registry.claim(99);
    EXPECT_EQ(ready_claim.status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
    EXPECT_EQ(ready_claim.entry, entry);
}

TEST(ResponseBufferRegistryTest, ResponseCanOnlyBeClaimedOnce) {
    ResponseBufferRegistry registry;
    auto                   entry = reserveAndPublish(registry, 1);

    auto first_claim  = registry.claim(1);
    auto second_claim = registry.claim(1);
    EXPECT_EQ(first_claim.status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
    EXPECT_EQ(first_claim.entry, entry);
    EXPECT_EQ(second_claim.status, ResponseBufferRegistry::ClaimStatus::ALREADY_CLAIMED);
    EXPECT_EQ(second_claim.entry, nullptr);
}

TEST(ResponseBufferRegistryTest, ConcurrentFetchClaimsHaveSingleWinner) {
    ResponseBufferRegistry registry;
    reserveAndPublish(registry, 1);
    std::atomic<int> successful_claims{0};

    std::vector<std::thread> claimers;
    for (int i = 0; i < 16; ++i) {
        claimers.emplace_back([&] {
            if (registry.claim(1).status == ResponseBufferRegistry::ClaimStatus::SUCCESS) {
                successful_claims.fetch_add(1);
            }
        });
    }
    for (auto& claimer : claimers) {
        claimer.join();
    }

    EXPECT_EQ(successful_claims.load(), 1);
}

TEST(ResponseBufferRegistryTest, AbortRequiresPendingStateAndMatchingEntryIdentity) {
    ResponseBufferRegistry registry;
    auto                   old_entry = registry.reserve(1);
    registry.abort(1, old_entry);

    auto new_entry = reserveAndPublish(registry, 1);
    EXPECT_ANY_THROW(registry.abort(1, old_entry));
    EXPECT_ANY_THROW(registry.abort(1, new_entry));
    EXPECT_EQ(registry.size(), 1u);
    auto claim = registry.claim(1);
    EXPECT_EQ(claim.status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
    EXPECT_EQ(claim.entry, new_entry);
}

TEST(ResponseBufferRegistryTest, InvalidStateTransitionsReportErrors) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.reserve(10);

    EXPECT_ANY_THROW(registry.releaseClaim(10, entry));
    registry.publish(10, entry);
    EXPECT_ANY_THROW(registry.publish(10, entry));
    EXPECT_ANY_THROW(registry.abort(10, entry));

    registry.finish(10, entry, grpc::Status::OK);
    EXPECT_ANY_THROW(registry.finish(10, entry, grpc::Status::OK));
    ASSERT_EQ(registry.claim(10).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
    registry.releaseClaim(10, entry);

    auto released_entry = reserveAndPublish(registry, 11);
    ASSERT_EQ(registry.claim(11).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
    registry.releaseClaim(11, released_entry);
    EXPECT_ANY_THROW(registry.releaseClaim(11, released_entry));
    registry.finish(11, released_entry, grpc::Status::OK);
}

TEST(ResponseBufferRegistryTest, GcSkipsLiveAndDrainsTerminalIdle) {
    ResponseBufferRegistry registry;

    auto alive = reserveAndPublish(registry, 1);
    (void)alive;
    auto done = reserveAndPublish(registry, 2);
    registry.finish(2, done, grpc::Status::OK);
    auto cancelled = reserveAndPublish(registry, 3);
    cancelled->cancel();
    registry.finish(3, cancelled, grpc::Status(grpc::StatusCode::CANCELLED, "cancelled"));

    EXPECT_EQ(registry.gc(std::chrono::hours(1)), 0u);
    EXPECT_EQ(registry.size(), 3u);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(registry.gc(std::chrono::microseconds(0)), 2u);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.claim(1).entry, alive);
}

TEST(ResponseBufferRegistryTest, GcSweepsTerminalEntryWithPendingQueueAfterTtl) {
    ResponseBufferRegistry registry;
    auto                   entry = reserveAndPublish(registry, 7);
    GenerateOutputsPB      output;
    ASSERT_TRUE(entry->write(output));
    registry.finish(7, entry, grpc::Status::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(registry.gc(std::chrono::microseconds(0)), 1u);
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_EQ(registry.claim(7).status, ResponseBufferRegistry::ClaimStatus::NOT_FOUND);
}

TEST(ResponseBufferRegistryTest, GcDoesNotRemovePendingEntry) {
    ResponseBufferRegistry registry;
    auto                   entry = registry.reserve(7);
    GenerateOutputsPB      output;
    ASSERT_TRUE(entry->write(output));
    registry.finish(7, entry, grpc::Status::OK);

    EXPECT_EQ(registry.gc(std::chrono::microseconds(0)), 0u);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(entry->waitAndDrain(std::chrono::milliseconds(0)).outputs.size(), 1u);
    registry.publish(7, entry);
}

TEST(ResponseBufferRegistryTest, GcDoesNotRemoveClaimedEntry) {
    ResponseBufferRegistry registry;
    auto                   entry = reserveAndPublish(registry, 8);
    GenerateOutputsPB      output;
    ASSERT_TRUE(entry->write(output));
    ASSERT_EQ(registry.claim(8).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
    registry.finish(8, entry, grpc::Status::OK);

    EXPECT_EQ(registry.gc(std::chrono::microseconds(0)), 0u);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(entry->waitAndDrain(std::chrono::milliseconds(0)).outputs.size(), 1u);
    registry.releaseClaim(8, entry);
    EXPECT_EQ(registry.size(), 0u);
}

TEST(ResponseBufferRegistryTest, CancelledFetchKeepsIdReservedUntilProducerFinishes) {
    ResponseBufferRegistry registry;
    auto                   entry = reserveAndPublish(registry, 9);
    ASSERT_EQ(registry.claim(9).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);

    entry->cancel();
    registry.releaseClaim(9, entry);
    EXPECT_EQ(registry.reserve(9), nullptr);
    EXPECT_EQ(registry.size(), 1u);

    registry.finish(9, entry, grpc::Status(grpc::StatusCode::CANCELLED, "cancelled"));
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_NE(registry.reserve(9), nullptr);
}

TEST(ResponseBufferRegistryTest, ProducerFinishAndFetchReleaseCanRace) {
    for (int i = 0; i < 64; ++i) {
        ResponseBufferRegistry registry;
        auto                   entry = reserveAndPublish(registry, i);
        ASSERT_EQ(registry.claim(i).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);
        std::atomic<bool> finish_ok{false};
        std::atomic<bool> release_ok{false};

        std::thread producer([&] {
            registry.finish(i, entry, grpc::Status::OK);
            finish_ok.store(true);
        });
        std::thread consumer([&] {
            registry.releaseClaim(i, entry);
            release_ok.store(true);
        });
        producer.join();
        consumer.join();

        EXPECT_TRUE(finish_ok.load());
        EXPECT_TRUE(release_ok.load());
        EXPECT_EQ(registry.size(), 0u);
    }
}

TEST(ResponseBufferRegistryTest, CancelAllMarksEntriesAndInvokesProducers) {
    ResponseBufferRegistry registry;
    auto                   first               = registry.reserve(1);
    auto                   second              = reserveAndPublish(registry, 2);
    int                    first_cancel_count  = 0;
    int                    second_cancel_count = 0;

    first->installCancelProducer([&] { ++first_cancel_count; });
    second->installCancelProducer([&] { ++second_cancel_count; });

    registry.cancelAll();
    registry.cancelAll();

    EXPECT_TRUE(first->isCancelled());
    EXPECT_TRUE(second->isCancelled());
    EXPECT_EQ(first_cancel_count, 1);
    EXPECT_EQ(second_cancel_count, 1);
}

TEST(ResponseBufferWriterTest, WritePushesAndNotifies) {
    auto                 entry = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB output;
    output.set_request_id(123);
    EXPECT_TRUE(writer.Write(output, grpc::WriteOptions{}));

    auto drained = entry->waitAndDrain(std::chrono::milliseconds(0));
    ASSERT_EQ(drained.outputs.size(), 1u);
    EXPECT_EQ(drained.outputs.front().request_id(), 123);
}

TEST(ResponseBufferWriterTest, PreservesAllQueuedOutputs) {
    constexpr int        kOutputCount = 1001;
    auto                 entry        = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);

    for (int request_id = 0; request_id < kOutputCount; ++request_id) {
        GenerateOutputsPB output;
        output.set_request_id(request_id);
        ASSERT_TRUE(writer.Write(output, grpc::WriteOptions{}));
    }

    auto drained = entry->waitAndDrain(std::chrono::milliseconds(0));
    ASSERT_EQ(drained.outputs.size(), kOutputCount);
    for (int request_id = 0; request_id < kOutputCount; ++request_id) {
        EXPECT_EQ(drained.outputs[request_id].request_id(), request_id);
    }
}

TEST(ResponseBufferWriterTest, WriteReturnsFalseWhenCancelled) {
    auto entry = std::make_shared<ResponseBufferEntry>();
    entry->cancel();
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB output;
    EXPECT_FALSE(writer.Write(output, grpc::WriteOptions{}));
    EXPECT_TRUE(entry->waitAndDrain(std::chrono::milliseconds(0)).outputs.empty());
}

TEST(ResponseBufferWriterTest, WriteAfterFinishReportsError) {
    ResponseBufferRegistry registry;
    auto                   entry = reserveAndPublish(registry, 123);
    ResponseBufferWriter   writer(entry);
    registry.finish(123, entry, grpc::Status::OK);

    GenerateOutputsPB output;
    output.set_request_id(123);
    EXPECT_ANY_THROW(writer.Write(output, grpc::WriteOptions{}));
}

TEST(ResponseBufferWriterTest, QueueOverflowCancelsRequestWithoutDroppingQueuedOutputs) {
    auto                 entry = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);
    int                  cancel_count = 0;
    entry->installCancelProducer([&] { ++cancel_count; });

    GenerateOutputsPB output;
    output.set_request_id(123);
    for (size_t i = 0; i < ResponseBufferEntry::kMaxQueueSize; ++i) {
        ASSERT_TRUE(writer.Write(output, grpc::WriteOptions{}));
    }
    EXPECT_FALSE(writer.Write(output, grpc::WriteOptions{}));

    EXPECT_TRUE(entry->isCancelled());
    EXPECT_EQ(cancel_count, 1);
    auto drained = entry->waitAndDrain(std::chrono::milliseconds(0));
    EXPECT_TRUE(drained.terminal);
    EXPECT_EQ(drained.terminal_status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(drained.outputs.size(), ResponseBufferEntry::kMaxQueueSize);
}

TEST(ResponseBufferEntryTest, CancelInvokesProducerOnce) {
    auto entry             = std::make_shared<ResponseBufferEntry>();
    int  cancel_count      = 0;
    int  late_cancel_count = 0;

    entry->installCancelProducer([&] { ++cancel_count; });
    entry->cancel();
    entry->cancel();

    EXPECT_EQ(cancel_count, 1);
    entry->installCancelProducer([&] { ++late_cancel_count; });
    EXPECT_EQ(late_cancel_count, 1);
}

TEST(ResponseBufferEntryTest, CancelAfterFinishPreservesTerminalStatus) {
    ResponseBufferRegistry registry;

    auto success = reserveAndPublish(registry, 1);
    registry.finish(1, success, grpc::Status::OK);
    success->cancel();
    auto success_result = success->waitAndDrain(std::chrono::milliseconds(0));
    EXPECT_FALSE(success->isCancelled());
    EXPECT_TRUE(success_result.terminal);
    EXPECT_TRUE(success_result.terminal_status.ok());

    auto error = reserveAndPublish(registry, 2);
    registry.finish(2, error, grpc::Status(grpc::StatusCode::INTERNAL, "producer failed"));
    error->cancel();
    auto error_result = error->waitAndDrain(std::chrono::milliseconds(0));
    EXPECT_FALSE(error->isCancelled());
    EXPECT_TRUE(error_result.terminal);
    EXPECT_EQ(error_result.terminal_status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(error_result.terminal_status.error_message(), "producer failed");
}

TEST(ResponseBufferWriterTest, WriteWakesBlockedConsumer) {
    auto                 entry = std::make_shared<ResponseBufferEntry>();
    ResponseBufferWriter writer(entry);

    GenerateOutputsPB observed;
    std::thread       consumer([&] {
        auto drained = entry->waitAndDrain(std::chrono::seconds(1));
        ASSERT_EQ(drained.outputs.size(), 1u);
        observed = drained.outputs.front();
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
    auto                   entry      = reserveAndPublish(registry, request_id);
    ASSERT_EQ(registry.claim(request_id).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);

    constexpr int kOutputCount = 8;
    std::thread   producer([&registry, entry, request_id] {
        ResponseBufferWriter writer(entry);
        for (int i = 0; i < kOutputCount; ++i) {
            GenerateOutputsPB output;
            output.set_request_id(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ASSERT_TRUE(writer.Write(output, grpc::WriteOptions{}));
        }
        registry.finish(request_id, entry, grpc::Status::OK);
    });

    std::vector<int64_t> observed;
    bool                 terminal = false;
    while (!terminal) {
        auto drained = entry->waitAndDrain(std::chrono::milliseconds(100));
        terminal     = drained.terminal;
        for (auto& output : drained.outputs) {
            observed.push_back(output.request_id());
        }
    }
    producer.join();

    ASSERT_EQ(observed.size(), static_cast<size_t>(kOutputCount));
    for (int i = 0; i < kOutputCount; ++i) {
        EXPECT_EQ(observed[i], i);
    }
    registry.releaseClaim(request_id, entry);
    EXPECT_EQ(registry.claim(request_id).status, ResponseBufferRegistry::ClaimStatus::NOT_FOUND);
}

TEST(AsyncSubmitChainTest, ErrorStatusPropagatesToConsumer) {
    ResponseBufferRegistry registry;
    auto                   entry = reserveAndPublish(registry, 3003);
    ASSERT_EQ(registry.claim(3003).status, ResponseBufferRegistry::ClaimStatus::SUCCESS);

    std::thread producer([&registry, entry] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        registry.finish(3003, entry, grpc::Status(grpc::StatusCode::INTERNAL, "simulated finishStream failure"));
    });

    grpc::Status terminal_status = grpc::Status::OK;
    while (true) {
        auto drained = entry->waitAndDrain(std::chrono::milliseconds(100));
        if (drained.terminal) {
            terminal_status = drained.terminal_status;
            break;
        }
    }
    producer.join();

    EXPECT_EQ(terminal_status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_NE(terminal_status.error_message().find("simulated finishStream failure"), std::string::npos);
}

TEST(ResponseBufferEntryTest, WriteRejectsOnByteLimit) {
    auto entry = std::make_shared<ResponseBufferEntry>();
    // Temporarily lower the byte limit for testing
    auto saved_limit                    = ResponseBufferEntry::kMaxQueueBytes;
    ResponseBufferEntry::kMaxQueueBytes = 64 * 1024;  // 64KB for test

    GenerateOutputsPB large_output;
    large_output.set_request_id(1);
    large_output.mutable_error_info()->set_error_message(std::string(64 * 1024, 'x'));

    EXPECT_FALSE(entry->write(large_output));
    EXPECT_TRUE(entry->isCancelled());
    EXPECT_EQ(entry->droppedCount(), 1u);

    ResponseBufferEntry::kMaxQueueBytes = saved_limit;  // restore
}

TEST(ResponseBufferEntryTest, WriteOverflowIncrementsDroppedCount) {
    auto entry = std::make_shared<ResponseBufferEntry>();

    GenerateOutputsPB output;
    output.set_request_id(1);
    for (size_t i = 0; i < ResponseBufferEntry::kMaxQueueSize; ++i) {
        ASSERT_TRUE(entry->write(output));
    }

    EXPECT_FALSE(entry->write(output));
    EXPECT_TRUE(entry->isCancelled());
    EXPECT_EQ(entry->droppedCount(), 1u);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
