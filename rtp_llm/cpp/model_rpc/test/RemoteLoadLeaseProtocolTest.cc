#include "rtp_llm/cpp/model_rpc/CacheTransferLease.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadLeaseRetainer.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

RemoteLoadLeaseRetainer::Config protocolTestConfig() {
    return RemoteLoadLeaseRetainer::Config{
        4,
        1ms,
        5ms,
        1s,
        1,
    };
}

template<typename Predicate>
bool waitUntil(Predicate&& predicate, std::chrono::milliseconds timeout = 2s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return predicate();
        }
        std::this_thread::yield();
    }
    return true;
}

TEST(RemoteLoadLeaseProtocolTest, TimeoutRetainsBothSidesUntilLastTransferBufferReleases) {
    const auto deadline_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() + 1min).time_since_epoch())
            .count();
    const auto token = makeRemoteLoadAllocationToken("test-owner", "late-transfer", deadline_unix_ms);
    ASSERT_TRUE(token.ok()) << token.status();

    RemoteLoadFenceRegistry fence;
    auto                    operation = fence.begin(*token, deadline_unix_ms);
    ASSERT_TRUE(operation.ok()) << operation.status();

    auto source_lease      = std::make_shared<int>(1);
    auto destination_lease = std::make_shared<KVCacheResource>();
    std::weak_ptr<int>             weak_source      = source_lease;
    std::weak_ptr<KVCacheResource> weak_destination = destination_lease;

    auto lifetime = makeCacheTransferLifetime(destination_lease, *operation);
    ASSERT_NE(lifetime, nullptr);
    char transfer_byte = 0;
    auto late_buffer   = makeCacheTransferAddress(lifetime, &transfer_byte);
    ASSERT_NE(late_buffer, nullptr);
    lifetime.reset();
    operation->reset();

    std::atomic<int> quiesce_attempts{0};
    auto quiesce = [&]() {
        ++quiesce_attempts;
        return fence.sealAndWait(*token, deadline_unix_ms, std::chrono::steady_clock::now() + 2ms).ok();
    };

    RemoteLoadLeaseRetainer source_retainer(protocolTestConfig());
    RemoteLoadLeaseRetainer destination_retainer(protocolTestConfig());
    auto source_ticket = source_retainer.reserve(*token, source_lease, quiesce);
    auto destination_ticket = destination_retainer.reserve(*token, destination_lease, quiesce);
    ASSERT_TRUE(source_ticket.ok()) << source_ticket.status();
    ASSERT_TRUE(destination_ticket.ok()) << destination_ticket.status();
    ASSERT_TRUE((*source_ticket)->markStarted());
    ASSERT_TRUE((*destination_ticket)->markStarted());
    source_lease.reset();
    destination_lease.reset();
    source_ticket->reset();
    destination_ticket->reset();

    ASSERT_TRUE(waitUntil([&]() { return quiesce_attempts.load() >= 2; }));
    EXPECT_FALSE(weak_source.expired());
    EXPECT_FALSE(weak_destination.expired());

    late_buffer.reset();
    EXPECT_TRUE(waitUntil([&]() { return weak_source.expired() && weak_destination.expired(); }));
    EXPECT_TRUE(source_retainer.stop(100ms));
    EXPECT_TRUE(destination_retainer.stop(100ms));
}

TEST(RemoteLoadLeaseProtocolTest, QuiesceBeforeBeginRejectsLateTransfer) {
    const auto deadline_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() + 1min).time_since_epoch())
            .count();
    const auto token = makeRemoteLoadAllocationToken("test-owner", "sealed-before-begin", deadline_unix_ms);
    ASSERT_TRUE(token.ok()) << token.status();

    RemoteLoadFenceRegistry  fence;
    RemoteLoadLeaseRetainer retainer(protocolTestConfig());
    std::atomic<int>         quiesce_attempts{0};
    auto                     lease      = std::make_shared<int>(1);
    std::weak_ptr<int>       weak_lease = lease;
    auto ticket = retainer.reserve(*token, lease, [&]() {
        ++quiesce_attempts;
        return fence.sealAndWait(*token, deadline_unix_ms, std::chrono::steady_clock::now() + 100ms).ok();
    });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();
    ticket->reset();

    ASSERT_TRUE(waitUntil([&]() { return weak_lease.expired(); }));
    EXPECT_EQ(quiesce_attempts.load(), 1);
    auto late_operation = fence.begin(*token, deadline_unix_ms);
    ASSERT_FALSE(late_operation.ok());
    EXPECT_EQ(late_operation.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_TRUE(retainer.stop(100ms));
}

}  // namespace
}  // namespace rtp_llm
