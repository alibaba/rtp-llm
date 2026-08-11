#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"

#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

int64_t futureUnixMs(std::chrono::milliseconds delta = 5s) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch() + delta)
        .count();
}

std::string makeToken(const std::string& unique_id, int64_t deadline_unix_ms) {
    auto token = makeRemoteLoadAllocationToken("test-owner", unique_id, deadline_unix_ms);
    EXPECT_TRUE(token.ok()) << token.status();
    return token.ok() ? *token : std::string();
}

TEST(RemoteLoadFenceTest, SealBeforeBeginRejectsLateOperation) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              token    = makeToken("allocation-a", deadline);

    ASSERT_TRUE(registry.sealAndWait(token, deadline, std::chrono::steady_clock::now() + 100ms).ok());

    auto operation = registry.begin(token, deadline);
    ASSERT_FALSE(operation.ok());
    EXPECT_EQ(operation.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(RemoteLoadFenceTest, SealWaitsForLastOperationAlias) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              token    = makeToken("allocation-a", deadline);
    auto                    operation = registry.begin(token, deadline);
    ASSERT_TRUE(operation.ok());

    auto retained_by_transfer = *operation;
    operation                 = absl::UnknownError("released local operation result");

    auto waiter = std::async(std::launch::async, [&registry, token, deadline]() {
        return registry.sealAndWait(token, deadline, std::chrono::steady_clock::now() + 2s);
    });

    EXPECT_EQ(waiter.wait_for(30ms), std::future_status::timeout);
    retained_by_transfer.reset();
    ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(waiter.get().ok());
}

TEST(RemoteLoadFenceTest, DuplicateOperationForTokenIsRejected) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              token    = makeToken("allocation-a", deadline);
    auto                    first = registry.begin(token, deadline);
    ASSERT_TRUE(first.ok());

    auto duplicate = registry.begin(token, deadline);
    ASSERT_FALSE(duplicate.ok());
    EXPECT_EQ(duplicate.status().code(), absl::StatusCode::kAlreadyExists);
}

TEST(RemoteLoadFenceTest, DifferentTokensDoNotBlockEachOther) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              first_token  = makeToken("allocation-a", deadline);
    const auto              second_token = makeToken("allocation-b", deadline);
    auto                    first = registry.begin(first_token, deadline);
    auto                    second = registry.begin(second_token, deadline);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());

    first = absl::UnknownError("released first operation result");
    EXPECT_TRUE(
        registry.sealAndWait(first_token, deadline, std::chrono::steady_clock::now() + 100ms).ok());

    auto wait_for_second = registry.sealAndWait(
        second_token, deadline, std::chrono::steady_clock::now() + 10ms);
    EXPECT_EQ(wait_for_second.code(), absl::StatusCode::kDeadlineExceeded);
}

TEST(RemoteLoadFenceTest, ExpiredOperationIsRejectedAndTombstoneCanBePruned) {
    RemoteLoadFenceRegistry registry;
    const auto              expired = futureUnixMs(-1ms);
    const auto              token   = makeToken("allocation-a", expired);

    auto operation = registry.begin(token, expired);
    ASSERT_FALSE(operation.ok());
    EXPECT_EQ(operation.status().code(), absl::StatusCode::kDeadlineExceeded);

    ASSERT_TRUE(registry.sealAndWait(
        token, expired, std::chrono::steady_clock::now() + 100ms).ok());
    EXPECT_EQ(registry.entryCountForTest(), 1);
    registry.pruneExpired();
    EXPECT_EQ(registry.entryCountForTest(), 0);
}

TEST(RemoteLoadFenceTest, LocallyTranslatedExpiryDoesNotTrustRemoteWallClock) {
    RemoteLoadFenceRegistry registry;
    const auto              remote_deadline = futureUnixMs(-1ms);
    const auto              token           = makeToken("clock-skew", remote_deadline);
    const auto              local_expiry    = std::chrono::steady_clock::now() + 100ms;

    auto operation = registry.begin(token, remote_deadline, local_expiry);
    ASSERT_TRUE(operation.ok()) << operation.status();
    operation = absl::UnknownError("released operation");
    EXPECT_TRUE(registry
                    .sealAndWait(token,
                                 remote_deadline,
                                 local_expiry,
                                 RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal,
                                 local_expiry)
                    .ok());
}

TEST(RemoteLoadFenceTest, UnseenTombstoneUsesClockSafeRetentionWindow) {
    RemoteLoadFenceRegistry registry;
    const auto              remote_deadline = futureUnixMs(-1ms);
    const auto              token           = makeToken("delayed-operation", remote_deadline);
    const auto              local_expiry    = std::chrono::steady_clock::now() + 100ms;

    ASSERT_TRUE(registry
                    .sealAndWait(token,
                                 remote_deadline,
                                 std::chrono::steady_clock::now() + 20ms,
                                 RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal,
                                 local_expiry)
                    .ok());
    auto delayed_operation = registry.begin(token, remote_deadline, local_expiry);
    ASSERT_FALSE(delayed_operation.ok());
    EXPECT_EQ(delayed_operation.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(RemoteLoadFenceTest, ExpiredTokenCannotBeRevivedWithNewDeadline) {
    RemoteLoadFenceRegistry registry;
    const auto              expired = futureUnixMs(-1ms);
    const auto              token   = makeToken("allocation-a", expired);

    ASSERT_TRUE(registry.sealAndWait(
        token, expired, std::chrono::steady_clock::now() + 100ms).ok());
    registry.pruneExpired();

    const auto refreshed_deadline = futureUnixMs();
    auto       operation          = registry.begin(token, refreshed_deadline);
    ASSERT_FALSE(operation.ok());
    EXPECT_EQ(operation.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(RemoteLoadFenceTest, PruneCannotInvalidateWaitingEntry) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs(50ms);
    const auto              token    = makeToken("allocation-a", deadline);
    auto                    operation = registry.begin(token, deadline);
    ASSERT_TRUE(operation.ok());
    auto retained_by_transfer = *operation;
    operation                 = absl::UnknownError("released local operation result");

    auto waiter = std::async(std::launch::async, [&registry, token, deadline]() {
        return registry.sealAndWait(token, deadline, std::chrono::steady_clock::now() + 2s);
    });
    EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);

    std::this_thread::sleep_until(std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline + 1)));
    retained_by_transfer.reset();
    registry.pruneExpired();

    ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(waiter.get().ok());
}

TEST(RemoteLoadFenceTest, SealTimeoutKeepsTokenSealedForRetry) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              token    = makeToken("allocation-a", deadline);
    auto                    operation = registry.begin(token, deadline);
    ASSERT_TRUE(operation.ok());
    auto retained_by_transfer = *operation;
    operation                 = absl::UnknownError("released local operation result");

    auto first_wait =
        registry.sealAndWait(token, deadline, std::chrono::steady_clock::now() + 1ms);
    EXPECT_EQ(first_wait.code(), absl::StatusCode::kDeadlineExceeded);

    auto late_begin = registry.begin(token, deadline);
    ASSERT_FALSE(late_begin.ok());
    EXPECT_EQ(late_begin.status().code(), absl::StatusCode::kFailedPrecondition);

    retained_by_transfer.reset();
    EXPECT_TRUE(
        registry.sealAndWait(token, deadline, std::chrono::steady_clock::now() + 100ms).ok());
}

TEST(RemoteLoadFenceTest, ActiveExpiredEntryIsNotPruned) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs(20ms);
    const auto              token    = makeToken("allocation-a", deadline);
    auto                    operation = registry.begin(token, deadline);
    ASSERT_TRUE(operation.ok());

    std::this_thread::sleep_until(std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline + 1)));
    registry.pruneExpired();
    EXPECT_EQ(registry.entryCountForTest(), 1);

    operation = absl::UnknownError("released operation");
    registry.pruneExpired();
    EXPECT_EQ(registry.entryCountForTest(), 0);
}

TEST(RemoteLoadFenceTest, OperationCanOutliveRegistry) {
    RemoteLoadFenceRegistry::Operation operation;
    {
        auto       registry = std::make_unique<RemoteLoadFenceRegistry>();
        const auto deadline = futureUnixMs();
        const auto token    = makeToken("allocation-a", deadline);
        auto       result   = registry->begin(token, deadline);
        ASSERT_TRUE(result.ok());
        operation = *result;
    }

    operation.reset();
}

TEST(RemoteLoadFenceTest, TokenParserRejectsMalformedAndMismatchedDeadline) {
    EXPECT_FALSE(remoteLoadAllocationDeadline("").ok());
    EXPECT_FALSE(remoteLoadAllocationDeadline("missing-separator").ok());
    EXPECT_FALSE(remoteLoadAllocationDeadline("0:allocation-a").ok());
    EXPECT_FALSE(remoteLoadAllocationDeadline("abc:allocation-a").ok());
    EXPECT_FALSE(remoteLoadAllocationDeadline("123:").ok());

    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              token    = makeToken("allocation-a", deadline);
    auto                    operation = registry.begin(token, deadline + 1);
    ASSERT_FALSE(operation.ok());
    EXPECT_EQ(operation.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(RemoteLoadFenceTest, ValidatesAllocationOwnerExactly) {
    const auto deadline = futureUnixMs();
    const auto token = makeRemoteLoadAllocationToken("server:request", "payload:1", deadline);
    ASSERT_TRUE(token.ok());

    EXPECT_TRUE(validateRemoteLoadAllocationOwner(*token, "server:request").ok());
    EXPECT_EQ(validateRemoteLoadAllocationOwner(*token, "serve").code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(validateRemoteLoadAllocationOwner(*token, "server").code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(validateRemoteLoadAllocationOwner(*token, "").code(), absl::StatusCode::kInvalidArgument);
}

TEST(RemoteLoadFenceTest, ValidatesIpv6OwnerWithoutDelimiterAmbiguity) {
    const auto deadline = futureUnixMs();
    const auto token = makeRemoteLoadAllocationToken("[2001:db8::1]:1234", "request:1", deadline);
    ASSERT_TRUE(token.ok());

    EXPECT_TRUE(validateRemoteLoadAllocationOwner(*token, "[2001:db8::1]:1234").ok());
    EXPECT_EQ(validateRemoteLoadAllocationOwner(*token, "[2001:db8::1]").code(),
              absl::StatusCode::kFailedPrecondition);
}

TEST(RemoteLoadFenceTest, RejectingUnseenTokenDoesNotCreateTombstone) {
    RemoteLoadFenceRegistry registry;
    const auto              deadline = futureUnixMs();
    const auto              token    = makeToken("unseen", deadline);

    const auto seal_status = registry.sealAndWait(token,
                                                  deadline,
                                                  std::chrono::steady_clock::now() + 100ms,
                                                  RemoteLoadFenceRegistry::UnseenTokenPolicy::Reject);
    EXPECT_EQ(seal_status.code(), absl::StatusCode::kNotFound);
    EXPECT_EQ(registry.entryCountForTest(), 0);
    EXPECT_TRUE(registry.begin(token, deadline).ok());
}

TEST(RemoteLoadFenceTest, UnseenTokenIsQuiescedAfterItsImmutableDeadline) {
    RemoteLoadFenceRegistry registry;
    const auto deadline = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count()
                          - 1;
    const auto token = makeToken("expired-unseen", deadline);

    EXPECT_TRUE(registry
                    .sealAndWait(token,
                                 deadline,
                                 std::chrono::steady_clock::now(),
                                 RemoteLoadFenceRegistry::UnseenTokenPolicy::Reject)
                    .ok());
    EXPECT_EQ(registry.entryCountForTest(), 0);
    EXPECT_EQ(registry.begin(token, deadline).status().code(), absl::StatusCode::kDeadlineExceeded);
}

}  // namespace
}  // namespace rtp_llm
