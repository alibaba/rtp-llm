#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include <limits>
#include <thread>

#include "rtp_llm/cpp/cache/connector/p2p/PrefillResultStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class PrefillResultStoreTest: public ::testing::Test {
protected:
    PrefillResultStore store_{10000, 60000};

    PrefillResultStore::Data data() {
        PrefillResultStore::Data result;
        result.has_first_token = true;
        result.first_token_id  = 42;
        return result;
    }

    int64_t deadline() {
        return currentTimeMs() + 5000;
    }

    void connectResources(P2PConnectorResourceStore& resources) {
        resources.setOnRequestRegistered([&](const std::string& key, int64_t hold, int64_t request) {
            return store_.registerRequest(key, hold, request);
        });
        resources.setOnRequestAcquired(
            [&](const std::string& key, int64_t transfer) { return store_.beginTransfer(key, transfer); });
        resources.setOnRequestReleased(
            [&](const std::string& key, int64_t, int64_t request) { store_.seal(key, request); });
    }

    std::shared_ptr<MockMeta> meta(int64_t request_deadline) {
        auto value = std::make_shared<MockMeta>();
        value->setUniqueKey("request");
        value->setRequestId(1);
        value->setDeadlineMs(request_deadline);
        return value;
    }
};

TEST_F(PrefillResultStoreTest, NotifyBeforeRegistrationAndConsumeOnce) {
    store_.notify("request", deadline(), data());
    ASSERT_TRUE(store_.registerRequest("request", deadline(), deadline()));
    ASSERT_TRUE(store_.beginTransfer("request", deadline()));
    PrefillResultStore::Data result;
    ASSERT_TRUE(store_.waitAndFill("request", deadline(), result));
    EXPECT_TRUE(result.has_first_token);
    EXPECT_EQ(result.first_token_id, 42);
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result));
    store_.notify("request", deadline(), data());
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result));
}

TEST_F(PrefillResultStoreTest, WaitBeforeNotify) {
    auto waiter = std::async(std::launch::async, [&]() {
        PrefillResultStore::Data result;
        return store_.waitAndFill("request", deadline(), result) && result.first_token_id == 42;
    });
    store_.notify("request", deadline(), data());
    EXPECT_TRUE(waiter.get());
}

TEST_F(PrefillResultStoreTest, SealBeforeNotifyRejectsLateResultAndRegistration) {
    store_.seal("request", deadline());
    EXPECT_FALSE(store_.registerRequest("request", deadline(), deadline()));
    EXPECT_FALSE(store_.beginTransfer("request", deadline()));
    store_.notify("request", deadline(), data());
    PrefillResultStore::Data result;
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result));
    EXPECT_FALSE(store_.entries_.at("request").data.has_value());
}

TEST_F(PrefillResultStoreTest, SealDropsPayloadAndWakesWaiter) {
    store_.notify("ready", deadline(), data());
    store_.seal("ready", deadline());
    PrefillResultStore::Data result;
    EXPECT_FALSE(store_.waitAndFill("ready", deadline(), result));
    auto waiter = std::async(std::launch::async, [&]() {
        PrefillResultStore::Data result;
        return store_.waitAndFill("pending", deadline(), result);
    });
    store_.seal("pending", deadline());
    EXPECT_FALSE(waiter.get());
}

TEST_F(PrefillResultStoreTest, CancelledWaitReturnsWithoutConsuming) {
    store_.notify("request", deadline(), data());
    PrefillResultStore::Data result;
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result, []() { return true; }));
    EXPECT_TRUE(store_.waitAndFill("request", deadline(), result));
}

TEST_F(PrefillResultStoreTest, ExpiryDropsPayloadAndRetainsTombstone) {
    store_.notify("request", deadline(), data());
    ASSERT_TRUE(store_.registerRequest("request", deadline(), deadline()));
    ASSERT_TRUE(store_.beginTransfer("request", currentTimeMs() - 1));
    store_.checkTimeout();
    PrefillResultStore::Data result;
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result));
    store_.notify("request", deadline(), data());
    EXPECT_FALSE(store_.entries_.at("request").data.has_value());
    store_.entries_.at("request").terminal_deadline_ms = currentTimeMs() - 1;
    store_.checkTimeout();
    EXPECT_EQ(store_.entries_.count("request"), 0);
}

TEST_F(PrefillResultStoreTest, EarlyResultUsesHoldDeadline) {
    PrefillResultStore short_hold(10000, 0);
    short_hold.notify("request", deadline(), data());
    PrefillResultStore::Data result;
    EXPECT_FALSE(short_hold.waitAndFill("request", deadline(), result));
    EXPECT_FALSE(short_hold.entries_.at("request").data.has_value());
}

TEST_F(PrefillResultStoreTest, AcquisitionUpdatesTransferDeadline) {
    store_.notify("request", deadline(), data());
    ASSERT_TRUE(store_.registerRequest("request", deadline(), deadline()));
    const auto transfer_deadline = currentTimeMs() + 1000;
    ASSERT_TRUE(store_.beginTransfer("request", transfer_deadline));
    EXPECT_EQ(store_.entries_.at("request").deadline_ms, transfer_deadline);
    PrefillResultStore::Data result;
    EXPECT_TRUE(store_.waitAndFill("request", deadline(), result));
}

TEST_F(PrefillResultStoreTest, ResourceCancellationSealsResultAfterSteal) {
    P2PConnectorResourceStore resources(nullptr, 10000);
    connectResources(resources);
    auto meta = std::make_shared<MockMeta>();
    meta->setUniqueKey("request");
    meta->setRequestId(1);
    meta->setDeadlineMs(deadline());
    ASSERT_TRUE(resources.addResource(meta, std::make_shared<KVCacheResource>()));
    auto resource = resources.waitAndStealResource("request", deadline());
    ASSERT_NE(resource, nullptr);
    resources.markTerminal("request", deadline());
    store_.notify("request", deadline(), data());
    PrefillResultStore::Data result;
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result));
}

TEST_F(PrefillResultStoreTest, ResourceExpirySealsResult) {
    P2PConnectorResourceStore resources(nullptr, 10000, 0);
    connectResources(resources);
    store_.notify("request", deadline(), data());
    auto meta = std::make_shared<MockMeta>();
    meta->setUniqueKey("request");
    meta->setRequestId(1);
    meta->setDeadlineMs(deadline());
    ASSERT_TRUE(resources.addResource(meta, std::make_shared<KVCacheResource>()));
    resources.checkTimeout();
    EXPECT_TRUE(resources.isMarkedCancelled("request"));
    PrefillResultStore::Data result;
    EXPECT_FALSE(store_.waitAndFill("request", deadline(), result));
}

TEST_F(PrefillResultStoreTest, InitUsesMillisecondsAndCleansExpiredEntries) {
    PrefillResultStore results(100, 60000);
    results.seal("expired", currentTimeMs() - 1);
    ASSERT_TRUE(results.init());
    EXPECT_EQ(results.cleanup_thread_->_loopInterval, 100000);
    const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    bool empty = false;
    while (std::chrono::steady_clock::now() < stop) {
        {
            std::lock_guard<std::mutex> lock(results.mutex_);
            empty = results.entries_.empty();
        }
        if (empty) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(empty);
}

TEST_F(PrefillResultStoreTest, ResourceAndResultUseIdenticalDeadlines) {
    P2PConnectorResourceStore resources(nullptr, 10000, 7200000, 1234);
    connectResources(resources);
    const auto request_deadline = currentTimeMs() + 7200000;
    ASSERT_TRUE(resources.addResource(meta(request_deadline), std::make_shared<KVCacheResource>()));
    auto entry = resources.resource_map_.at("request");
    EXPECT_EQ(store_.entries_.at("request").deadline_ms, entry->deadline_ms);
    EXPECT_EQ(store_.entries_.at("request").terminal_deadline_ms, entry->request_deadline_ms);
    EXPECT_LE(entry->deadline_ms, currentTimeMs() + 3600000);

    const auto transfer_deadline = currentTimeMs() + 1000;
    ASSERT_NE(resources.waitAndStealResource("request", transfer_deadline), nullptr);
    EXPECT_EQ(entry->deadline_ms, transfer_deadline);
    EXPECT_EQ(store_.entries_.at("request").deadline_ms, entry->deadline_ms);
}

TEST_F(PrefillResultStoreTest, MissingBusinessDeadlineUsesResourceFallbackOnce) {
    P2PConnectorResourceStore resources(nullptr, 10000, 60000, 1234);
    connectResources(resources);
    ASSERT_TRUE(resources.addResource(meta(0), std::make_shared<KVCacheResource>()));
    auto entry = resources.resource_map_.at("request");
    EXPECT_EQ(store_.entries_.at("request").deadline_ms, entry->deadline_ms);
    EXPECT_EQ(store_.entries_.at("request").terminal_deadline_ms, entry->request_deadline_ms);
    EXPECT_EQ(entry->deadline_ms, entry->request_deadline_ms);
    const auto request_deadline = entry->request_deadline_ms;
    ASSERT_NE(resources.waitAndStealResource("request", deadline()), nullptr);
    EXPECT_EQ(entry->deadline_ms, request_deadline);
    EXPECT_EQ(store_.entries_.at("request").deadline_ms, request_deadline);
    resources.markTerminal("request", request_deadline);
    EXPECT_EQ(store_.entries_.at("request").terminal_deadline_ms, request_deadline);
}

TEST_F(PrefillResultStoreTest, CleanupDuringAcquisitionPreservesResult) {
    for (const bool notify_before : {true, false}) {
        SCOPED_TRACE(notify_before);
        const auto request_deadline = deadline();
        P2PConnectorResourceStore resources(nullptr, 10000, 1000);
        connectResources(resources);
        ASSERT_TRUE(resources.addResource(meta(request_deadline), std::make_shared<KVCacheResource>()));
        const auto hold_deadline = resources.resource_map_.at("request")->deadline_ms;
        const auto transfer_deadline = hold_deadline + 1000;
        if (notify_before) {
            store_.notify("request", request_deadline, data());
        }
        resources.setOnRequestAcquired([&](const std::string& key, int64_t transfer) {
            // Force cleanup between successful hold validation and deadline publication.
            store_.checkTimeout(hold_deadline + 1);
            EXPECT_FALSE(store_.entries_.at(key).terminal);
            if (!notify_before) {
                store_.notify(key, request_deadline, data());
            }
            return store_.beginTransfer(key, transfer);
        });
        auto entry = resources.waitAndStealResource("request", transfer_deadline);
        ASSERT_NE(entry, nullptr);
        store_.checkTimeout(hold_deadline + 1);
        PrefillResultStore::Data result;
        EXPECT_TRUE(store_.waitAndFill("request", transfer_deadline, result));
        EXPECT_EQ(result.first_token_id, 42);
        EXPECT_EQ(store_.entries_.at("request").deadline_ms, entry->deadline_ms);
        store_.entries_.clear();
    }
}

TEST_F(PrefillResultStoreTest, HoldExpiryBeforeAcquisitionRejectsBothSides) {
    P2PConnectorResourceStore resources(nullptr, 10000, 0);
    connectResources(resources);
    const auto request_deadline = deadline();
    ASSERT_TRUE(resources.addResource(meta(request_deadline), std::make_shared<KVCacheResource>()));
    store_.notify("request", request_deadline, data());
    // No resource cleanup thread has run; acquisition itself must check hold expiry.
    EXPECT_EQ(resources.waitAndStealResource("request", request_deadline), nullptr);
    EXPECT_TRUE(resources.isMarkedCancelled("request"));
    EXPECT_TRUE(store_.entries_.at("request").terminal);
    EXPECT_FALSE(store_.entries_.at("request").data.has_value());
    EXPECT_FALSE(store_.beginTransfer("request", request_deadline));
}

TEST_F(PrefillResultStoreTest, TerminalResultRejectsResourceRegistration) {
    P2PConnectorResourceStore resources(nullptr, 10000);
    connectResources(resources);
    const auto request_deadline = deadline();
    store_.seal("request", request_deadline);
    EXPECT_FALSE(resources.addResource(meta(request_deadline), std::make_shared<KVCacheResource>()));
    EXPECT_TRUE(resources.resource_map_.empty());
    EXPECT_TRUE(resources.isMarkedCancelled("request"));
}

TEST_F(PrefillResultStoreTest, TerminalResultRejectsResourceAcquisition) {
    P2PConnectorResourceStore resources(nullptr, 10000);
    connectResources(resources);
    const auto request_deadline = deadline();
    ASSERT_TRUE(resources.addResource(meta(request_deadline), std::make_shared<KVCacheResource>()));
    store_.seal("request", request_deadline);
    EXPECT_EQ(resources.waitAndStealResource("request", request_deadline), nullptr);
    EXPECT_TRUE(resources.resource_map_.empty());
    EXPECT_TRUE(resources.isMarkedCancelled("request"));
}

TEST_F(PrefillResultStoreTest, ExpiredBusinessDeadlineDoesNotRestartTerminalTtl) {
    const auto expired = currentTimeMs() - 1000;
    store_.seal("sealed", expired);
    store_.notify("notified", expired, data());
    EXPECT_LE(store_.entries_.at("sealed").terminal_deadline_ms, currentTimeMs());
    EXPECT_LE(store_.entries_.at("notified").terminal_deadline_ms, currentTimeMs());
    store_.checkTimeout();
    EXPECT_TRUE(store_.entries_.empty());
}

TEST_F(PrefillResultStoreTest, MissingAndInfiniteBusinessDeadlineUseConfiguredFallback) {
    PrefillResultStore results(10000, 60000, 1234);
    const auto before = currentTimeMs();
    results.seal("missing", 0);
    results.seal("infinite", std::numeric_limits<int64_t>::max());
    const auto after = currentTimeMs();
    for (const auto* key : {"missing", "infinite"}) {
        EXPECT_GE(results.entries_.at(key).terminal_deadline_ms, before + 1234);
        EXPECT_LE(results.entries_.at(key).terminal_deadline_ms, after + 1234);
    }
}

TEST_F(PrefillResultStoreTest, FillsTokenReuseAndMtpPayload) {
    auto input             = data();
    input.total_reuse_len  = 10;
    input.local_reuse_len  = 1;
    input.remote_reuse_len = 2;
    input.memory_reuse_len = 3;
    input.disk_reuse_len   = 4;
    input.propose_tokens   = {7, 8};
    input.position_ids     = {11, 12};
    input.propose_probs.set_data_type(TensorPB::FP32);
    input.propose_probs.set_fp32_data("prob");
    input.propose_hidden.set_data_type(TensorPB::BF16);
    input.propose_hidden.set_bf16_data("hidden");
    store_.notify("request", deadline(), input);
    P2PConnectorStartLoadResponsePB response;
    store_.waitAndFill("request", deadline(), response);
    ASSERT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR);
    const auto& payload = response.payload();
    EXPECT_EQ(payload.first_generate_token_id(), 42);
    EXPECT_EQ(payload.total_reuse_len(), 10);
    EXPECT_EQ(payload.local_reuse_len(), 1);
    EXPECT_EQ(payload.remote_reuse_len(), 2);
    EXPECT_EQ(payload.memory_reuse_len(), 3);
    EXPECT_EQ(payload.disk_reuse_len(), 4);
    EXPECT_EQ(payload.tensors().at("propose_tokens").tensor().int32_data().size(), 2 * sizeof(int32_t));
    EXPECT_EQ(payload.tensors().at("position_ids").tensor().int32_data().size(), 2 * sizeof(int32_t));
    EXPECT_EQ(payload.tensors().at("propose_probs").tensor().SerializeAsString(),
              input.propose_probs.SerializeAsString());
    EXPECT_EQ(payload.tensors().at("propose_hidden").tensor().SerializeAsString(),
              input.propose_hidden.SerializeAsString());
}

}  // namespace rtp_llm
