#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include "rtp_llm/cpp/cache/connector/p2p/PrefillResultStore.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class PrefillResultStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        deadline_ = currentTimeMs() + 5000;
        store_.updateDeadline("request", deadline_);
    }

    PrefillResultStore::SideChannelData data(int64_t token = 42) {
        PrefillResultStore::SideChannelData result;
        result.has_first_token = true;
        result.first_token_id  = token;
        return result;
    }

    PrefillResultStore store_;
    int64_t            deadline_;
};

TEST_F(PrefillResultStoreTest, PublishBeforeWaitAndConsumeOnce) {
    store_.publishPrefillPayload("request", data(0));
    ASSERT_TRUE(store_.waitPrefillPayloadReady("request", deadline_));
    auto result = store_.takePrefillPayload("request");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->has_first_token);
    EXPECT_EQ(result->first_token_id, 0);
    EXPECT_FALSE(store_.takePrefillPayload("request").has_value());
    store_.publishPrefillPayload("request", data(99));
    EXPECT_FALSE(store_.waitPrefillPayloadReady("request", deadline_));
}

TEST_F(PrefillResultStoreTest, WaitBeforePublish) {
    std::promise<void> entered;
    std::atomic<bool>  signalled{false};
    auto               waiter = std::async(std::launch::async, [&] {
        return store_.waitPrefillPayloadReady("request", deadline_, [&] {
            if (!signalled.exchange(true)) {
                entered.set_value();
            }
            return false;
        });
    });
    ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(1)), std::future_status::ready);
    store_.publishPrefillPayload("request", data());
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(waiter.get());
}

TEST_F(PrefillResultStoreTest, TerminalBeforePublishRejectsLateResultAndDeadlineUpdate) {
    store_.markTerminal("request");
    store_.updateDeadline("request", deadline_ + 5000);
    store_.publishPrefillPayload("request", data());
    EXPECT_FALSE(store_.waitPrefillPayloadReady("request", deadline_));
    EXPECT_FALSE(store_.takePrefillPayload("request").has_value());
}

TEST_F(PrefillResultStoreTest, TerminalDropsPayloadOutsideStoreLock) {
    auto result          = data();
    bool destroyed       = false;
    result.propose_probs = torch::from_blob(
        new float[1]{1.0f},
        {1},
        [&](void* ptr) {
            // Tensor destruction must not run under the result lock.
            store_.updateDeadline("other", deadline_);
            destroyed = true;
            delete[] static_cast<float*>(ptr);
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    store_.publishPrefillPayload("request", std::move(result));
    auto retired = store_.markTerminal("request");
    EXPECT_FALSE(destroyed);
    retired.reset();
    EXPECT_TRUE(destroyed);
    EXPECT_FALSE(store_.takePrefillPayload("request").has_value());
}

TEST_F(PrefillResultStoreTest, TerminalWakesWaiter) {
    std::promise<void> entered;
    std::atomic<bool>  signalled{false};
    auto               waiter = std::async(std::launch::async, [&] {
        return store_.waitPrefillPayloadReady("request", deadline_, [&] {
            if (!signalled.exchange(true)) {
                entered.set_value();
            }
            return false;
        });
    });
    ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(1)), std::future_status::ready);
    store_.markTerminal("request");
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(waiter.get());
}

TEST_F(PrefillResultStoreTest, CancelledWaitDoesNotConsumePayload) {
    store_.publishPrefillPayload("request", data());
    EXPECT_FALSE(store_.waitPrefillPayloadReady("request", deadline_, [] { return true; }));
    EXPECT_TRUE(store_.takePrefillPayload("request").has_value());
}

TEST_F(PrefillResultStoreTest, WaitUsesEarlierCallerOrRequestDeadline) {
    store_.publishPrefillPayload("request", data());
    EXPECT_FALSE(store_.waitPrefillPayloadReady("request", currentTimeMs() - 1));
    store_.updateDeadline("request", currentTimeMs() - 1);
    EXPECT_FALSE(store_.waitPrefillPayloadReady("request", deadline_));
}

TEST_F(PrefillResultStoreTest, ExpiredResultCannotBePublishedOrConsumedBeforeCleanup) {
    store_.publishPrefillPayload("request", data());
    store_.updateDeadline("request", currentTimeMs() - 1);
    EXPECT_FALSE(store_.takePrefillPayload("request").has_value());
    auto retired = store_.clearPrefillPayload("request");
    ASSERT_TRUE(retired.has_value());
    store_.publishPrefillPayload("request", data(99));
    EXPECT_FALSE(store_.clearPrefillPayload("request").has_value());
}

TEST_F(PrefillResultStoreTest, DeadlineUpdateWakesPendingWaiter) {
    std::promise<void> entered;
    std::atomic<bool>  signalled{false};
    auto               waiter = std::async(std::launch::async, [&] {
        return store_.waitPrefillPayloadReady("request", deadline_, [&] {
            if (!signalled.exchange(true)) {
                entered.set_value();
            }
            return false;
        });
    });
    ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(1)), std::future_status::ready);
    store_.updateDeadline("request", currentTimeMs() - 1);
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(waiter.get());
}

TEST_F(PrefillResultStoreTest, ClearDoesNotEndRequest) {
    store_.publishPrefillPayload("request", data());
    EXPECT_TRUE(store_.clearPrefillPayload("request").has_value());
    EXPECT_FALSE(store_.takePrefillPayload("request").has_value());
    store_.publishPrefillPayload("request", data(99));
    auto result = store_.takePrefillPayload("request");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->first_token_id, 99);
}

TEST_F(PrefillResultStoreTest, StopWakesWaiter) {
    std::promise<void> entered;
    std::atomic<bool>  signalled{false};
    auto               waiter = std::async(std::launch::async, [&] {
        return store_.waitPrefillPayloadReady("request", deadline_, [&] {
            if (!signalled.exchange(true)) {
                entered.set_value();
            }
            return false;
        });
    });
    ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(1)), std::future_status::ready);
    store_.stop();
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(waiter.get());
}

TEST_F(PrefillResultStoreTest, FillsTokenReuseMtpAndFirstTokenTensors) {
    auto result             = data(0);
    result.total_reuse_len  = 12;
    result.local_reuse_len  = 8;
    result.remote_reuse_len = 4;
    result.memory_reuse_len = 3;
    result.disk_reuse_len   = 1;
    result.propose_tokens   = {7, 8};
    result.propose_probs    = torch::ones({2}, torch::kFloat32);
    result.propose_hidden   = torch::ones({2, 4}, torch::kFloat16);
    result.position_ids     = {10, 11};
    result.first_token_tensors.emplace("first_token_logits", torch::ones({1, 4}, torch::kFloat32));
    P2PConnectorStartLoadResponsePB response;
    (*response.mutable_payload()->mutable_tensors())["old"].mutable_tensor();
    ASSERT_TRUE(PrefillResultStore::fillStartLoadResponsePayload(result, response).ok());
    const auto& payload = response.payload();
    EXPECT_TRUE(payload.has_first_generate_token());
    EXPECT_EQ(payload.first_generate_token_id(), 0);
    EXPECT_EQ(payload.total_reuse_len(), 12);
    EXPECT_EQ(payload.local_reuse_len(), 8);
    EXPECT_EQ(payload.remote_reuse_len(), 4);
    EXPECT_EQ(payload.memory_reuse_len(), 3);
    EXPECT_EQ(payload.disk_reuse_len(), 1);
    EXPECT_EQ(payload.tensors().count("old"), 0);
    EXPECT_EQ(payload.tensors().at("propose_tokens").tensor().int32_data().size(), 2 * sizeof(int32_t));
    EXPECT_EQ(payload.tensors().at("propose_probs").tensor().fp32_data().size(), 2 * sizeof(float));
    EXPECT_EQ(payload.tensors().at("propose_hidden").tensor().fp16_data().size(), 16u);
    EXPECT_EQ(payload.tensors().at("position_ids").tensor().int32_data().size(), 2 * sizeof(int32_t));
    EXPECT_EQ(payload.tensors().at("first_token_logits").tensor().shape_size(), 2);
}

TEST_F(PrefillResultStoreTest, SerializationFailureClearsPartialPayload) {
    auto result          = data();
    result.propose_probs = torch::ones({2}, torch::kInt64);
    P2PConnectorStartLoadResponsePB response;
    EXPECT_FALSE(PrefillResultStore::fillStartLoadResponsePayload(result, response).ok());
    EXPECT_FALSE(response.has_payload());
}

}  // namespace rtp_llm
