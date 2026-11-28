// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlReadWorker.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <future>

namespace rtp_llm {
namespace {
using namespace testing;
using namespace std::chrono_literals;

class MockReadDriver: public HixlReadDriver {
public:
    MOCK_METHOD(bool, activate, (), (override));
    MOCK_METHOD(bool, connect, (const std::string&), (override));
    MOCK_METHOD(HixlLinkState, connection, (const std::string&), (override));
    MOCK_METHOD(bool, read, (const std::string&, const std::vector<HixlTransferSlice>&, void*&), (override));
    MOCK_METHOD(HixlReadState, poll, (void*), (override));
    MOCK_METHOD(bool, disconnect, (const std::string&), (override));
};

HixlReadJob load(std::string instance, std::promise<HixlReadResult>& done, std::shared_ptr<void> lease = {}) {
    HixlReadJob job{"P", {{1, 2, 1}}, std::chrono::steady_clock::now() + 5s, std::move(lease), [&done](auto result) {
                        done.set_value(result);
                    }};
    job.source_instance = std::move(instance);
    return job;
}

TEST(HixlReadProgressTest, SourceRestartDrainsBeforeReconnecting) {
    auto               driver = std::make_unique<StrictMock<MockReadDriver>>();
    void*              handle = driver.get();
    std::promise<void> draining, allow_drain;
    auto               draining_result = draining.get_future(), can_drain = allow_drain.get_future();
    {
        InSequence order;
        EXPECT_CALL(*driver, activate()).WillOnce(Return(true));
        EXPECT_CALL(*driver, connect("P")).WillOnce(Return(true));
        EXPECT_CALL(*driver, connection("P")).WillOnce(Return(HixlLinkState::READY));
        EXPECT_CALL(*driver, read("P", _, _)).WillOnce(DoAll(SetArgReferee<2>(handle), Return(true)));
        EXPECT_CALL(*driver, poll(handle)).WillOnce(Return(HixlReadState::COMPLETE));
        EXPECT_CALL(*driver, disconnect("P")).WillOnce(Return(true));
        EXPECT_CALL(*driver, connection("P")).WillOnce([&](const auto&) {
            draining.set_value();
            can_drain.wait();
            return HixlLinkState::CLOSED;
        });
        EXPECT_CALL(*driver, connect("P")).WillOnce(Return(true));
        EXPECT_CALL(*driver, connection("P")).WillOnce(Return(HixlLinkState::READY));
        EXPECT_CALL(*driver, read("P", _, _)).WillOnce(DoAll(SetArgReferee<2>(handle), Return(true)));
        EXPECT_CALL(*driver, poll(handle)).WillOnce(Return(HixlReadState::COMPLETE));
        EXPECT_CALL(*driver, disconnect("P")).WillOnce(Return(true));
        EXPECT_CALL(*driver, connection("P")).WillOnce(Return(HixlLinkState::CLOSED));
    }
    HixlReadProgress worker(std::move(driver));
    ASSERT_TRUE(worker.start());
    std::promise<HixlReadResult> first, restarted, recovered;
    auto                         first_result = first.get_future(), restarted_result = restarted.get_future(),
         recovered_result = recovered.get_future();
    ASSERT_TRUE(worker.submit(load("boot-A", first)));
    ASSERT_EQ(first_result.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(first_result.get(), HixlReadResult::OK);
    auto               owner = std::make_shared<int>(1);
    std::weak_ptr<int> alive = owner;
    ASSERT_TRUE(worker.submit(load("boot-B", restarted, std::move(owner))));
    EXPECT_EQ(draining_result.wait_for(3s), std::future_status::ready);
    EXPECT_FALSE(alive.expired());
    EXPECT_EQ(restarted_result.wait_for(0s), std::future_status::timeout);
    allow_drain.set_value();
    ASSERT_EQ(restarted_result.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(restarted_result.get(), HixlReadResult::CONNECT_ERROR);
    ASSERT_TRUE(worker.submit(load("boot-B", recovered)));
    ASSERT_EQ(recovered_result.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(recovered_result.get(), HixlReadResult::OK);
    worker.stop();
    EXPECT_TRUE(alive.expired());
}

TEST(HixlReadProgressTest, SourceRestartRetainsInFlightLeasesUntilDrain) {
    auto               driver = std::make_unique<StrictMock<MockReadDriver>>();
    void*              handle = driver.get();
    std::promise<void> reading, draining, allow_drain;
    auto               reading_result = reading.get_future(), draining_result = draining.get_future(),
         can_drain = allow_drain.get_future();
    EXPECT_CALL(*driver, activate()).WillOnce(Return(true));
    EXPECT_CALL(*driver, connect("P")).WillOnce(Return(true));
    EXPECT_CALL(*driver, read("P", _, _)).WillOnce(DoAll(SetArgReferee<2>(handle), Return(true)));
    EXPECT_CALL(*driver, poll(handle))
        .WillOnce([&](void*) {
            reading.set_value();
            return HixlReadState::WAITING;
        })
        .WillRepeatedly(Return(HixlReadState::WAITING));
    EXPECT_CALL(*driver, disconnect("P")).WillOnce(Return(true));
    EXPECT_CALL(*driver, connection("P")).WillOnce(Return(HixlLinkState::READY)).WillOnce([&](const auto&) {
        draining.set_value();
        can_drain.wait();
        return HixlLinkState::CLOSED;
    });
    HixlReadProgress worker(std::move(driver));
    ASSERT_TRUE(worker.start());
    std::promise<HixlReadResult> old_source, new_source;
    auto                         old_result = old_source.get_future(), new_result = new_source.get_future();
    auto                         owner = std::make_shared<int>(1);
    std::weak_ptr<int>           alive = owner;
    ASSERT_TRUE(worker.submit(load("boot-A", old_source, std::move(owner))));
    ASSERT_EQ(reading_result.wait_for(3s), std::future_status::ready);
    ASSERT_TRUE(worker.submit(load("boot-B", new_source)));
    EXPECT_EQ(draining_result.wait_for(3s), std::future_status::ready);
    EXPECT_FALSE(alive.expired());
    EXPECT_EQ(old_result.wait_for(0s), std::future_status::timeout);
    EXPECT_EQ(new_result.wait_for(0s), std::future_status::timeout);
    allow_drain.set_value();
    ASSERT_EQ(old_result.wait_for(3s), std::future_status::ready);
    ASSERT_EQ(new_result.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(old_result.get(), HixlReadResult::CONNECT_ERROR);
    EXPECT_EQ(new_result.get(), HixlReadResult::CONNECT_ERROR);
    worker.stop();
    EXPECT_TRUE(alive.expired());
}

TEST(HixlReadProgressTest, SameSourceReusesTheConnection) {
    auto  driver = std::make_unique<StrictMock<MockReadDriver>>();
    void* handle = driver.get();
    EXPECT_CALL(*driver, activate()).WillOnce(Return(true));
    EXPECT_CALL(*driver, connect("P")).WillOnce(Return(true));
    EXPECT_CALL(*driver, connection("P"))
        .WillOnce(Return(HixlLinkState::READY))
        .WillOnce(Return(HixlLinkState::CLOSED));
    EXPECT_CALL(*driver, read("P", _, _)).Times(2).WillRepeatedly(DoAll(SetArgReferee<2>(handle), Return(true)));
    EXPECT_CALL(*driver, poll(handle)).Times(2).WillRepeatedly(Return(HixlReadState::COMPLETE));
    EXPECT_CALL(*driver, disconnect("P")).WillOnce(Return(true));
    HixlReadProgress worker(std::move(driver));
    ASSERT_TRUE(worker.start());
    for (int i = 0; i < 2; ++i) {
        std::promise<HixlReadResult> done;
        auto                         result = done.get_future();
        ASSERT_TRUE(worker.submit(load("boot-A", done)));
        ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
        EXPECT_EQ(result.get(), HixlReadResult::OK);
    }
    worker.stop();
}

}  // namespace
}  // namespace rtp_llm
