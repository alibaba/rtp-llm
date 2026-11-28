// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/cache/TransferBlockLease.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlReadWorker.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <future>

namespace rtp_llm {
namespace {

class LeaseTestPool: public IBlockPool {
public:
    LeaseTestPool(): IBlockPool(makeConfig()) {
        markInitialized();
    }
    size_t blockSizeBytes() const override { return 16; }
    BlockIndicesType allocate(size_t count) {
        auto blocks = IBlockPool::malloc(count);
        if (!blocks) {
            return {};
        }
        incTreeRef(*blocks, BlockTreeRefType::LOAD);
        return *blocks;
    }
    void requestFree(const BlockIndicesType& blocks) {
        decTreeRef(blocks, BlockTreeRefType::LOAD);
    }
    size_t connectorRefBlocksNum() const {
        return referencedBlocksNum(BlockTreeRefType::STORE);
    }

private:
    static std::shared_ptr<const BlockPoolConfigBase> makeConfig() {
        auto config = std::make_shared<BlockPoolConfigBase>();
        config->physical_block_count = 8;
        return config;
    }
};

class TransferBlockLeaseTest: public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_shared<LeaseTestPool>();
    }
    std::shared_ptr<LeaseTestPool> pool;
};

TEST_F(TransferBlockLeaseTest, RequestCancellationDoesNotRecycleInFlightPages) {
    auto ids   = pool->allocate(2);
    auto lease = std::make_shared<TransferBlockLease>(pool, ids);
    pool->requestFree(ids);
    EXPECT_EQ(pool->freeBlocksNum(), 5);
    EXPECT_EQ(pool->connectorRefBlocksNum(), 2);
    EXPECT_EQ(pool->allocate(5).size(), 5);
    EXPECT_TRUE(pool->allocate(1).empty());
    lease.reset();
    EXPECT_EQ(pool->allocate(2), ids);
}

TEST_F(TransferBlockLeaseTest, PublishedAndCopiedDescriptorsRetainLease) {
    auto ids    = pool->allocate(1);
    auto lease  = std::make_shared<TransferBlockLease>(pool, ids);
    auto buffer = std::make_shared<RequestBlockBuffer>("request");
    pool->requestFree(ids);
    int calls = 0;
    ASSERT_TRUE(buffer->setWatchFunc([&](bool ok, const auto& blocks) {
        if (!blocks.empty()) {
            EXPECT_TRUE(ok);
            EXPECT_EQ(blocks.size(), 1);
            EXPECT_EQ(blocks.front()->addr.get(), pool.get());
            EXPECT_EQ(pool->freeBlocksNum(), 6);
            ++calls;
        }
    }));
    buffer->addBlock("kv", std::shared_ptr<void>(lease, pool.get()), 16, true, true);
    auto copy = std::make_shared<BlockBuffer>(*buffer->getBlock("kv"));
    lease.reset();
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(pool->connectorRefBlocksNum(), 1);
    buffer.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 6);
    copy.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 7);
}

TEST_F(TransferBlockLeaseTest, IndependentReadersReleaseIndependently) {
    auto ids    = pool->allocate(1);
    auto first  = std::make_shared<TransferBlockLease>(pool, ids);
    auto second = std::make_shared<TransferBlockLease>(pool, ids);
    pool->requestFree(ids);
    first.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 6);
    second.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 7);
}

TEST_F(TransferBlockLeaseTest, RepeatedLayerIdsDoNotLeakReferences) {
    auto ids   = pool->allocate(1);
    auto lease = std::make_shared<TransferBlockLease>(pool, BlockIndicesType{ids[0], ids[0], ids[0]});
    pool->requestFree(ids);
    lease.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 7);
    EXPECT_EQ(pool->connectorRefBlocksNum(), 0);
}

TEST_F(TransferBlockLeaseTest, ReaderGrantOutlivesPublicationAndRequest) {
    auto ids   = pool->allocate(1);
    auto block = std::make_shared<BlockBuffer>(
        "kv", std::shared_ptr<void>(std::make_shared<TransferBlockLease>(pool, ids), pool.get()), 16, true, true);
    HixlSourceGrants grants;
    ASSERT_TRUE(grants.reserve("reader", "engine", 1));
    ASSERT_TRUE(grants.retire("reader", "engine", 1));
    EXPECT_FALSE(grants.acquireReserved("reader", 1, {block}));
    EXPECT_FALSE(grants.reserve("reader", "engine", 1));
    ASSERT_TRUE(grants.reserve("reader", "engine", 2));
    ASSERT_TRUE(grants.acquireReserved("reader", 2, {block}));
    pool->requestFree(ids);
    block.reset();
    EXPECT_TRUE(grants.retire("other-reader", "other-engine", 2));
    EXPECT_FALSE(grants.retire("reader", "wrong-engine", 2));
    EXPECT_TRUE(grants.retire("reader", "engine", 1));
    EXPECT_EQ(pool->freeBlocksNum(), 6);
    EXPECT_TRUE(grants.retire("reader", "engine", 2));
    EXPECT_TRUE(grants.retire("reader", "engine", 2));
    EXPECT_EQ(pool->freeBlocksNum(), 7);
}

TEST_F(TransferBlockLeaseTest, SparseBlockRowsKeepOnlyPhysicalPages) {
    auto ids = pool->allocate(1);
    auto lease = std::make_shared<TransferBlockLease>(
        pool, BlockIndicesType{0, NULL_BLOCK_IDX, ids[0], 0, ids[0], NULL_BLOCK_IDX});
    pool->requestFree(ids);
    EXPECT_EQ(pool->connectorRefBlocksNum(), 1);
    lease.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 7);
}

TEST_F(TransferBlockLeaseTest, PaddingOnlyBlockRowsRetainNoPages) {
    auto lease = std::make_shared<TransferBlockLease>(pool, BlockIndicesType{0, NULL_BLOCK_IDX, 0});
    EXPECT_EQ(pool->connectorRefBlocksNum(), 0);
    EXPECT_EQ(pool->freeBlocksNum(), 7);
    lease.reset();
    EXPECT_EQ(pool->freeBlocksNum(), 7);
}

class LeaseMemoryUtil: public NoRdmaMemoryUtilImpl {
public:
    bool requiresBlockLease() const override { return true; }
};

TEST_F(TransferBlockLeaseTest, EarlyCancellationOnlyBlocksDirectPublication) {
    RequestBlockBufferStore tcp(std::make_shared<NoRdmaMemoryUtilImpl>());
    RequestBlockBufferStore direct(std::make_shared<LeaseMemoryUtil>());
    auto request = std::make_shared<RequestBlockBuffer>("request");
    tcp.delRequestBlockBuffer("request");
    direct.delRequestBlockBuffer("request");
    EXPECT_TRUE(tcp.setRequestBlockBuffer(request));
    EXPECT_FALSE(direct.setRequestBlockBuffer(request));
}

class MockReadDriver: public HixlReadDriver {
public:
    MOCK_METHOD(bool, activate, (), (override));
    MOCK_METHOD(bool, connect, (const std::string&), (override));
    MOCK_METHOD(HixlLinkState, connection, (const std::string&), (override));
    MOCK_METHOD(bool, read, (const std::string&, const std::vector<HixlTransferSlice>&, void*&), (override));
    MOCK_METHOD(HixlReadState, poll, (void*), (override));
    MOCK_METHOD(bool, disconnect, (const std::string&), (override));
};

TEST_F(TransferBlockLeaseTest, TimedOutReadDrainsBeforeReconnecting) {
    using namespace testing;
    using namespace std::chrono_literals;
    std::promise<void>           waiting, drained;
    auto                         at_drain = waiting.get_future(), allow_drain = drained.get_future();
    std::promise<HixlReadResult> done;
    auto                         result = done.get_future();
    auto                         driver = std::make_unique<StrictMock<MockReadDriver>>();
    EXPECT_CALL(*driver, activate()).WillOnce(Return(true));
    void* recovered_handle = driver.get();
    EXPECT_CALL(*driver, connect("P")).Times(2).WillRepeatedly(Return(true));
    EXPECT_CALL(*driver, read("P", _, _))
        .WillOnce(DoAll(SetArgReferee<2>(pool.get()), Return(true)))
        .WillOnce(DoAll(SetArgReferee<2>(recovered_handle), Return(true)));
    EXPECT_CALL(*driver, poll(pool.get())).WillRepeatedly(Return(HixlReadState::WAITING));
    EXPECT_CALL(*driver, poll(recovered_handle)).WillOnce(Return(HixlReadState::COMPLETE));
    EXPECT_CALL(*driver, disconnect("P")).Times(2).WillRepeatedly(Return(true));
    EXPECT_CALL(*driver, connection("P"))
        .WillOnce(Return(HixlLinkState::READY))
        .WillOnce(Return(HixlLinkState::CONNECTING))
        .WillOnce([&](const auto&) {
            waiting.set_value();
            allow_drain.wait();  // Keep the READ physically in flight until the assertions below.
            return HixlLinkState::CLOSED;
        })
        .WillOnce(Return(HixlLinkState::READY))
        .WillOnce(Return(HixlLinkState::CLOSED));
    HixlReadProgress worker(std::move(driver));
    ASSERT_TRUE(worker.start());
    auto ids = pool->allocate(1);
    ASSERT_TRUE(worker.submit({"P",
                               {{1, 2, 1}},
                               std::chrono::steady_clock::now() + 1s,
                               std::make_shared<TransferBlockLease>(pool, ids),
                               [&](auto status) {
                                   EXPECT_EQ(pool->freeBlocksNum(), 6);
                                   done.set_value(status);
                               }}));
    pool->requestFree(ids);
    EXPECT_EQ(at_drain.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(pool->freeBlocksNum(), 6);
    EXPECT_EQ(result.wait_for(0s), std::future_status::timeout);
    drained.set_value();
    ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(result.get(), HixlReadResult::TIMEOUT);
    std::promise<HixlReadResult> recovered;
    auto                         recovered_result = recovered.get_future();
    ASSERT_TRUE(worker.submit({"P",
                               {{1, 2, 1}},
                               std::chrono::steady_clock::now() + 3s,
                               std::make_shared<int>(0),
                               [&](auto status) { recovered.set_value(status); }}));
    ASSERT_EQ(recovered_result.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(recovered_result.get(), HixlReadResult::OK);
    worker.stop();
    EXPECT_EQ(pool->freeBlocksNum(), 7);
}

}  // namespace
}  // namespace rtp_llm
