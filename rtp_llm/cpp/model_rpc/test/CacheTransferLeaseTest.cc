#include "rtp_llm/cpp/model_rpc/CacheTransferLease.h"

#include <chrono>
#include <future>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace test {

TEST(CacheTransferLeaseTest, BuildsResourceForEveryConfiguredGroup) {
    auto full_blocks = std::make_shared<BlockIds>(2);
    full_blocks->assign({11, 12});
    auto linear_blocks = std::make_shared<BlockIds>();
    linear_blocks->assign({21, 22, 23});

    auto result = makeCacheTransferLeaseResource(
        /*configured_group_count=*/2,
        /*block_ids_by_group=*/{full_blocks, linear_blocks},
        /*cache_key_count=*/2,
        /*max_block_id=*/32);

    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->cacheKeys(), (CacheKeysType{0, 1}));
    EXPECT_EQ(result->blocks(0), (BlockIndicesType{11, 12}));
    EXPECT_EQ(result->blocks(1), (BlockIndicesType{21, 22, 23}));
}

TEST(CacheTransferLeaseTest, RejectsIncompleteGroupLayout) {
    auto blocks = std::make_shared<BlockIds>();
    blocks->assign({11});

    const auto make_resource = [](const GroupBlockIds& groups, size_t key_count) {
        return makeCacheTransferLeaseResource(/*configured_group_count=*/2, groups, key_count, /*max_block_id=*/32);
    };

    EXPECT_FALSE(make_resource({blocks}, 1).ok());
    EXPECT_FALSE(make_resource({blocks, nullptr}, 1).ok());
    EXPECT_FALSE(make_resource({blocks, blocks}, 2).ok());
}

TEST(CacheTransferLeaseTest, ValidatesPhysicalBlockRangeBeforeReferenceAcquisition) {
    auto valid_blocks = std::make_shared<BlockIds>();
    valid_blocks->assign({NULL_BLOCK_IDX, 4});
    EXPECT_TRUE(makeCacheTransferLeaseResource(
                    /*configured_group_count=*/1,
                    /*block_ids_by_group=*/{valid_blocks},
                    /*cache_key_count=*/2,
                    /*max_block_id=*/4)
                    .ok());

    auto reserved_block = std::make_shared<BlockIds>();
    reserved_block->assign({0});
    EXPECT_FALSE(makeCacheTransferLeaseResource(
                     /*configured_group_count=*/1,
                     /*block_ids_by_group=*/{reserved_block},
                     /*cache_key_count=*/1,
                     /*max_block_id=*/4)
                     .ok());

    auto out_of_range_block = std::make_shared<BlockIds>();
    out_of_range_block->assign({5});
    EXPECT_FALSE(makeCacheTransferLeaseResource(
                     /*configured_group_count=*/1,
                     /*block_ids_by_group=*/{out_of_range_block},
                     /*cache_key_count=*/1,
                     /*max_block_id=*/4)
                     .ok());
}

TEST(CacheTransferLeaseTest, AddressKeepsLeaseAliveUntilTransferCompletion) {
    auto                    lease = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> weak_lease = lease;
    int                     target = 0;

    auto address = makeCacheTransferAddress(lease, &target);
    lease.reset();

    EXPECT_EQ(address.get(), &target);
    EXPECT_FALSE(weak_lease.expired());

    address.reset();
    EXPECT_TRUE(weak_lease.expired());
}

TEST(CacheTransferLeaseTest, LastAddressReleaseCompletesOperationFence) {
    using namespace std::chrono_literals;

    RemoteLoadFenceRegistry registry;
    const auto request_deadline_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch() + 5s)
            .count();
    auto token = makeRemoteLoadAllocationToken("test-owner", "allocation-a", request_deadline_unix_ms);
    ASSERT_TRUE(token.ok()) << token.status();
    auto operation_result = registry.begin(*token, request_deadline_unix_ms);
    ASSERT_TRUE(operation_result.ok()) << operation_result.status();
    auto operation = *operation_result;

    auto                           block_lease = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> weak_block_lease = block_lease;
    auto lifetime = makeCacheTransferLifetime(block_lease, operation);
    ASSERT_NE(lifetime, nullptr);

    int  first_target   = 0;
    int  second_target  = 0;
    auto first_address  = makeCacheTransferAddress(lifetime, &first_target);
    auto second_address = makeCacheTransferAddress(lifetime, &second_target);
    ASSERT_NE(first_address, nullptr);
    ASSERT_NE(second_address, nullptr);

    operation_result = absl::UnknownError("released operation result");
    operation.reset();
    block_lease.reset();
    lifetime.reset();

    auto waiter = std::async(std::launch::async, [&registry, token = *token, request_deadline_unix_ms]() {
        return registry.sealAndWait(
            token, request_deadline_unix_ms, std::chrono::steady_clock::now() + 2s);
    });

    EXPECT_EQ(waiter.wait_for(30ms), std::future_status::timeout);
    first_address.reset();
    EXPECT_EQ(waiter.wait_for(30ms), std::future_status::timeout);
    EXPECT_FALSE(weak_block_lease.expired());

    second_address.reset();
    ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(waiter.get().ok());
    EXPECT_TRUE(weak_block_lease.expired());
}

}  // namespace test
}  // namespace rtp_llm
