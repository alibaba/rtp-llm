#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::expectStatus;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::poolMalloc;
using block_transfer_engine_test::submitSucceeded;

class StatusDiskBlockIO: public DiskBlockIO {
public:
    explicit StatusDiskBlockIO(DiskBlockIOStatus status): status_(status) {}

    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override {
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus read(uint64_t, void*, size_t) override {
        return status_;
    }
    DiskBlockIOStatus write(uint64_t, const void*, size_t) override {
        return status_;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>&) override {
        return status_;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>&) override {
        return status_;
    }
    void        close() override {}
    std::string debugString() const override {
        return "StatusDiskBlockIO";
    }

private:
    DiskBlockIOStatus status_;
};

GroupSetPtr makeHostDiskGroup(size_t                                  group_set_id,
                              std::shared_ptr<HostBlockPool>          host_pool,
                              std::shared_ptr<BlockTreeDiskBlockPool> disk_pool,
                              size_t                                  payload_bytes) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology              = makeTestTopology({{"host_disk", policy, {0}, payload_bytes, 0}});
    auto device_pool = makeTestDevicePool({{payload_bytes, 0}}, 2, "host_disk_group_" + std::to_string(group_set_id));
    auto group       = makeTestGroupSet(group_set_id, std::move(topology), {0}, {std::move(device_pool)});
    group->setHostPool(std::move(host_pool));
    group->setDiskPool(std::move(disk_pool));
    return group;
}

std::shared_ptr<PerRankBlockTransferEngine> makeEngine(std::vector<GroupSetPtr> groups) {
    return std::make_shared<PerRankBlockTransferEngine>(std::move(groups));
}

class PerRankBlockTransferEngineHostDiskTest: public ::testing::Test {
protected:
    void SetUp() override {
        host_block_size_ = 384;

        host_pool_ = makeHostPool(host_block_size_, 4, false);
        disk_pool_ = makeDiskPool(host_block_size_, 7, temp_dir_.path);

        group_set_ = makeHostDiskGroup(0, host_pool_, disk_pool_, host_block_size_);
        ASSERT_EQ(group_set_->payloadBytes(), host_block_size_);
        per_rank_transfer_engine_ = makeEngine({group_set_});
    }

    TempDirGuard                                temp_dir_{"block_transfer_engine_test"};
    size_t                                      host_block_size_;
    std::shared_ptr<HostBlockPool>              host_pool_;
    std::shared_ptr<BlockTreeDiskBlockPool>     disk_pool_;
    std::shared_ptr<PerRankBlockTransferEngine> per_rank_transfer_engine_;
    GroupSetPtr                                 group_set_;
};

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitHostToDiskRoundTrip) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < host_block_size_; ++i)
        host_data[i] = static_cast<uint8_t>(i & 0xFF);

    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();

    auto host_to_disk = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_slot);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine_, host_to_disk));

    std::memset(host_data, 0, host_block_size_);

    auto disk_to_host = makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_slot);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine_, disk_to_host));

    for (size_t i = 0; i < host_block_size_; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF)) << "byte " << i;

    host_pool_->free(host_block);
    disk_pool_->free(disk_slot);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitRejectsMissingRequiredBlocks) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const auto disk_block = poolMalloc(*disk_pool_);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, NULL_BLOCK_IDX, disk_block),
                 TransferStatus::INVALID_ARGS);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, NULL_BLOCK_IDX, disk_block),
                 TransferStatus::INVALID_ARGS);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, NULL_BLOCK_IDX),
                 TransferStatus::INVALID_ARGS);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, NULL_BLOCK_IDX),
                 TransferStatus::INVALID_ARGS);

    host_pool_->free(host_block);
    disk_pool_->free(disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitHostToDiskAcceptsValidUnallocatedDiskBlock) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    constexpr BlockIdxType unallocated_disk_block = 1;
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, unallocated_disk_block),
                 TransferStatus::OK);

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitHostToDiskRejectsOutOfRangeDiskBlock) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    const BlockIdxType out_of_range = static_cast<BlockIdxType>(disk_pool_->totalBlocksNum() + 1);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, out_of_range),
                 TransferStatus::INVALID_ARGS);

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitRejectsInvalidHostDiskLayout) {
    auto host_block = poolMalloc(*host_pool_);
    auto disk_block = poolMalloc(*disk_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto missing_host_group  = makeHostDiskGroup(0, nullptr, disk_pool_, host_block_size_);
    auto missing_host_engine = makeEngine({missing_host_group});
    expectStatus(missing_host_engine,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block),
                 TransferStatus::INVALID_ARGS);
    expectStatus(missing_host_engine,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block),
                 TransferStatus::INVALID_ARGS);

    auto missing_disk_group  = makeHostDiskGroup(0, host_pool_, nullptr, host_block_size_);
    auto missing_disk_engine = makeEngine({missing_disk_group});
    expectStatus(missing_disk_engine,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block),
                 TransferStatus::INVALID_ARGS);
    expectStatus(missing_disk_engine,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block),
                 TransferStatus::INVALID_ARGS);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, HostDiskStatusMapping) {
    const std::array<std::pair<BlockIOStatus, TransferStatus>, 6> mappings = {
        std::pair{BlockIOStatus::OK, TransferStatus::OK},
        std::pair{BlockIOStatus::INVALID_BLOCK, TransferStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::INVALID_SIZE, TransferStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::ALIGNMENT_ERROR, TransferStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::IO_ERROR, TransferStatus::DISK_IO_ERROR},
        std::pair{BlockIOStatus::PARTIAL_FAILURE, TransferStatus::DISK_IO_ERROR},
    };
    for (const auto& [input, expected] : mappings) {
        EXPECT_EQ(HostDiskTransferExecutor::blockIOStatusToTransferStatus(input), expected);
    }

    const std::array<std::pair<DiskBlockIOStatus, TransferStatus>, 5> io_mappings = {
        std::pair{DiskBlockIOStatus::OK, TransferStatus::OK},
        std::pair{DiskBlockIOStatus::INVALID_SIZE, TransferStatus::INVALID_ARGS},
        std::pair{DiskBlockIOStatus::ALIGNMENT_ERROR, TransferStatus::INVALID_ARGS},
        std::pair{DiskBlockIOStatus::IO_ERROR, TransferStatus::DISK_IO_ERROR},
        std::pair{DiskBlockIOStatus::PARTIAL_FAILURE, TransferStatus::DISK_IO_ERROR},
    };
    int pool_suffix = 0;
    for (const auto& [io_status, expected] : io_mappings) {
        SCOPED_TRACE(::testing::Message() << "io_mapping=" << pool_suffix);
        auto disk_pool  = makeDiskPool(host_block_size_,
                                      2,
                                      temp_dir_.path,
                                      std::make_unique<StatusDiskBlockIO>(io_status),
                                      "per_rank_transfer_engine_status_" + std::to_string(pool_suffix++));
        auto host_block = poolMalloc(*host_pool_);
        auto disk_block = poolMalloc(*disk_pool);
        auto group      = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
        auto engine     = makeEngine({group});
        expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block), expected);
        expectStatus(engine, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block), expected);
        host_pool_->free(host_block);
    }
}

TEST(GroupSetPayloadTest, PayloadBytesUsesLogicalStridesAcrossLayers) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology              = makeTestTopology({{"abc", policy, {0, 1, 2}, 160, 40}});
    auto pool                  = makeTestDevicePool({{200, 40}, {220, 40}, {240, 40}}, 2, "group_set_payload_test");
    auto group                 = makeTestGroupSet(0, std::move(topology), {0}, {std::move(pool)});
    EXPECT_EQ(group->payloadBytes(), 600u);
}

}  // namespace
}  // namespace rtp_llm
