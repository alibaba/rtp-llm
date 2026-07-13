#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CopyEngineTestUtils.h"

namespace rtp_llm {
namespace {

using copy_engine_test::TempDirGuard;
using copy_engine_test::expectStatus;
using copy_engine_test::makeDescriptor;
using copy_engine_test::makeDiskPool;
using copy_engine_test::makeHostPool;
using copy_engine_test::poolMalloc;

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

ComponentGroupPtr
makeHostDiskGroup(int group_id, std::shared_ptr<HostBlockPool> host_pool, std::shared_ptr<DiskBlockPool> disk_pool) {
    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = group_id;
    group->group_type         = CacheGroupType::FULL;
    group->setHostPool(std::move(host_pool));
    group->setDiskPool(std::move(disk_pool));
    return group;
}

class CopyEngineHostDiskTest: public ::testing::Test {
protected:
    void SetUp() override {
        slots_ = {
            {0, "layer_0", 128},
            {1, "layer_1", 256},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);

        host_pool_ = makeHostPool(host_block_size_, 4, false);
        disk_pool_ = makeDiskPool(host_block_size_, 7, temp_dir_.path);

        component_group_ = makeHostDiskGroup(0, host_pool_, disk_pool_);
        copy_engine_ =
            std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{component_group_}, std::vector<Component>{});
    }

    TempDirGuard                         temp_dir_{"copy_engine_test"};
    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>       host_pool_;
    std::shared_ptr<DiskBlockPool>       disk_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    ComponentGroupPtr                    component_group_;
};

TEST_F(CopyEngineHostDiskTest, SubmitHostToDiskRoundTrip) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < host_block_size_; ++i)
        host_data[i] = static_cast<uint8_t>(i & 0xFF);

    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();

    auto host_to_disk = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_slot);
    ASSERT_TRUE(copy_engine_->submit(host_to_disk).ok());

    std::memset(host_data, 0, host_block_size_);

    auto disk_to_host = makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_slot);
    ASSERT_TRUE(copy_engine_->submit(disk_to_host).ok());

    for (size_t i = 0; i < host_block_size_; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF)) << "byte " << i;

    host_pool_->free(host_block);
    disk_pool_->free(disk_slot);
}

TEST_F(CopyEngineHostDiskTest, SubmitRejectsMissingRequiredBlocks) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const auto disk_block = poolMalloc(*disk_pool_);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DISK, {}, NULL_BLOCK_IDX, disk_block), CopyStatus::INVALID_ARGS);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::DISK, Tier::HOST, {}, NULL_BLOCK_IDX, disk_block), CopyStatus::INVALID_ARGS);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, NULL_BLOCK_IDX), CopyStatus::INVALID_ARGS);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, NULL_BLOCK_IDX), CopyStatus::INVALID_ARGS);

    host_pool_->free(host_block);
    disk_pool_->free(disk_block);
}

TEST_F(CopyEngineHostDiskTest, SubmitHostToDiskAcceptsValidUnallocatedDiskBlock) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    constexpr BlockIdxType unallocated_disk_block = 1;
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, unallocated_disk_block), CopyStatus::OK);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineHostDiskTest, SubmitHostToDiskRejectsOutOfRangeDiskBlock) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    const BlockIdxType out_of_range = static_cast<BlockIdxType>(disk_pool_->totalBlocksNum() + 1);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, out_of_range), CopyStatus::INVALID_ARGS);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineHostDiskTest, SubmitRejectsInvalidHostDiskLayout) {
    auto host_block = poolMalloc(*host_pool_);
    auto disk_block = poolMalloc(*disk_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto missing_host_group = makeHostDiskGroup(0, nullptr, disk_pool_);
    auto missing_host_engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{missing_host_group}, std::vector<Component>{});
    expectStatus(missing_host_engine,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block),
                 CopyStatus::INVALID_ARGS);
    expectStatus(missing_host_engine,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block),
                 CopyStatus::INVALID_ARGS);

    auto missing_disk_group = makeHostDiskGroup(0, host_pool_, nullptr);
    auto missing_disk_engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{missing_disk_group}, std::vector<Component>{});
    expectStatus(missing_disk_engine,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block),
                 CopyStatus::INVALID_ARGS);
    expectStatus(missing_disk_engine,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block),
                 CopyStatus::INVALID_ARGS);

    auto mismatched_host_pool  = makeHostPool(host_block_size_ + 1, 2, false);
    auto mismatched_host_block = poolMalloc(*mismatched_host_pool);
    auto mismatch_group        = makeHostDiskGroup(0, mismatched_host_pool, disk_pool_);
    auto mismatch_engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{mismatch_group}, std::vector<Component>{});
    expectStatus(mismatch_engine,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, mismatched_host_block, disk_block),
                 CopyStatus::INVALID_ARGS);
    expectStatus(mismatch_engine,
                 makeDescriptor(Tier::DISK, Tier::HOST, {}, mismatched_host_block, disk_block),
                 CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineHostDiskTest, HostDiskStatusMapping) {
    const std::array<std::pair<BlockIOStatus, CopyStatus>, 6> mappings = {
        std::pair{BlockIOStatus::OK, CopyStatus::OK},
        std::pair{BlockIOStatus::INVALID_BLOCK, CopyStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::INVALID_SIZE, CopyStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::ALIGNMENT_ERROR, CopyStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::IO_ERROR, CopyStatus::DISK_IO_ERROR},
        std::pair{BlockIOStatus::PARTIAL_FAILURE, CopyStatus::DISK_IO_ERROR},
    };
    for (const auto& [input, expected] : mappings) {
        EXPECT_EQ(HostDiskTransferExecutor::blockIOStatusToCopyStatus(input), expected);
    }

    const std::array<std::pair<DiskBlockIOStatus, CopyStatus>, 5> io_mappings = {
        std::pair{DiskBlockIOStatus::OK, CopyStatus::OK},
        std::pair{DiskBlockIOStatus::INVALID_SIZE, CopyStatus::INVALID_ARGS},
        std::pair{DiskBlockIOStatus::ALIGNMENT_ERROR, CopyStatus::INVALID_ARGS},
        std::pair{DiskBlockIOStatus::IO_ERROR, CopyStatus::DISK_IO_ERROR},
        std::pair{DiskBlockIOStatus::PARTIAL_FAILURE, CopyStatus::DISK_IO_ERROR},
    };
    int pool_suffix = 0;
    for (const auto& [io_status, expected] : io_mappings) {
        SCOPED_TRACE(::testing::Message() << "io_mapping=" << pool_suffix);
        auto disk_pool  = makeDiskPool(host_block_size_,
                                      2,
                                      temp_dir_.path,
                                      std::make_unique<StatusDiskBlockIO>(io_status),
                                      "copy_engine_status_" + std::to_string(pool_suffix++));
        auto host_block = poolMalloc(*host_pool_);
        auto disk_block = poolMalloc(*disk_pool);
        auto group      = makeHostDiskGroup(0, host_pool_, disk_pool);
        auto engine     = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{});
        expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block), expected);
        expectStatus(engine, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block), expected);
        host_pool_->free(host_block);
    }
}

TEST(CopyEngineTest, ComputeHostBlockSize) {
    std::vector<MemoryBlockLayerTagSlot> test_slots = {
        {0, "a", 100},
        {1, "b", 200},
        {2, "c", 300},
    };
    EXPECT_EQ(CopyEngine::computeHostBlockSize(test_slots), 600u);
}

}  // namespace
}  // namespace rtp_llm
