#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::DirectAlignmentDiskBlockIO;
using block_transfer_engine_test::StatusDiskBlockIO;
using block_transfer_engine_test::expectStatus;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::poolMalloc;
using block_transfer_engine_test::submitSucceeded;

GroupSetPtr makeHostDiskGroup(size_t                                  group_set_id,
                              std::shared_ptr<HostBlockPool>          host_pool,
                              std::shared_ptr<BlockTreeDiskBlockPool> disk_pool,
                              size_t                                  payload_bytes) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology              = makeTestTopology({makeTestGroupBase(policy, {0}, payload_bytes)});
    auto device_pool = makeTestDevicePool({{payload_bytes, 0}}, 2, "host_disk_group_" + std::to_string(group_set_id));
    auto group       = makeTestGroupSet(
        group_set_id, std::move(topology), {0}, {std::move(device_pool)}, std::move(host_pool), std::move(disk_pool));
    return group;
}

std::shared_ptr<PerRankBlockTransferEngine> makeEngine(std::vector<GroupSetPtr> groups) {
    return std::make_shared<PerRankBlockTransferEngine>(std::move(groups));
}

class RecordingBatchDiskBlockIO: public DiskBlockIO {
public:
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override {
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus read(uint64_t, void*, size_t) override {
        std::lock_guard<std::mutex> lock(recording_mutex);
        ++single_read_calls;
        return status;
    }
    DiskBlockIOStatus write(uint64_t, const void*, size_t) override {
        std::lock_guard<std::mutex> lock(recording_mutex);
        ++single_write_calls;
        return status;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override {
        std::lock_guard<std::mutex> lock(recording_mutex);
        ++batch_read_calls;
        last_reads = reads;
        return status;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        std::lock_guard<std::mutex> lock(recording_mutex);
        ++batch_write_calls;
        batch_write_sizes.push_back(writes.size());
        last_writes = writes;
        return status;
    }
    void        close() override {}
    std::string debugString() const override {
        return "RecordingBatchDiskBlockIO";
    }

    DiskBlockIOStatus      status{DiskBlockIOStatus::OK};
    size_t                 single_read_calls{0};
    size_t                 single_write_calls{0};
    size_t                 batch_read_calls{0};
    size_t                 batch_write_calls{0};
    std::vector<size_t>    batch_write_sizes;
    std::vector<DiskRead>  last_reads;
    std::vector<DiskWrite> last_writes;

protected:
    std::mutex recording_mutex;
};

class BlockingBatchDiskBlockIO: public RecordingBatchDiskBlockIO {
public:
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entered_ = true;
        }
        entered_cv_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        release_cv_.wait(lock, [this] { return released_; });
        lock.unlock();
        return RecordingBatchDiskBlockIO::write(writes);
    }

    bool waitUntilEntered(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return entered_cv_.wait_for(lock, timeout, [this] { return entered_; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        release_cv_.notify_all();
    }

private:
    std::mutex              mutex_;
    std::condition_variable entered_cv_;
    std::condition_variable release_cv_;
    bool                    entered_{false};
    bool                    released_{false};
};

class TwoCallBarrierDiskBlockIO: public RecordingBatchDiskBlockIO {
public:
    enum class Direction {
        READ,
        WRITE,
    };

    explicit TwoCallBarrierDiskBlockIO(Direction direction): direction_(direction) {}

    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override {
        if (direction_ == Direction::READ) {
            enterAndWait();
        }
        return RecordingBatchDiskBlockIO::read(reads);
    }

    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        if (direction_ == Direction::WRITE) {
            enterAndWait();
        }
        return RecordingBatchDiskBlockIO::write(writes);
    }

    bool waitForCallCount(size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return entered_cv_.wait_for(lock, timeout, [this, expected] { return entered_count_ >= expected; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        release_cv_.notify_all();
    }

private:
    void enterAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++entered_count_;
        entered_cv_.notify_all();
        release_cv_.wait(lock, [this] { return released_; });
    }

    Direction               direction_;
    std::mutex              mutex_;
    std::condition_variable entered_cv_;
    std::condition_variable release_cv_;
    size_t                  entered_count_{0};
    bool                    released_{false};
};

class FailSecondBatchDiskBlockIO: public RecordingBatchDiskBlockIO {
public:
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        if (batch_write_calls == 1) {
            status = DiskBlockIOStatus::IO_ERROR;
        }
        return RecordingBatchDiskBlockIO::write(writes);
    }
};

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

TEST_F(PerRankBlockTransferEngineHostDiskTest, MultipleDescriptorsUseOneVectorCallPerDirection) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(recording_io), "host_disk_batch", true);
    auto                     group = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    HostDiskTransferExecutor executor;

    const BlockIdxType first_host  = poolMalloc(*host_pool_);
    const BlockIdxType second_host = poolMalloc(*host_pool_);
    const BlockIdxType first_disk  = poolMalloc(*disk_pool);
    const BlockIdxType second_disk = poolMalloc(*disk_pool);
    ASSERT_NE(first_host, NULL_BLOCK_IDX);
    ASSERT_NE(second_host, NULL_BLOCK_IDX);
    ASSERT_NE(first_disk, NULL_BLOCK_IDX);
    ASSERT_NE(second_disk, NULL_BLOCK_IDX);

    const auto                        first_buffer  = host_pool_->blockBuffer(first_host);
    const auto                        second_buffer = host_pool_->blockBuffer(second_host);
    const std::vector<HostBufferView> hosts         = {
        {first_buffer.addr, first_buffer.payload_bytes, first_buffer.stride_bytes},
        {second_buffer.addr, second_buffer.payload_bytes, second_buffer.stride_bytes},
    };
    const std::vector<const GroupSet*>    groups = {group.get(), group.get()};
    const std::vector<TransferDescriptor> writes = {
        makeDescriptor(Tier::HOST, Tier::DISK, {}, first_host, first_disk),
        makeDescriptor(Tier::HOST, Tier::DISK, {}, second_host, second_disk),
    };

    EXPECT_EQ(executor.hostToDisk(hosts, writes, groups), TransferStatus::OK);
    EXPECT_EQ(io->batch_write_calls, 1u);
    EXPECT_EQ(io->single_write_calls, 0u);
    ASSERT_EQ(io->last_writes.size(), 2u);
    EXPECT_EQ(io->last_writes[0].offset, disk_pool->blockOffset(first_disk));
    EXPECT_EQ(io->last_writes[1].offset, disk_pool->blockOffset(second_disk));
    EXPECT_EQ(io->last_writes[0].buffer, first_buffer.addr);
    EXPECT_EQ(io->last_writes[1].buffer, second_buffer.addr);
    EXPECT_EQ(io->last_writes[0].bytes, disk_pool->strideBytes());
    EXPECT_EQ(io->last_writes[1].bytes, disk_pool->strideBytes());

    const std::vector<TransferDescriptor> reads = {
        makeDescriptor(Tier::DISK, Tier::HOST, {}, first_host, first_disk),
        makeDescriptor(Tier::DISK, Tier::HOST, {}, second_host, second_disk),
    };
    EXPECT_EQ(executor.diskToHost(reads, groups, hosts), TransferStatus::OK);
    EXPECT_EQ(io->batch_read_calls, 1u);
    EXPECT_EQ(io->single_read_calls, 0u);
    ASSERT_EQ(io->last_reads.size(), 2u);
    EXPECT_EQ(io->last_reads[0].offset, disk_pool->blockOffset(first_disk));
    EXPECT_EQ(io->last_reads[1].offset, disk_pool->blockOffset(second_disk));
    EXPECT_EQ(io->last_reads[0].buffer, first_buffer.addr);
    EXPECT_EQ(io->last_reads[1].buffer, second_buffer.addr);
    EXPECT_EQ(io->last_reads[0].bytes, disk_pool->strideBytes());
    EXPECT_EQ(io->last_reads[1].bytes, disk_pool->strideBytes());

    io->status = DiskBlockIOStatus::IO_ERROR;
    EXPECT_EQ(executor.hostToDisk(hosts, writes, groups), TransferStatus::DISK_IO_ERROR);

    host_pool_->free(first_host);
    host_pool_->free(second_host);
    disk_pool->free(first_disk);
    disk_pool->free(second_disk);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, BatchRejectsDuplicateTargetBeforeIo) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(recording_io), "duplicate_target", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType first_host  = poolMalloc(*host_pool_);
    const BlockIdxType second_host = poolMalloc(*host_pool_);
    const BlockIdxType disk_block  = poolMalloc(*disk_pool);
    ASSERT_NE(first_host, NULL_BLOCK_IDX);
    ASSERT_NE(second_host, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    const std::vector<TransferDescriptor> descriptors = {
        makeDescriptor(Tier::HOST, Tier::DISK, {}, first_host, disk_block),
        makeDescriptor(Tier::HOST, Tier::DISK, {}, second_host, disk_block),
    };
    const auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(io->batch_write_calls, 0u);
    EXPECT_EQ(io->single_write_calls, 0u);

    host_pool_->free(first_host);
    host_pool_->free(second_host);
    disk_pool->free(disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, BatchRejectsDuplicateHostTargetBeforeIo) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(recording_io), "duplicate_host_target", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType host_block  = poolMalloc(*host_pool_);
    const BlockIdxType first_disk  = poolMalloc(*disk_pool);
    const BlockIdxType second_disk = poolMalloc(*disk_pool);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_disk, NULL_BLOCK_IDX);
    ASSERT_NE(second_disk, NULL_BLOCK_IDX);

    const auto context = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, first_disk),
        makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, second_disk),
    });
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(io->batch_read_calls, 0u);
    EXPECT_EQ(io->single_read_calls, 0u);

    host_pool_->free(host_block);
    disk_pool->free(first_disk);
    disk_pool->free(second_disk);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, BatchAllowsSharedSourceWithDistinctTargets) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool = makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(recording_io), "shared_source", true);
    auto  group     = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto  engine    = makeEngine({group});

    const BlockIdxType host_block  = poolMalloc(*host_pool_);
    const BlockIdxType first_disk  = poolMalloc(*disk_pool);
    const BlockIdxType second_disk = poolMalloc(*disk_pool);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_disk, NULL_BLOCK_IDX);
    ASSERT_NE(second_disk, NULL_BLOCK_IDX);

    const auto context = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, first_disk),
        makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, second_disk),
    });
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(io->batch_write_calls, 1u);
    ASSERT_EQ(io->last_writes.size(), 2u);

    host_pool_->free(host_block);
    disk_pool->free(first_disk);
    disk_pool->free(second_disk);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitVectorSplitsDirectBatchAtConfiguredDescriptorLimit) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(recording_io), "host_disk_split", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4,
                                                               /*max_descriptors_per_transfer_batch=*/2);

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType>       disk_blocks;
    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 5; ++index) {
        const BlockIdxType disk_block = poolMalloc(*disk_pool);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        disk_blocks.push_back(disk_block);
        descriptors.push_back(makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block));
    }

    const auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(io->batch_write_calls, 3u);
    EXPECT_EQ(io->single_write_calls, 0u);
    EXPECT_EQ(io->batch_write_sizes, (std::vector<size_t>{2, 2, 1}));

    host_pool_->free(host_block);
    disk_pool->free(disk_blocks);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, DefaultDescriptorLimitSplitsSixtyFiveIntoSixtyFourAndOne) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 65, temp_dir_.path, std::move(recording_io), "host_disk_default_limit", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType>       disk_blocks;
    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 65; ++index) {
        const BlockIdxType disk_block = poolMalloc(*disk_pool);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        disk_blocks.push_back(disk_block);
        descriptors.push_back(makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block));
    }

    const auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(io->batch_write_sizes, (std::vector<size_t>{64, 1}));

    host_pool_->free(host_block);
    disk_pool->free(disk_blocks);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, InvalidDescriptorPreventsAnyVectorIo) {
    auto  recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io           = recording_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(recording_io), "host_disk_validate_all", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    const BlockIdxType disk_block = poolMalloc(*disk_pool);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    const std::vector<TransferDescriptor> descriptors = {
        makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block),
        makeDescriptor(Tier::HOST, Tier::DISK, {}, NULL_BLOCK_IDX, disk_block),
    };

    const auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(io->batch_write_calls, 0u);
    EXPECT_EQ(io->single_write_calls, 0u);

    host_pool_->free(host_block);
    disk_pool->free(disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitVectorReturnsBeforeDirectWorkerFinishes) {
    using namespace std::chrono_literals;

    auto  blocking_io = std::make_unique<BlockingBatchDiskBlockIO>();
    auto* io          = blocking_io.get();
    auto disk_pool = makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(blocking_io), "host_disk_async", true);
    auto group     = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine    = makeEngine({group});

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    const BlockIdxType disk_block = poolMalloc(*disk_pool);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    const std::vector<TransferDescriptor> descriptors = {
        makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block),
    };

    auto submit_future = std::async(std::launch::async, [&] { return engine->submit(descriptors); });
    ASSERT_TRUE(io->waitUntilEntered(2s));
    EXPECT_EQ(submit_future.wait_for(100ms), std::future_status::ready);

    io->release();
    const auto context = submit_future.get();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();

    host_pool_->free(host_block);
    disk_pool->free(disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, HostToDiskRunsTwoLogicalBatchesConcurrently) {
    using namespace std::chrono_literals;

    auto  barrier_io = std::make_unique<TwoCallBarrierDiskBlockIO>(TwoCallBarrierDiskBlockIO::Direction::WRITE);
    auto* io         = barrier_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(barrier_io), "host_to_disk_workers", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    const BlockIdxType first_disk_block  = poolMalloc(*disk_pool);
    const BlockIdxType second_disk_block = poolMalloc(*disk_pool);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_disk_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_disk_block, NULL_BLOCK_IDX);

    auto first = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::HOST, Tier::DISK, {}, first_host_block, first_disk_block)});
    ASSERT_TRUE(io->waitForCallCount(1, 5s));
    auto       second                      = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::HOST, Tier::DISK, {}, second_host_block, second_disk_block)});
    const bool both_entered_before_release = io->waitForCallCount(2, 1s);

    io->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(both_entered_before_release);
    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second->success()) << second->errorInfo().ToString();

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    disk_pool->free(first_disk_block);
    disk_pool->free(second_disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, DiskToHostRunsTwoLogicalBatchesConcurrently) {
    using namespace std::chrono_literals;

    auto  barrier_io = std::make_unique<TwoCallBarrierDiskBlockIO>(TwoCallBarrierDiskBlockIO::Direction::READ);
    auto* io         = barrier_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(barrier_io), "disk_to_host_workers", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    const BlockIdxType first_disk_block  = poolMalloc(*disk_pool);
    const BlockIdxType second_disk_block = poolMalloc(*disk_pool);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_disk_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_disk_block, NULL_BLOCK_IDX);

    auto first = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DISK, Tier::HOST, {}, first_host_block, first_disk_block)});
    ASSERT_TRUE(io->waitForCallCount(1, 5s));
    auto       second                      = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DISK, Tier::HOST, {}, second_host_block, second_disk_block)});
    const bool both_entered_before_release = io->waitForCallCount(2, 1s);

    io->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(both_entered_before_release);
    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second->success()) << second->errorInfo().ToString();

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    disk_pool->free(first_disk_block);
    disk_pool->free(second_disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, InFlightDiskWriteConflictIsRejectedAndReleasedAfterCompletion) {
    using namespace std::chrono_literals;

    auto  blocking_io = std::make_unique<BlockingBatchDiskBlockIO>();
    auto* io          = blocking_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(blocking_io), "host_disk_conflict", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    const BlockIdxType disk_block        = poolMalloc(*disk_pool);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    const auto first_descriptor  = makeDescriptor(Tier::HOST, Tier::DISK, {}, first_host_block, disk_block);
    const auto second_descriptor = makeDescriptor(Tier::HOST, Tier::DISK, {}, second_host_block, disk_block);
    auto       first             = engine->submit(std::vector<TransferDescriptor>{first_descriptor});
    ASSERT_TRUE(io->waitUntilEntered(2s));
    auto       second                          = engine->submit(second_descriptor);
    const bool second_completed_before_release = second->done();

    io->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second_completed_before_release);
    EXPECT_FALSE(second->success());
    EXPECT_NE(second->errorInfo().ToString().find("RESOURCE_EXHAUSTED: transfer endpoint conflict"), std::string::npos);

    auto retried = engine->submit(std::vector<TransferDescriptor>{second_descriptor});
    retried->waitDone();
    EXPECT_TRUE(retried->success()) << retried->errorInfo().ToString();

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    disk_pool->free(disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, FullDirectionQueueReturnsImmediateResourceExhausted) {
    using namespace std::chrono_literals;

    constexpr size_t endpoint_count = 1105;
    auto             host_pool      = makeHostPool(host_block_size_, endpoint_count, true);
    auto             blocking_io    = std::make_unique<BlockingBatchDiskBlockIO>();
    auto*            io             = blocking_io.get();
    auto             disk_pool      = makeDiskPool(
        host_block_size_, endpoint_count, temp_dir_.path, std::move(blocking_io), "host_disk_queue_full", true);
    auto group  = makeHostDiskGroup(0, host_pool, disk_pool, host_block_size_);
    auto engine = makeEngine({group});

    std::vector<BlockIdxType>       host_blocks;
    std::vector<BlockIdxType>       disk_blocks;
    std::vector<TransferDescriptor> descriptors;
    host_blocks.reserve(endpoint_count);
    disk_blocks.reserve(endpoint_count);
    descriptors.reserve(endpoint_count);
    for (size_t index = 0; index < endpoint_count; ++index) {
        const BlockIdxType host_block = poolMalloc(*host_pool);
        const BlockIdxType disk_block = poolMalloc(*disk_pool);
        ASSERT_NE(host_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        host_blocks.push_back(host_block);
        disk_blocks.push_back(disk_block);
        descriptors.push_back(makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block));
    }

    std::vector<std::shared_ptr<AsyncContext>> accepted_contexts;
    accepted_contexts.push_back(engine->submit(std::vector<TransferDescriptor>{descriptors.front()}));
    ASSERT_TRUE(io->waitUntilEntered(2s));

    std::shared_ptr<AsyncContext> rejected_context;
    size_t                        rejected_index = 0;
    for (size_t index = 1; index < descriptors.size(); ++index) {
        auto context = engine->submit(std::vector<TransferDescriptor>{descriptors[index]});
        if (context->done()) {
            rejected_context = std::move(context);
            rejected_index   = index;
            break;
        }
        accepted_contexts.push_back(std::move(context));
    }

    ASSERT_NE(rejected_context, nullptr);
    EXPECT_FALSE(rejected_context->success());
    EXPECT_NE(rejected_context->errorInfo().ToString().find("RESOURCE_EXHAUSTED"), std::string::npos);
    EXPECT_NE(rejected_context->errorInfo().ToString().find("transfer queue is full"), std::string::npos);

    io->release();
    for (const auto& context : accepted_contexts) {
        context->waitDone();
        EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    }

    auto retried = engine->submit(std::vector<TransferDescriptor>{descriptors[rejected_index]});
    retried->waitDone();
    EXPECT_TRUE(retried->success()) << retried->errorInfo().ToString();

    host_pool->free(host_blocks);
    disk_pool->free(disk_blocks);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, FailedPhysicalBatchFailsLogicalBatchAndStopsLaterBatches) {
    auto  failing_io = std::make_unique<FailSecondBatchDiskBlockIO>();
    auto* io         = failing_io.get();
    auto  disk_pool =
        makeDiskPool(host_block_size_, 7, temp_dir_.path, std::move(failing_io), "host_disk_fail_second", true);
    auto group  = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4,
                                                               /*max_descriptors_per_transfer_batch=*/2);

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType>       disk_blocks;
    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 5; ++index) {
        const BlockIdxType disk_block = poolMalloc(*disk_pool);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        disk_blocks.push_back(disk_block);
        descriptors.push_back(makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block));
    }

    const auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(io->batch_write_calls, 2u);
    EXPECT_EQ(io->batch_write_sizes, (std::vector<size_t>{2, 2}));
    EXPECT_NE(context->errorInfo().ToString().find("physical_sub_batch=[2,4)"), std::string::npos);
    EXPECT_NE(context->errorInfo().ToString().find("logical_descriptors=5"), std::string::npos);

    host_pool_->free(host_block);
    disk_pool->free(disk_blocks);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, BatchRejectsMixedDiskPoolsBeforeIo) {
    auto  first_recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* first_io           = first_recording_io.get();
    auto  first_disk_pool =
        makeDiskPool(host_block_size_, 4, temp_dir_.path, std::move(first_recording_io), "mixed_disk_first", true);
    auto  second_recording_io = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* second_io           = second_recording_io.get();
    auto  second_disk_pool =
        makeDiskPool(host_block_size_, 4, temp_dir_.path, std::move(second_recording_io), "mixed_disk_second", true);
    auto first_group  = makeHostDiskGroup(0, host_pool_, first_disk_pool, host_block_size_);
    auto second_group = makeHostDiskGroup(1, host_pool_, second_disk_pool, host_block_size_);
    auto engine       = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{first_group, second_group}, DeviceHostCopyOptions{});

    const BlockIdxType first_host  = poolMalloc(*host_pool_);
    const BlockIdxType second_host = poolMalloc(*host_pool_);
    const BlockIdxType first_disk  = poolMalloc(*first_disk_pool);
    const BlockIdxType second_disk = poolMalloc(*second_disk_pool);
    ASSERT_NE(first_host, NULL_BLOCK_IDX);
    ASSERT_NE(second_host, NULL_BLOCK_IDX);
    ASSERT_NE(first_disk, NULL_BLOCK_IDX);
    ASSERT_NE(second_disk, NULL_BLOCK_IDX);

    const std::vector<TransferDescriptor> descriptors = {
        makeDescriptor(Tier::HOST, Tier::DISK, {}, first_host, first_disk, 0),
        makeDescriptor(Tier::HOST, Tier::DISK, {}, second_host, second_disk, 1),
    };

    auto write_context = engine->submit(descriptors);
    write_context->waitDone();
    EXPECT_FALSE(write_context->success());
    EXPECT_EQ(first_io->batch_write_calls, 0u);
    EXPECT_EQ(second_io->batch_write_calls, 0u);
    EXPECT_EQ(first_io->single_write_calls, 0u);
    EXPECT_EQ(second_io->single_write_calls, 0u);

    const std::vector<TransferDescriptor> read_descriptors = {
        makeDescriptor(Tier::DISK, Tier::HOST, {}, first_host, first_disk, 0),
        makeDescriptor(Tier::DISK, Tier::HOST, {}, second_host, second_disk, 1),
    };
    auto read_context = engine->submit(read_descriptors);
    read_context->waitDone();
    EXPECT_FALSE(read_context->success());
    EXPECT_EQ(first_io->batch_read_calls, 0u);
    EXPECT_EQ(second_io->batch_read_calls, 0u);
    EXPECT_EQ(first_io->single_read_calls, 0u);
    EXPECT_EQ(second_io->single_read_calls, 0u);

    host_pool_->free(first_host);
    host_pool_->free(second_host);
    first_disk_pool->free(first_disk);
    second_disk_pool->free(second_disk);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, HostDiskDirectIoWritesAlignedStrideAndZeroPads) {
    auto  owned_io  = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto* direct_io = owned_io.get();
    auto  direct_disk =
        makeDiskPool(host_block_size_, 4, temp_dir_.path, std::move(owned_io), "host_disk_direct", false);
    auto                     group = makeHostDiskGroup(0, host_pool_, direct_disk, host_block_size_);
    HostDiskTransferExecutor executor;
    const size_t stride = direct_disk->strideBytes();
    ASSERT_GT(stride, host_block_size_);
    EXPECT_FALSE(direct_io->bufferedIo());

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    // Seed non-zero padding to verify that the executor clears it.
    for (size_t i = 0; i < stride; ++i) {
        host_data[i] = static_cast<uint8_t>((i * 7 + 1) & 0xFF);
    }

    const auto disk_slot = poolMalloc(*direct_disk);
    ASSERT_NE(disk_slot, NULL_BLOCK_IDX);
    ASSERT_EQ(executor.execute(HostBufferView{host_data, host_block_size_, stride},
                               TransferDescriptor::deviceToDisk(0, {0}, disk_slot),
                               *group),
              TransferStatus::OK);
    EXPECT_EQ(direct_io->lastWriteBytes(), stride);

    const BlockIdxType dst_block = poolMalloc(*host_pool_);
    ASSERT_NE(dst_block, NULL_BLOCK_IDX);
    uint8_t* dst_data = static_cast<uint8_t*>(host_pool_->blockBuffer(dst_block).addr);
    std::memset(dst_data, 0xAB, stride);
    ASSERT_EQ(executor.execute(HostBufferView{dst_data, host_block_size_, stride},
                               TransferDescriptor::diskToDevice(0, disk_slot, {0}),
                               *group),
              TransferStatus::OK);
    EXPECT_EQ(direct_io->lastReadBytes(), stride);

    for (size_t i = 0; i < host_block_size_; ++i) {
        EXPECT_EQ(dst_data[i], static_cast<uint8_t>((i * 7 + 1) & 0xFF)) << "payload byte " << i;
    }
    for (size_t i = host_block_size_; i < stride; ++i) {
        EXPECT_EQ(dst_data[i], 0) << "padding byte " << i;
    }

    host_pool_->free(host_block);
    host_pool_->free(dst_block);
    direct_disk->free(disk_slot);
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
    auto topology              = makeTestTopology({makeTestGroupBase(policy, {0, 1, 2}, 160, 40)});
    auto pool                  = makeTestDevicePool({{200, 40}, {220, 40}, {240, 40}}, 2, "group_set_payload_test");
    auto group                 = makeTestGroupSet(0, std::move(topology), {0}, {std::move(pool)});
    EXPECT_EQ(group->payloadBytes(), 600u);
}

}  // namespace
}  // namespace rtp_llm
